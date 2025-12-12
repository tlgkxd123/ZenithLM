"""
Training script for Titans Language Model.
Optimized for Multi-GPU (DDP) on H100/H200 clusters.

Features:
- Distributed Data Parallel (DDP)
- Flash Attention 2 + BFloat16
- torch.compile support
- Streaming dataset sharding
"""

import os
import math
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
from typing import Optional, Dict, Iterator, Any
from itertools import islice
import argparse
from tqdm import tqdm
import time

# Use TF32 on Ampere/Hopper
torch.set_float32_matmul_precision('high')

from titans import TitansLM, TitansConfig

def setup_distributed():
    """Initialize DDP."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_tokenizer(name: str = "gpt2"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_dataset_no_remote_code(*args, dataset_name_for_error: str, **kwargs):
    from datasets import load_dataset
    try:
        return load_dataset(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "trust_remote_code" in msg or "remote code" in msg:
            print(f"Warning: {dataset_name_for_error} requires remote code. trying to proceed...")
            raise e
        raise

class StreamingTextDataset(IterableDataset):
    """DDP-aware streaming dataset."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        tokenizer,
        seq_len: int = 1024,
        split: str = "train",
        text_column: str = "text",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.text_column = text_column
        self.rank = rank
        self.world_size = world_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        load_kwargs = {"split": self.split, "streaming": True}
        if self.dataset_config:
            load_kwargs["name"] = self.dataset_config
            
        dataset = load_dataset_no_remote_code(
            self.dataset_name, **load_kwargs, dataset_name_for_error=self.dataset_name
        )
        
        # 1. Shard by process (DDP)
        # 2. Shard by worker (DataLoader workers)
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker
            iter_start = self.rank
            iter_step = self.world_size
        else:
            # Multiple workers: global stride = world_size * num_workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_start = self.rank * num_workers + worker_id
            iter_step = self.world_size * num_workers
            
        iterator = islice(dataset, iter_start, None, iter_step)
        
        buffer = []
        for example in iterator:
            text = example.get(self.text_column, "")
            if not text:
                continue
                
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long)
                }

class TitansTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        lr: float,
        max_steps: int,
        grad_accum_steps: int,
        save_dir: str,
        log_every: int,
        rank: int,
    ):
        self.model = model
        self.train_loader = train_loader
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        try:
            import bitsandbytes as bnb
            print("Using 8-bit AdamW optimizer")
            self.optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
        except ImportError:
            print("bitsandbytes not found, using standard AdamW")
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True
            )
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, max_steps)
        
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.rank = rank
        self.save_dir = save_dir
        self.log_every = log_every
        
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            
    def train(self):
        self.model.train()
        # Enable gradient checkpointing on the backbone specifically
        if hasattr(self.model, "module"):
            backbone = self.model.module.backbone
        else:
            backbone = self.model.backbone
            
        # Enable gradient checkpointing if available
        if hasattr(backbone, "gradient_checkpointing_enable"):
             backbone.gradient_checkpointing_enable()
             
        step = 0
        accum_loss = torch.tensor(0.0, device="cuda")
        
        data_iter = iter(self.train_loader)
        if self.rank == 0:
            pbar = tqdm(total=self.max_steps, desc="Training")
        
        start_time = time.time()
        
        while step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            
            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                
                # BFloat16 context
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"] / self.grad_accum_steps
                
                loss.backward()
                accum_loss += loss.detach()
            
            # Clip grad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            step += 1
            
            # Logging
            if step % self.log_every == 0:
                # Reduce loss across GPUs for logging
                if dist.is_initialized():
                    dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)
                
                if self.rank == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    tps = (self.log_every * self.grad_accum_steps * 
                           batch["input_ids"].shape[0] * batch["input_ids"].shape[1] * 
                           (dist.get_world_size() if dist.is_initialized() else 1)) / (time.time() - start_time)
                    
                    pbar.set_postfix({"loss": f"{accum_loss.item():.4f}", "tok/s": f"{int(tps):,}"})
                    pbar.update(self.log_every)
                    start_time = time.time()
                
                accum_loss.fill_(0.0)
            
            if step % 1000 == 0 and self.rank == 0:
                self.save_checkpoint(step)

        if self.rank == 0:
            self.save_checkpoint(step, final=True)
            pbar.close()

    def save_checkpoint(self, step, final=False):
        # Unwrap DDP
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # Unwrap Compile
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
            
        path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        torch.save(model_to_save.state_dict(), path)
        print(f"\nSaved checkpoint {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="mag")
    parser.add_argument("--dataset", default="fineweb")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--quantize", type=str, default="none", 
                        choices=["none", "nf4", "fp4"],
                        help="4-bit quantization: nf4, fp4, or none")
    args = parser.parse_args()

    # Dist setup
    rank, local_rank, world_size = setup_distributed()
    if rank == 0:
        print(f"🚀 Training on {world_size} GPUs (Rank 0)")

    # Data
    tokenizer = get_tokenizer()
    ds_config = "sample-10BT" if args.dataset == "fineweb" else None
    dataset_name = "HuggingFaceFW/fineweb-edu" if args.dataset == "fineweb" else args.dataset
    
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        dataset_config=ds_config,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        rank=rank,
        world_size=world_size
    )
    
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True
    )
    
    # Model config
    config = TitansConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.d_model // 64,
        variant=args.variant,
        max_seq_len=args.seq_len
    )
    
    # Create model (quantized or full precision)
    if args.quantize != "none":
        if rank == 0:
            print(f"🔢 Using {args.quantize.upper()} 4-bit quantization")
        from titans.quantization import create_quantized_model
        model = create_quantized_model(config, args.quantize)
    else:
        model = TitansLM(config).cuda()
    
    # Compile (optional)
    if args.compile:
        if rank == 0: print("Compiling model...")
        model = torch.compile(model)
    else:
        if rank == 0: print("Skipping torch.compile (use --compile to enable)")
    
    # DDP Wrapper
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
        
    trainer = TitansTrainer(
        model, train_loader, args.lr, args.max_steps, 
        args.grad_accum, "checkpoints", 50, rank
    )
    
    trainer.train()
    cleanup_distributed()

if __name__ == "__main__":
    main()

