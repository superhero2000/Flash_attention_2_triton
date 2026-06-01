# In this scrip we will perform basic end to end benchmarking of the model

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
import numpy as np
import torch
import time
import argparse
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import get_cosine_lr, AdamW
import matplotlib.pyplot as plt


def create_random_data_batch_from_args(args):
    return create_random_data_batch(args.vocab_size, args.context_length, args.batch_size, args.device)


def create_random_data_batch(vocab_size, context_length, batch_size, device):
    random_int_array = np.random.randint(0, vocab_size, size=(1000000,))
    x, y = get_batch(random_int_array, batch_size, context_length, device)
    return x, y

def benchmark_model(model, x, y, optimizer, warmup_steps, timing_steps):
    for step in range(warmup_steps):
        optimizer.zero_grad()
        
        torch.cuda.nvtx.range_push(f"Warmup Step {step} - Forward Pass") 
        logits = model(x)
        loss = cross_entropy(logits, y)
        torch.cuda.nvtx.range_pop() 
        
        torch.cuda.nvtx.range_push(f"Warmup Step {step} - Backward Pass")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push(f"Warmup Optimizer Step {step}")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        
        print(f"Warmup step {step} completed, loss currently: {loss.item()}")
    
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for step in range(timing_steps):
        # optimizer.zero_grad()
        # torch.cuda.nvtx.range_push("Forward Pass")
        # with torch.no_grad():
        #     with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        #         logits = model(x)
        # torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()
        
        # model.zero_grad()
        # torch.cuda.nvtx.range_push("Forward and Backward Pass")
        # with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        #     logits = model(x)
        #     loss = cross_entropy(logits, y)
        #     loss.backward()
        # torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()
       
        model.zero_grad()
        torch.cuda.nvtx.range_push("Forward and Backward and Step")
        #with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop() 
        
        
        print(f"Timing step {step} completed, loss currently: {loss.item()}")

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    memory_history = torch.cuda.memory._record_memory_history(enabled = None)
def create_model_from_args(args):
    return BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )


def create_optimizer_from_args(model, args):
    return AdamW(model.parameters(), lr=args.learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for the BasicsTransformerLM model")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--timing_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_context_lengths", type=int, default=512*4)
    args = parser.parse_args()

    model = create_model_from_args(args)
    model.to(args.device)
    torch.compile(model)
    
    optimizer = create_optimizer_from_args(model, args)
    x, y = create_random_data_batch_from_args(args)
    benchmark_model(model, x, y, optimizer, args.warmup_steps, args.timing_steps)
