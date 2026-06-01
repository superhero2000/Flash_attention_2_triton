import torch
from triton_flash_attention_2 import FlashAttentionTriton
import pandas as pd
import itertools
import numpy as np
import time
import gc

WARMUP_ITERATIONS = 10
TIMING_ITERATIONS = 20
device = "cuda"


def benchmark_triton_kernel(
    sequence_length, model_dimension, precision, is_causal, Q, K, V
):
    torch.cuda.empty_cache()
    gc.collect()
    forward_times = []
    backward_times = []
    attention = FlashAttentionTriton.apply
    for i in range(WARMUP_ITERATIONS):
        O = attention(Q, K, V, is_causal)
        O.sum().backward()
    torch.cuda.synchronize()

    for i in range(TIMING_ITERATIONS):
        start_time = time.time()
        # Forward pass without gradient
        with torch.no_grad():
            O = FlashAttentionTriton.apply(Q, K, V, is_causal)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)

    for i in range(TIMING_ITERATIONS):
        start_time = time.time()
        O = attention(Q, K, V, is_causal)
        O.sum().backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start_time)
    return np.mean(forward_times), np.mean(backward_times)


def pytorch_naive_attention(Q, K, V, is_causal):
    S = torch.matmul(Q, K.transpose(-2, -1))
    if is_causal:
        q_len, k_len = S.shape[-2], S.shape[-1]
        mask = torch.tril(torch.ones((q_len, k_len), device=Q.device, dtype=torch.bool))
        S = S.masked_fill(~mask, float("-inf"))
    S = torch.softmax(S, dim=-1)
    return torch.matmul(S, V)


def benchmark_pytorch_naive_attention(
    sequence_length, model_dimension, precision, is_causal, Q, K, V
):
    torch.cuda.empty_cache()
    gc.collect()
    forward_times = []
    backward_times = []
    model = pytorch_naive_attention
    # We will compile the model to improve performance
    model = torch.compile(model)

    for i in range(WARMUP_ITERATIONS):
        with torch.autocast(device_type=device, dtype=precision):
            O = model(Q, K, V, is_causal)
        O.sum().backward()
        Q.grad.zero_()  # Zero out the gradients for the next iteration
        K.grad.zero_()
        V.grad.zero_()
    torch.cuda.synchronize()

    for i in range(TIMING_ITERATIONS):
        start_time = time.time()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=precision):
            O = model(Q, K, V, is_causal)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)

    for i in range(TIMING_ITERATIONS):
        start_time = time.time()
        with torch.autocast(device_type=device, dtype=precision):
            O = model(Q, K, V, is_causal)
        O.sum().backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start_time)
        Q.grad.zero_()  # Zero out the gradients for the next iteration
        K.grad.zero_()
        V.grad.zero_()
    return np.mean(forward_times), np.mean(backward_times)


def benchmark_pytorch_scaled_dot_product_attention(
    sequence_length, model_dimension, precision, is_causal, Q, K, V
):
    torch.cuda.empty_cache()
    gc.collect()
    forward_times = []
    backward_times = []
    model = torch.nn.functional.scaled_dot_product_attention
    # We will compile the model to improve performance
    model = torch.compile(model)
    for i in range(WARMUP_ITERATIONS):
        with torch.autocast(device_type=device, dtype=precision):
            O = model(Q, K, V, is_causal=is_causal)
        O.sum().backward()
        Q.grad.zero_()  # Zero out the gradients for the next iteration
        K.grad.zero_()
        V.grad.zero_()
    torch.cuda.synchronize()
    for i in range(TIMING_ITERATIONS):
        start_time = time.time()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=precision):
            O = model(Q, K, V, is_causal=is_causal)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
    for i in range(TIMING_ITERATIONS):
        start_time = time.time()
        with torch.autocast(device_type=device, dtype=precision):
            O = model(Q, K, V, is_causal=is_causal)
        O.sum().backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start_time)
        Q.grad.zero_()  # Zero out the gradients for the next iteration
        K.grad.zero_()
        V.grad.zero_()
    return np.mean(forward_times), np.mean(backward_times)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    df = pd.DataFrame(
        columns=[
            "sequence_length",
            "model_dimension",
            "precision",
            "forward_time",
            "backward_time",
            "implementation",
        ]
    )
    batch_size = 1
    is_causal = True
    sequence_lengths = []
    val = 128
    while val <= 65536:
        sequence_lengths.append(val)
        val *= 2
    model_dimensions = [16, 32, 64, 128, 256]
    precisions = [torch.bfloat16, torch.float16, torch.float32]
    cartesian_product = list(
        itertools.product(sequence_lengths, model_dimensions, precisions)
    )
    print("Cartesian product of sequence lengths, model dimensions, and precisions:")
for sequence_length, model_dimension, precision in cartesian_product:
    print(
        f"Sequence length: {sequence_length}, Model dimension: {model_dimension}, Precision: {precision}"
    )
    Q = torch.randn(
        batch_size,
        sequence_length,
        model_dimension,
        device="cuda",
        dtype=precision,
        requires_grad=True,
    )
    K = torch.randn(
        batch_size,
        sequence_length,
        model_dimension,
        device="cuda",
        dtype=precision,
        requires_grad=True,
    )
    V = torch.randn(
        batch_size,
        sequence_length,
        model_dimension,
        device="cuda",
        dtype=precision,
        requires_grad=True,
    )

    try:
        # benchmark the triton kernel
        forward_time, backward_time = benchmark_triton_kernel(
            sequence_length, model_dimension, precision, is_causal, Q, K, V
        )
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            forward_time,
            backward_time,
            "triton_kernel",
        ]
    except torch.cuda.OutOfMemoryError:
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            forward_time,
            backward_time,
            "triton_kernel",
        ]
        print(f"Error benchmarking triton kernel: {e}")
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            None,
            None,
            "triton_kernel",
        ]
        continue
    try:
        # benchmark the pytorch kernel
        forward_time, backward_time = benchmark_pytorch_naive_attention(
            sequence_length, model_dimension, precision, is_causal, Q, K, V
        )
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            forward_time,
            backward_time,
            "pytorch_naive",
        ]
    except torch.cuda.OutOfMemoryError as e:
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            None,
            None,
            "pytorch_naive",
        ]
        print(f"Error benchmarking pytorch naive kernel: {e}")
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            None,
            None,
            "pytorch_naive",
        ]
        continue
    try:
        # benchmark the pytorch scaled dot product attention kernel
        forward_time, backward_time = benchmark_pytorch_scaled_dot_product_attention(
            sequence_length, model_dimension, precision, is_causal, Q, K, V
        )
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            forward_time,
            backward_time,
            "pytorch_scaled_dot_product_attention",
        ]
    except torch.cuda.OutOfMemoryError as e:
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            None,
            None,
            "pytorch_scaled_dot_product_attention",
        ]
        print(f"Error benchmarking pytorch scaled dot product attention kernel: {e}")
        df.loc[len(df)] = [
            sequence_length,
            model_dimension,
            precision,
            None,
            None,
            "pytorch_scaled_dot_product_attention",
        ]
        continue
    # benchmark the triton kernel
df.to_csv(
    f"attention_implementations_comparison_batch_size_{batch_size}_is_causal_{is_causal}.csv",
    index=False,
)
