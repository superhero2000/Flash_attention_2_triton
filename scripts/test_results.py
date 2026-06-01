# In this file, we will test the results of the attention triton attention implementation
# with the results of the naive attention implementation

import torch
from flash_attention_triton import FlashAttentionTriton
from torch.nn.functional import scaled_dot_product_attention
import itertools
import pandas as pd


def test_attention_triton(batch_size, sequence_length, model_dimension, precision, is_causal, device):
    
    Q = torch.randn(batch_size, sequence_length, model_dimension, device=device, dtype=precision, requires_grad=True)
    K = torch.randn(batch_size, sequence_length, model_dimension, device=device, dtype=precision, requires_grad=True)
    V = torch.randn(batch_size, sequence_length, model_dimension, device=device, dtype=precision, requires_grad=True)

    O_triton = FlashAttentionTriton.apply(Q, K, V, is_causal)
    O_triton.sum().backward()
    # Cloning the gradients like this is correct. By using .clone(), you ensure that you store a copy of the gradient
    # values before they are zeroed out or modified elsewhere. This is the standard way to "save" the gradient tensors
    # for later comparison.
    K_grad_triton = K.grad.clone()
    V_grad_triton = V.grad.clone()
    Q_grad_triton = Q.grad.clone()

    K.grad.zero_()
    V.grad.zero_()
    Q.grad.zero_()
    
    O_naive = scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
    O_naive.sum().backward()
    K_grad_naive = K.grad.clone()
    V_grad_naive = V.grad.clone()
    Q_grad_naive = Q.grad.clone()

    return O_triton, O_naive, K_grad_triton, K_grad_naive, V_grad_triton, V_grad_naive, Q_grad_triton, Q_grad_naive


if __name__ == "__main__":
    batch_size = [1, 3, 5]
    sequence_length = [256, 512, 1024, 2048, 4096, 8192]
    model_dimension = [32, 128]
    precision = [torch.bfloat16, torch.float16, torch.float32]
    is_causal = [True, False]
    device = ["cuda"]
    cartesian_product = list(itertools.product(batch_size, sequence_length, model_dimension, precision, is_causal, device))
    df = pd.DataFrame(columns=["batch_size", "sequence_length", "model_dimension", "precision", "is_causal", "device", "O_diff_l2", "O_diff_max", "K_grad_diff_l2_rel", "K_grad_diff_max", "V_grad_diff_l2_rel", "V_grad_diff_max", "Q_grad_diff_l2_rel", "Q_grad_diff_max"])
    for batch_size, sequence_length, model_dimension, precision, is_causal, device in cartesian_product:
        O_triton, O_naive, K_grad_triton, K_grad_naive, V_grad_triton, V_grad_naive, Q_grad_triton, Q_grad_naive = test_attention_triton(batch_size, sequence_length, model_dimension, precision, is_causal, device)
        O_diff_l2 = torch.norm(O_triton - O_naive).item()
        O_diff_max = torch.max(torch.abs(O_triton - O_naive)).item()
        K_grad_diff_l2_rel = (torch.norm(K_grad_triton - K_grad_naive) / (torch.norm(K_grad_naive) + 1e-8)).item()
        K_grad_diff_max = torch.max(torch.abs(K_grad_triton - K_grad_naive)).item()
        V_grad_diff_l2_rel = (torch.norm(V_grad_triton - V_grad_naive) / (torch.norm(V_grad_naive) + 1e-8)).item()
        V_grad_diff_max = torch.max(torch.abs(V_grad_triton - V_grad_naive)).item()
        Q_grad_diff_l2_rel = (torch.norm(Q_grad_triton - Q_grad_naive) / (torch.norm(Q_grad_naive) + 1e-8)).item()
        Q_grad_diff_max = torch.max(torch.abs(Q_grad_triton - Q_grad_naive)).item()
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}, Model dimension: {model_dimension}, Precision: {precision}, Is causal: {is_causal}, Device: {device}")
        print(f"O_diff_l2: {O_diff_l2}, O_diff_max: {O_diff_max}")
        print(f"K_grad_diff_l2_rel: {K_grad_diff_l2_rel}, K_grad_diff_max: {K_grad_diff_max}")
        print(f"V_grad_diff_l2_rel: {V_grad_diff_l2_rel}, V_grad_diff_max: {V_grad_diff_max}")
        print(f"Q_grad_diff_l2_rel: {Q_grad_diff_l2_rel}, Q_grad_diff_max: {Q_grad_diff_max}")
        print(f"K_grad_triton: {K_grad_triton}")
        print(f"K_grad_naive: {K_grad_naive}")
        print(f"V_grad_triton: {V_grad_triton}")
        print(f"V_grad_naive: {V_grad_naive}")
        print(f"Q_grad_triton: {Q_grad_triton}")
        print(f"Q_grad_naive: {Q_grad_naive}")
        print("-" * 80)
        df.loc[len(df)] = [batch_size, sequence_length, model_dimension, precision, is_causal, device, O_diff_l2, O_diff_max, K_grad_diff_l2_rel, K_grad_diff_max, V_grad_diff_l2_rel, V_grad_diff_max, Q_grad_diff_l2_rel, Q_grad_diff_max]
    df.to_csv("correctness_results.csv", index=False)