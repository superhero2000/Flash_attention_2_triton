# Triton Flash Attention 2 Implementation

When the classical attention mechanism is implemented naively in PyTorch, its memory usage is very high. Given the Query, Key, and Value matrices, a standard forward pass through the attention block first computes the raw, unnormalized attention scores, then applies the softmax function to those scores (with masking in the causal case). The final output is obtained by multiplying the normalized scores with the Value matrix. To perform backpropagation, we need values from both the unnormalized and normalized attention matrices, each of size `sequence_length x sequence_length`. As a result, the memory footprint grows quadratically with the sequence length. Flash Attention reduces this memory usage by computing the attention scores block by block, which lowers the memory footprint to `O(sequence_length * model_dimension)`.

# What This Project Does

I implemented Flash Attention 2 in Triton to gain a deeper understanding of the technique and to measure how it performs against a naive implementation.

- `src/triton_flash_attention_2/flash_attention_triton.py` — the Flash Attention implementation in Triton.
- `scripts/attention_implementations_comparison.py` — a script that benchmarks the Triton implementation against the naive one.
- `scripts/test_results.py` — a script that verifies the correctness of the Triton implementation against the naive one.
- `notebooks/visualizations.ipynb` — a notebook that visualizes the benchmark results.

# Results

The comparison was performed for single-head attention only. I timed one complete forward pass (without tracking gradients internally, to mimic inference) and one complete forward-and-backward pass, for batch sizes of 1 and 3.

Two main observations stand out. First, the Triton kernel is faster than the naive implementation across all model dimensions and precisions. Second, the Triton kernel never runs out of memory, whereas both the naive implementation and PyTorch's scaled dot-product attention do.

# Conclusion

The Triton kernel is a very fast and memory-efficient implementation of Flash Attention; it does not run out of memory for any of the tested model dimensions and precisions. The naive implementation is also fast, but it is memory-inefficient and runs out of memory in some configurations. PyTorch's scaled dot-product attention is the slowest and likewise runs out of memory in some configurations.

# Future Work

This project already demonstrates the advantages of Flash Attention well. Here, only a single head of size `model_dimension` was considered. In practice, multi-head attention is used, and it increases memory usage linearly with the number of heads, since each head requires computing its own attention matrix. This case has already been implemented and studied in the original Flash Attention paper.

# Links

Original Flash Attention 2 paper: https://arxiv.org/abs/2307.08691
