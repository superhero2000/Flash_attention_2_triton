# Triton Flash Attention 2 implementation

When using the classical attention mechanism in pytorch in the naive way, the memory usage is very high. A classical forward pass in the attention block given the Query, Key and Value matrix is done by first calculating the raw, unnormalized attention scores , then applying the softmax function to the attention scores (with masking in the causal case). After that we get the final output value by multiplying the attention scores with the Value matrix. In order to perform backpropagation, we have to use values from the unnormalized and normalized attention matrix, which is of size sequence_length x sequence_length. We can see, that the memory footprint grows quadratically with the sequence length. Flash attention is a technique to reduce the memory usage of the attention mechanism by using a block-wise approach to compute the attention scores. This technique results in a memory footprint of O(sequence_length \* model_dimension).

# What has been done in this project

I implemented Flash attention 2 in triton, to gain a better understanding of the technique and to see how it performs compared to the naive implementation. In src/triton_flash_attention_2/flash_attention_triton.py you can find the implementation of the flash attention mechanism in triton. In scripts/attention_implementations_comparison.py you can find the script to benchmark the performance of the flash attention mechanism in triton compared to the naive implementation. In scripts/test_results.py you can find the script to test the results of the flash attention mechanism in triton compared to the naive implementation. In notebooks/visualizations.ipynb you can find the notebook to visualize the results of the benchmark.

# Results

The comparison was done only for singleheadattention. I timed one complete forward pass (without tracking the gradiends internally, mimiking inference) and one complete forward and backward pass for a batch size of 1 and 3.
First of all one can see, that the triton kernel is faster than the naive implementation for all model dimensions and precisions. Furthermore the triton kernel never runs out of memory, while the naive and pytorch scaled dot product attention implementation are doing this.

# Conclusion

The triton kernel is a very fast implementation of the flash attention mechanism. It is also memory efficient and does not run out of memory for any model dimensions and precisions. The naive implementation is also very fast, but it is memory inefficient and runs out of memory for some model dimensions and precisions. The pytorch scaled dot product attention implementation is the slowest and runs out of memory for some model dimensions and precisions.

# Future work

This project already demonstrates the advantages of flash attention pretty well. In this repository only a single head of size model_dimension was considered. In practice, multiheadattention is used. Multihead attention also increases memory usage linearly with the number of heads, since for each new head, one must compute a new attention matrix. But this implementation has been done and studied in the original flash attention paper.

# Links

Original flash attention 2 paper: https://arxiv.org/abs/2307.08691
