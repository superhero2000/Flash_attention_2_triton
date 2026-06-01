import triton
import triton.language as tl
import torch

import einops

GLOBAL_Q_TILE_SIZE = 16
GLOBAL_K_TILE_SIZE = 16

assert GLOBAL_Q_TILE_SIZE == GLOBAL_K_TILE_SIZE, (
    "Q_TILE_SIZE and K_TILE_SIZE must be the same"
)


@triton.jit
def compute_attention_block(
    Q_block, K_block, V_block, O_block, m_old, l_sum_block, scale
):
    # Compute scaled dot product attention scores
    S = tl.dot(Q_block, tl.trans(K_block)) * scale

    # Update maximum for numerical stability
    m_new = tl.maximum(m_old, tl.max(S, axis=1))

    # Compute exponentiated scores for softmax numerator
    P = tl.exp(S - m_new[:, None])

    # Update running sum for softmax denominator
    l_sum_block = l_sum_block * tl.exp(m_old - m_new) + tl.sum(P, axis=1)

    # Scale previous output and accumulate new weighted values
    O_block = O_block * tl.exp((m_old - m_new)[:, None])
    O_block += tl.dot(tl.cast(P, V_block.dtype), V_block)

    return O_block, m_new, l_sum_block


@triton.jit
def flash_attention_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    NUM_KEY_TILES: tl.constexpr,
) -> None:
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    shape = (N_QUERIES, D)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # Query block pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # set the beginning of the block to
        shape=(N_QUERIES, D),  # total numbers of N_query queries and D_model elements
        strides=(
            stride_qq,
            stride_qd,
        ),  # to jump from one query to the next, we need to jump stride_qq elements, and to jump from one D_model element to the next, we need to jump stride_qd elements
        offsets=(
            query_tile_index * Q_TILE_SIZE,
            0,
        ),  # set the starting point of the block to the query_tile_index * Q_TILE_SIZE element, and 0 for the D_model elements
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Value block pointer
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(
            0,
            0,
        ),  # we start at the beginning of the block, we don't need to offset by the tile index because we are loading a whole block
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Output block pointer
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Log-sum-exp block pointer
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Value block pointer
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # we start at the beginning of the block
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    m_old = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_sum_block = tl.full((Q_TILE_SIZE,), 0.0, dtype=tl.float32)

    O_block = tl.load(O_block_ptr, boundary_check=(0,), padding_option="zero")
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")

    if not is_causal:
        for i in range(NUM_KEY_TILES):
            K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

            # Compute the attention scores and then apply the scale factor
            O_block, m_new, l_sum_block = compute_attention_block(
                Q_block, K_block, V_block, O_block, m_old, l_sum_block, scale
            )

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

            m_old = m_new

        O_block /= l_sum_block[:, None]
        L_block += m_old + tl.log(l_sum_block)
        tl.store(O_block_ptr, O_block, boundary_check=(0))
        tl.store(L_block_ptr, L_block, boundary_check=(0))

    else:
        for i in range(query_tile_index):
            K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

            # Compute the attention scores and then apply the scale factor
            O_block, m_new, l_sum_block = compute_attention_block(
                Q_block, K_block, V_block, O_block, m_old, l_sum_block, scale
            )

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

            m_old = m_new

        K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        # Compute the attention scores and then apply the scale factor
        S = tl.dot(Q_block, tl.trans(K_block))
        S = S * scale
        column_range = tl.arange(0, K_TILE_SIZE) + query_tile_index * K_TILE_SIZE
        row_range = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
        mask = column_range[None, :] <= row_range[:, None]
        S = tl.where(mask, S, -float("inf"))
        # get the current max block
        m_new = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, m_new)

        # apply the softmax function
        P = tl.exp(S - m_new[:, None])  # apply the softmax function

        # modify the running sum block
        l_sum_block *= tl.exp(m_old - m_new)
        l_sum_block += tl.sum(P, axis=1)

        # apply the weighted sum operation
        O_block *= tl.exp((m_old - m_new)[:, None])
        P = tl.cast(P, V_block.dtype)
        O_block += tl.dot(P, V_block)

        # calculate the final output
        O_block /= l_sum_block[:, None]
        # Use m_new (final row max), not m_old: for the first causal tile m_old is -inf.
        L_block += m_new + tl.log(l_sum_block)
        tl.store(O_block_ptr, O_block, boundary_check=(0))
        tl.store(L_block_ptr, L_block, boundary_check=(0))


@triton.jit
def flash_attention_backward_K_V_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    d_K_ptr,
    d_V_ptr,
    d_O_ptr,
    D_ptr,
    stride_Q_batch,
    stride_Q_row,
    stride_Q_col,
    stride_K_batch,
    stride_K_row,
    stride_K_col,
    stride_V_batch,
    stride_V_row,
    stride_V_col,
    stride_L_batch,
    stride_L_row,
    stride_dK_batch,
    stride_dK_row,
    stride_dK_col,
    stride_dV_batch,
    stride_dV_row,
    stride_dV_col,
    stride_dO_batch,
    stride_dO_row,
    stride_dO_col,
    stride_D_batch,
    stride_D_row,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    NUM_KEY_TILES: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
) -> None:
    column_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    shape = (N_KEYS, D)

    # Key block pointer
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_K_batch,
        shape=(N_KEYS, D),
        strides=(stride_K_row, stride_K_col),
        offsets=(column_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_V_batch,
        shape=(N_KEYS, D),
        strides=(stride_V_row, stride_V_col),
        offsets=(column_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    d_K_block_ptr = tl.make_block_ptr(
        d_K_ptr + batch_index * stride_dK_batch,
        shape=(N_KEYS, D),
        strides=(stride_dK_row, stride_dK_col),
        offsets=(column_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    d_V_block_ptr = tl.make_block_ptr(
        d_V_ptr + batch_index * stride_dV_batch,
        shape=(N_KEYS, D),
        strides=(stride_dV_row, stride_dV_col),
        offsets=(column_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_block_pointer = tl.make_block_ptr(
        Q_ptr + batch_index * stride_Q_batch,
        shape=(N_QUERIES, D),
        strides=(stride_Q_row, stride_Q_col),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    d_O_block_pointer = tl.make_block_ptr(
        d_O_ptr + batch_index * stride_dO_batch,
        shape=(N_QUERIES, D),
        strides=(stride_dO_row, stride_dO_col),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_pointer = tl.make_block_ptr(
        L_ptr + batch_index * stride_L_batch,
        shape=(N_QUERIES,),
        strides=(stride_L_row,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_pointer = tl.make_block_ptr(
        D_ptr + batch_index * stride_D_batch,
        shape=(N_QUERIES,),
        strides=(stride_D_row,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    d_K_block = tl.load(d_K_block_ptr, boundary_check=(0,), padding_option="zero")
    d_V_block = tl.load(d_V_block_ptr, boundary_check=(0,), padding_option="zero")
    if not is_causal:
        for i in range(NUM_QUERY_TILES):
            # load the necessary blocks needed for the computation
            d_O_block = tl.load(
                d_O_block_pointer, boundary_check=(0,), padding_option="zero"
            )
            Q_block = tl.load(
                Q_block_pointer, boundary_check=(0,), padding_option="zero"
            )
            L_block = tl.load(
                L_block_pointer, boundary_check=(0,), padding_option="zero"
            )
            D_block = tl.load(
                D_block_pointer, boundary_check=(0,), padding_option="zero"
            )

            S = tl.dot(Q_block, tl.trans(K_block))
            S = S * scale

            P = tl.exp(S - L_block[:, None])
            d_V_block += tl.dot(tl.trans(tl.cast(P, V_block.dtype)), tl.cast(d_O_block, V_block.dtype))
            dP_block = tl.dot(tl.cast(d_O_block, V_block.dtype), tl.trans(V_block))
            dS_block = P * (dP_block - D_block[:, None])

            d_K_block += tl.dot(tl.trans(tl.cast(dS_block, K_block.dtype)), Q_block) * scale

            Q_block_pointer = Q_block_pointer.advance((Q_TILE_SIZE, 0))
            d_O_block_pointer = d_O_block_pointer.advance((Q_TILE_SIZE, 0))
            L_block_pointer = L_block_pointer.advance((Q_TILE_SIZE,))
            D_block_pointer = D_block_pointer.advance((Q_TILE_SIZE,))

    else:
        # advance the pointers to the correct position
        d_O_block_pointer = tl.advance(
            d_O_block_pointer, (column_tile_index * Q_TILE_SIZE, 0)
        )
        Q_block_pointer = tl.advance(
            Q_block_pointer, (column_tile_index * Q_TILE_SIZE, 0)
        )
        L_block_pointer = tl.advance(
            L_block_pointer, (column_tile_index * Q_TILE_SIZE,)
        )
        D_block_pointer = tl.advance(
            D_block_pointer, (column_tile_index * Q_TILE_SIZE,)
        )

        # load the necessary blocks needed for the computation
        d_O_block = tl.load(
            d_O_block_pointer, boundary_check=(0,), padding_option="zero"
        )
        Q_block = tl.load(Q_block_pointer, boundary_check=(0,), padding_option="zero")
        L_block = tl.load(L_block_pointer, boundary_check=(0,), padding_option="zero")
        D_block = tl.load(D_block_pointer, boundary_check=(0,), padding_option="zero")

        # compute the attention scores
        S = tl.dot(Q_block, tl.trans(K_block))
        S = S * scale
        column_range = tl.arange(0, K_TILE_SIZE) + column_tile_index * K_TILE_SIZE
        row_range = tl.arange(0, Q_TILE_SIZE) + column_tile_index * Q_TILE_SIZE
        mask = column_range[None, :] <= row_range[:, None]
        S = tl.where(mask, S, -float("inf"))

        # apply the softmax function
        P = tl.exp(S - L_block[:, None])
        d_V_block += tl.dot(
            tl.trans(tl.cast(P, V_block.dtype)), tl.cast(d_O_block, V_block.dtype)
        )
        dP_block = tl.dot(tl.cast(d_O_block, V_block.dtype), tl.trans(V_block))
        dS_block = P * (dP_block - D_block[:, None])

        d_K_block += tl.dot(tl.trans(tl.cast(dS_block, K_block.dtype)), Q_block) * scale

        Q_block_pointer = Q_block_pointer.advance((Q_TILE_SIZE, 0))
        d_O_block_pointer = d_O_block_pointer.advance((Q_TILE_SIZE, 0))
        L_block_pointer = L_block_pointer.advance((Q_TILE_SIZE,))
        D_block_pointer = D_block_pointer.advance((Q_TILE_SIZE,))

        for i in range(column_tile_index + 1, NUM_QUERY_TILES):
            # load the necessary blocks needed for the computation
            d_O_block = tl.load(
                d_O_block_pointer, boundary_check=(0,), padding_option="zero"
            )
            Q_block = tl.load(
                Q_block_pointer, boundary_check=(0,), padding_option="zero"
            )
            L_block = tl.load(
                L_block_pointer, boundary_check=(0,), padding_option="zero"
            )
            D_block = tl.load(
                D_block_pointer, boundary_check=(0,), padding_option="zero"
            )

            S = tl.dot(Q_block, tl.trans(K_block))
            S = S * scale

            P = tl.exp(S - L_block[:, None])
            d_V_block += tl.dot(
                tl.trans(tl.cast(P, V_block.dtype)), tl.cast(d_O_block, V_block.dtype)
            )
            dP_block = tl.dot(tl.cast(d_O_block, V_block.dtype), tl.trans(V_block))
            dS_block = P * (dP_block - D_block[:, None])

            d_K_block += (
                tl.dot(tl.trans(tl.cast(dS_block, K_block.dtype)), Q_block) * scale
            )

            Q_block_pointer = Q_block_pointer.advance((Q_TILE_SIZE, 0))
            d_O_block_pointer = d_O_block_pointer.advance((Q_TILE_SIZE, 0))
            L_block_pointer = L_block_pointer.advance((Q_TILE_SIZE,))
            D_block_pointer = D_block_pointer.advance((Q_TILE_SIZE,))

    tl.store(d_K_block_ptr, d_K_block, boundary_check=(0))
    tl.store(d_V_block_ptr, d_V_block, boundary_check=(0))


@triton.jit
def flash_attention_backward_Q_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    d_Q_ptr,
    d_O_ptr,
    D_ptr,
    stride_Q_batch,
    stride_Q_row,
    stride_Q_col,
    stride_K_batch,
    stride_K_row,
    stride_K_col,
    stride_V_batch,
    stride_V_row,
    stride_V_col,
    stride_L_batch,
    stride_L_row,
    stride_dQ_batch,
    stride_dQ_row,
    stride_dQ_col,
    stride_dO_batch,
    stride_dO_row,
    stride_dO_col,
    stride_D_batch,
    stride_D_row,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    NUM_KEY_TILES: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
) -> None:
    row_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    shape = (N_QUERIES, D)

    # Query block pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_Q_batch,
        shape=(N_QUERIES, D),
        strides=(stride_Q_row, stride_Q_col),
        offsets=(row_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_K_batch,
        shape=(N_KEYS, D),
        strides=(stride_K_row, stride_K_col),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_V_batch,
        shape=(N_KEYS, D),
        strides=(stride_V_row, stride_V_col),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    d_Q_block_ptr = tl.make_block_ptr(
        d_Q_ptr + batch_index * stride_dQ_batch,
        shape=(N_QUERIES, D),
        strides=(stride_dQ_row, stride_dQ_col),
        offsets=(row_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    d_O_block_ptr = tl.make_block_ptr(
        d_O_ptr + batch_index * stride_dO_batch,
        shape=(N_QUERIES, D),
        strides=(stride_dO_row, stride_dO_col),
        offsets=(row_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_D_batch,
        shape=(N_QUERIES,),
        strides=(stride_D_row,),
        offsets=(row_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_L_batch,
        shape=(N_QUERIES,),
        strides=(stride_L_row,),
        offsets=(row_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    d_Q_block = tl.load(d_Q_block_ptr, boundary_check=(0,), padding_option="zero")
    d_Q_block = tl.cast(d_Q_block, tl.float32)
    d_O_block = tl.load(d_O_block_ptr, boundary_check=(0,), padding_option="zero")
    d_O_block = tl.cast(d_O_block, K_block.dtype)
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    if not is_causal:
        for i in range(NUM_KEY_TILES):
            K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

            S = tl.dot(tl.cast(Q_block, K_block.dtype), tl.trans(K_block)) * scale
            P = tl.exp(S - L_block[:, None])
            dP = tl.dot(tl.cast(d_O_block, V_block.dtype), tl.trans(V_block))
            dS = P * (dP - D_block[:, None])

            d_Q_block += tl.dot(tl.cast(dS, K_block.dtype), K_block) * scale

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    else:
        for i in range(row_tile_index):
            K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

            S = tl.dot(tl.cast(Q_block, K_block.dtype), tl.trans(K_block)) * scale
            P = tl.exp(S - L_block[:, None])
            dP = tl.dot(tl.cast(d_O_block, V_block.dtype), tl.trans(V_block))
            dS = P * (dP - D_block[:, None])

            d_Q_block += tl.dot(tl.cast(dS, K_block.dtype), K_block) * scale

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        S = tl.dot(tl.cast(Q_block, K_block.dtype), tl.trans(K_block)) * scale
        column_range = tl.arange(0, K_TILE_SIZE)
        row_range = tl.arange(0, Q_TILE_SIZE)
        mask = column_range[None, :] <= row_range[:, None]
        S = tl.where(mask, S, -float("inf"))
        P = tl.exp(S - L_block[:, None])
        dP = tl.dot(tl.cast(d_O_block, V_block.dtype), tl.trans(V_block))
        dS = P * (dP - D_block[:, None])

        d_Q_block += tl.dot(tl.cast(dS, K_block.dtype), K_block) * scale

    tl.store(d_Q_block_ptr, d_Q_block, boundary_check=(0))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        batch_size, seq_len, d_model = Q.shape
        Q_TILE_SIZE = GLOBAL_Q_TILE_SIZE
        K_TILE_SIZE = GLOBAL_K_TILE_SIZE
        scale = 1 / (d_model**0.5)
        ctx.scale = scale
        O = torch.zeros(
            (batch_size, seq_len, d_model), device=Q.device, dtype=torch.float32
        )
        L = torch.zeros((batch_size, seq_len), device=Q.device, dtype=torch.float32)
        flash_attention_forward_kernel[(triton.cdiv(seq_len, Q_TILE_SIZE), batch_size)](
            Q,  # Q_ptr
            K,  # K_ptr
            V,  # V_ptr
            O,  # O_ptr
            L,  # L_ptr
            Q.stride(0),  # stride_Q_batch
            Q.stride(1),  # stride_Q_row
            Q.stride(2),  # stride_Q_col
            K.stride(0),  # stride_K_batch
            K.stride(1),  # stride_K_row
            K.stride(2),  # stride_K_col
            V.stride(0),  # stride_V_batch
            V.stride(1),  # stride_V_row
            V.stride(2),  # stride_V_col
            O.stride(0),  # stride_O_batch
            O.stride(1),  # stride_O_row
            O.stride(2),  # stride_O_col
            L.stride(0),  # stride_L_batch
            L.stride(1),  # stride_L_row
            N_QUERIES=seq_len,
            N_KEYS=seq_len,
            scale=scale,
            D=d_model,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            NUM_KEY_TILES=triton.cdiv(seq_len, K_TILE_SIZE),
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        # Q = Q.to(torch.float32)
        # K = K.float()
        # V = V.float()
        # O = O.float()
        is_causal = ctx.is_causal

        D = torch.sum(O * dO, dim=-1)
        scale = ctx.scale

        # allocate memory for the gradients of the query, key, and value
        dQ = torch.zeros_like(Q).to(torch.float32)
        dK = torch.zeros_like(K).to(torch.float32)
        dV = torch.zeros_like(V).to(torch.float32)

        # Run the kernel for the backward pass of the Key and Value matrices
        batch_size, seq_len, d_model = Q.shape
        K_TILE_SIZE = GLOBAL_K_TILE_SIZE
        Q_TILE_SIZE = GLOBAL_Q_TILE_SIZE
        NUM_KEY_TILES = triton.cdiv(seq_len, K_TILE_SIZE)
        NUM_QUERY_TILES = triton.cdiv(seq_len, Q_TILE_SIZE)

        flash_attention_backward_K_V_kernel[
            (triton.cdiv(seq_len, K_TILE_SIZE), batch_size)
        ](
            Q,
            K,
            V,
            L,
            dK,
            dV,
            dO,
            D,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            D.stride(0),
            D.stride(1),
            N_QUERIES=seq_len,
            N_KEYS=seq_len,
            scale=scale,
            D=d_model,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=ctx.is_causal,
            NUM_KEY_TILES=NUM_KEY_TILES,
            NUM_QUERY_TILES=NUM_QUERY_TILES,
        )
        # Run the kernel for the backward pass of the Query matrix
        flash_attention_backward_Q_kernel[
            (triton.cdiv(seq_len, Q_TILE_SIZE), batch_size)
        ](
            Q,
            K,
            V,
            L,
            dQ,
            dO,
            D,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            D.stride(0),
            D.stride(1),
            N_QUERIES=seq_len,
            N_KEYS=seq_len,
            scale=scale,
            D=d_model,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
            NUM_KEY_TILES=NUM_KEY_TILES,
            NUM_QUERY_TILES=NUM_QUERY_TILES,
        )

        return dQ, dK, dV, None


# Naive PyTorch causal attention implementation for comparison
def naive_causal_attention(Q, K, V, is_causal=True):
    # Q,K,V: (b, q, d), (b, k, d), (b, k, d)
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (b, q, k)

    if is_causal:
        q_len, k_len = attn_scores.shape[-2], attn_scores.shape[-1]
        mask = torch.tril(torch.ones((q_len, k_len), device=Q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output


def pytorch_scaled_dot_product_attention(Q, K, V, is_causal=True):
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=is_causal
    )


if __name__ == "__main__":
    is_causal = True
    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU")
    else:
        device = "cpu"
        print("Using CPU")
    batch_size = 3
    model_dims = [16, 32, 64, 128]
    seq_lengths = [256 * i for i in [1, 2, 4, 8, 16]]
    is_causal_options = [False]

    for is_causal in is_causal_options:
        for model_dimension in model_dims:
            for sequence_length in seq_lengths:
                print("=" * 80)
                print(
                    f"Testing: is_causal={is_causal} | model_dim={model_dimension} | seq_len={sequence_length}"
                )

                Q = torch.randn(
                    batch_size,
                    sequence_length,
                    model_dimension,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                K = torch.randn(
                    batch_size,
                    sequence_length,
                    model_dimension,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                V = torch.randn(
                    batch_size,
                    sequence_length,
                    model_dimension,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )

                # Forward pass using FlashAttentionForward
                fa_forward = FlashAttentionTriton.apply
                O = fa_forward(Q, K, V, is_causal)

                sum_of_elements = torch.sum(O)
                print("Sum of output elements:", sum_of_elements.item())
                sum_of_elements.backward()
                print(
                    "Q.grad.shape:",
                    Q.grad.shape,
                    "K.grad.shape:",
                    K.grad.shape,
                    "V.grad.shape:",
                    V.grad.shape,
                )
                print(
                    "Q.grad.dtype:",
                    Q.grad.dtype,
                    "K.grad.dtype:",
                    K.grad.dtype,
                    "V.grad.dtype:",
                    V.grad.dtype,
                )

                # Naive PyTorch reference
                Q_naive = Q.detach().clone().requires_grad_(True)
                K_naive = K.detach().clone().requires_grad_(True)
                V_naive = V.detach().clone().requires_grad_(True)
                O_naive = naive_causal_attention(Q_naive, K_naive, V_naive, is_causal)
                sum_naive = torch.sum(O_naive)
                sum_naive.backward()
                grad_Q_naive = Q_naive.grad
                grad_K_naive = K_naive.grad
                grad_V_naive = V_naive.grad

                # Compare outputs & gradients
                print("Testing FlashAttentionForward vs naive implementation")
                print("Output diff (max abs):", (O - O_naive).abs().max().item())
                print(
                    "Q.grad diff (max abs):", (Q.grad - grad_Q_naive).abs().max().item()
                )
                print(
                    "K.grad diff (max abs):", (K.grad - grad_K_naive).abs().max().item()
                )
                print(
                    "V.grad diff (max abs):", (V.grad - grad_V_naive).abs().max().item()
                )
                print("Output diff l2 norm:", torch.norm(O - O_naive).item())
                print("Q.grad diff l2 norm:", torch.norm(Q.grad - grad_Q_naive).item())
                print("K.grad diff l2 norm:", torch.norm(K.grad - grad_K_naive).item())
                print(
                    "V.grad relative diff l2 norm:",
                    torch.norm(V.grad - grad_V_naive).item()
                    / torch.norm(grad_V_naive).item(),
                )
                print(
                    "K.grad relative diff l2 norm:",
                    torch.norm(K.grad - grad_K_naive).item()
                    / torch.norm(grad_K_naive).item(),
                )
                print(
                    "Q.grad relative diff l2 norm:",
                    torch.norm(Q.grad - grad_Q_naive).item()
                    / torch.norm(grad_Q_naive).item(),
                )
                print("O shape:", O.shape, "O_naive shape:", O_naive.shape)
                print(
                    "Q.grad dtype:",
                    Q.grad.dtype,
                    "grad_Q_naive dtype:",
                    grad_Q_naive.dtype,
                )

                # Avoid double backward
                Q.grad = None
                K.grad = None
                V.grad = None
