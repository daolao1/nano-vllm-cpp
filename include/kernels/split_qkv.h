#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void split_qkv(const Tensor& qkv,
               int q_size,
               int kv_size,
               Tensor& q,
               Tensor& k,
               Tensor& v,
               cudaStream_t stream = nullptr);

// Fused kernel: split QKV + RMSNorm(Q,K) + RoPE(Q,K) + store K/V to paged KV cache.
// Replaces 5 separate kernels (split_qkv + q_norm + k_norm + rotary + store_kvcache).
// q_out: [rows, num_q_heads, head_dim] — output for flash attention query input.
// K/V are written directly to paged kv_cache via slot_mapping (no separate output).
void fused_split_norm_rope_store(
    const Tensor& qkv,              // [rows, q_size + 2*kv_size]
    const Tensor& positions,         // [rows] int32
    const Tensor& q_norm_weight,     // [head_dim]
    const Tensor& k_norm_weight,     // [head_dim]
    float norm_eps,
    const Tensor& cos_sin_cache,     // [max_pos, head_dim] float32
    const Tensor& slot_mapping,      // [rows] int32
    Tensor& k_cache,                 // paged KV cache
    Tensor& v_cache,                 // paged KV cache
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    Tensor& q_out,                   // [rows, num_q_heads, head_dim]
    cudaStream_t stream = nullptr);

} // namespace nano_vllm
