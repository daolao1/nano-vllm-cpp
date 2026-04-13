#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

// Flash Attention 2 forward pass for prefill (variable-length sequences).
// Q: [total_q, num_heads, head_dim]  (contiguous bf16 on CUDA)
// K: [total_k, num_kv_heads, head_dim]
// V: [total_k, num_kv_heads, head_dim]
// cu_seqlens_q: [batch_size+1] int32 on CUDA
// cu_seqlens_k: [batch_size+1] int32 on CUDA
// Output: [total_q, num_heads, head_dim] bf16
// softmax_lse: [num_heads, total_q] float32 (allocated internally)
void flash_attn_varlen_fwd(const Tensor& query,
                           const Tensor& key,
                           const Tensor& value,
                           const Tensor& cu_seqlens_q,
                           const Tensor& cu_seqlens_k,
                           int max_seqlen_q,
                           int max_seqlen_k,
                           float scale,
                           bool is_causal,
                           Tensor& output,
                           DeviceAllocator& allocator,
                           cudaStream_t stream = nullptr);

// Initialize the Flash Attention library (dlopen the .so).
// Called automatically on first use.
bool flash_attn_available();

// Flash Attention 2 decode pass with paged KV cache.
// query:       [batch_size, num_heads, head_dim]  bf16 on CUDA
// k_cache:     [num_blocks, page_block_size, num_kv_heads, head_dim]
// v_cache:     [num_blocks, page_block_size, num_kv_heads, head_dim]
// context_lens: [batch_size] int32 on CUDA (per-sequence context lengths)
// block_tables: [batch_size, max_num_blocks_per_seq] int32 on CUDA
// output:      [batch_size, num_heads, head_dim]  bf16 on CUDA
void flash_attn_decode_kvcache(const Tensor& query,
                               const Tensor& k_cache,
                               const Tensor& v_cache,
                               const Tensor& context_lens,
                               const Tensor& block_tables,
                               int num_kv_heads,
                               int head_dim,
                               int block_size,
                               float scale,
                               Tensor& output,
                               DeviceAllocator& allocator,
                               cudaStream_t stream = nullptr);

} // namespace nano_vllm
