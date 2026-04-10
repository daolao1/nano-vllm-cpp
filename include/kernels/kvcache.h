#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void store_kvcache(const Tensor& key,
                   const Tensor& value,
                   Tensor& k_cache,
                   Tensor& v_cache,
                   const Tensor& slot_mapping,
                   cudaStream_t stream = nullptr);

/// Quantize fp16/bf16 K,V to int8 and store with per-head per-token scales.
/// k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim] int8
/// k_scale, v_scale: [num_blocks, block_size, num_kv_heads] float16
void store_kvcache_int8(const Tensor& key,
                        const Tensor& value,
                        Tensor& k_cache,
                        Tensor& v_cache,
                        Tensor& k_scale,
                        Tensor& v_scale,
                        const Tensor& slot_mapping,
                        cudaStream_t stream = nullptr);

} // namespace nano_vllm
