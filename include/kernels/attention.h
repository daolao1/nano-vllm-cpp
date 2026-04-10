#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void launch_prefill_attention_dense(const Tensor& query,
                                    const Tensor* cu_seqlens_q,
                                    const Tensor& key,
                                    const Tensor& value,
                                    const Tensor* cu_seqlens_k,
                                    int num_kv_heads,
                                    int head_dim,
                                    float scale,
                                    Tensor& output,
                                    cudaStream_t stream = nullptr);

void launch_prefill_attention_paged(const Tensor& query,
                                    const Tensor* cu_seqlens_q,
                                    const Tensor& k_cache,
                                    const Tensor& v_cache,
                                    const Tensor* cu_seqlens_k,
                                    const Tensor& block_tables,
                                    int num_kv_heads,
                                    int head_dim,
                                    int block_size,
                                    float scale,
                                    Tensor& output,
                                    cudaStream_t stream = nullptr);

void launch_decode_attention_paged(const Tensor& query,
                                   const Tensor& k_cache,
                                   const Tensor& v_cache,
                                   const Tensor& context_lens,
                                   const Tensor& block_tables,
                                   int num_kv_heads,
                                   int head_dim,
                                   int block_size,
                                   float scale,
                                   Tensor& output,
                                   cudaStream_t stream = nullptr);

/// Decode attention reading from int8 quantized KV cache with fp16 scales.
void launch_decode_attention_paged_int8(const Tensor& query,
                                        const Tensor& k_cache,
                                        const Tensor& v_cache,
                                        const Tensor& k_scale,
                                        const Tensor& v_scale,
                                        const Tensor& context_lens,
                                        const Tensor& block_tables,
                                        int num_kv_heads,
                                        int head_dim,
                                        int block_size,
                                        float scale,
                                        Tensor& output,
                                        cudaStream_t stream = nullptr);

} // namespace nano_vllm
