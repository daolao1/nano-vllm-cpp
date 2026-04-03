#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

namespace nano_vllm {

class Attention {
public:
    Attention(int num_heads, int num_kv_heads, int head_dim, int block_size);

    void set_kv_cache(const Tensor& k_cache, const Tensor& v_cache);
    void set_kv_cache_int8(const Tensor& k_cache, const Tensor& v_cache,
                           const Tensor& k_scale, const Tensor& v_scale);

    Tensor forward(const Tensor& query,
                   const Tensor& key,
                   const Tensor& value,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr);

    // Decode-only: run flash attention without store_kvcache (for fused path).
    Tensor forward_decode_flash_only(const Tensor& query,
                                     DeviceAllocator& allocator,
                                     cudaStream_t stream = nullptr);

    Tensor& k_cache() { return k_cache_; }
    Tensor& v_cache() { return v_cache_; }
    const Tensor& k_cache() const { return k_cache_; }
    const Tensor& v_cache() const { return v_cache_; }
    Tensor& k_scale() { return k_scale_; }
    Tensor& v_scale() { return v_scale_; }
    bool is_int8() const { return is_int8_; }

private:
    int num_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int block_size_ = 0;
    float scale_ = 1.0f;
    bool is_int8_ = false;
    Tensor k_cache_;
    Tensor v_cache_;
    Tensor k_scale_;
    Tensor v_scale_;
};

} // namespace nano_vllm