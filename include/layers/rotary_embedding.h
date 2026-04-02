#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

namespace nano_vllm {

class RotaryEmbedding {
public:
    RotaryEmbedding(int head_size,
                    int rotary_dim,
                    int max_position,
                    float base,
                    Device device,
                    DeviceAllocator& allocator);

    void forward(const Tensor& positions,
                 Tensor& query,
                 Tensor& key,
                 cudaStream_t stream = nullptr) const;

    int head_size() const { return head_size_; }
    int rotary_dim() const { return rotary_dim_; }
    int max_position() const { return max_position_; }
    float base() const { return base_; }

    const Tensor& cos_sin_cache() const { return cos_sin_cache_; }

private:
    int head_size_ = 0;
    int rotary_dim_ = 0;
    int max_position_ = 0;
    float base_ = 10000.0f;
    Tensor cos_sin_cache_;
};

} // namespace nano_vllm