#include "layers/rotary_embedding.h"

#include "layers/kernel_ops.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nano_vllm {

RotaryEmbedding::RotaryEmbedding(int head_size,
                                 int rotary_dim,
                                 int max_position,
                                 float base,
                                 Device device,
                                 DeviceAllocator& allocator)
    : head_size_(head_size),
      rotary_dim_(rotary_dim),
      max_position_(max_position),
      base_(base),
      cos_sin_cache_(Tensor::zeros({max_position, rotary_dim}, ScalarType::kFloat32, device, allocator)) {
    if (head_size_ <= 0 || rotary_dim_ <= 0 || max_position_ <= 0) {
        throw std::invalid_argument("rotary embedding dimensions must be positive");
    }
    if (rotary_dim_ != head_size_) {
        throw std::invalid_argument("rotary_dim must equal head_size in the current implementation");
    }
    if (rotary_dim_ % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even");
    }
    if (cos_sin_cache_.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument("RotaryEmbedding currently expects a CUDA cache tensor");
    }

    std::vector<float> host(static_cast<size_t>(max_position_) * static_cast<size_t>(rotary_dim_), 0.0f);
    const int half_dim = rotary_dim_ / 2;
    for (int position = 0; position < max_position_; ++position) {
        for (int idx = 0; idx < half_dim; ++idx) {
            const float exponent = static_cast<float>(2 * idx) / static_cast<float>(rotary_dim_);
            const float inv_freq = 1.0f / std::pow(base_, exponent);
            const float angle = static_cast<float>(position) * inv_freq;
            host[static_cast<size_t>(position * rotary_dim_ + idx)] = std::cos(angle);
            host[static_cast<size_t>(position * rotary_dim_ + idx + half_dim)] = std::sin(angle);
        }
    }

    allocator.copy_to_device_async(cos_sin_cache_.data(), host.data(), cos_sin_cache_.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
}

void RotaryEmbedding::forward(const Tensor& positions,
                              Tensor& query,
                              Tensor& key,
                              cudaStream_t stream) const {
    rotary_embedding(positions, cos_sin_cache_, query, key, stream);
}

} // namespace nano_vllm