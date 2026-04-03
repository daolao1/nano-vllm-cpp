#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <cstdint>
#include <vector>

namespace nano_vllm {

class Sampler {
public:
    explicit Sampler(uint64_t seed = 1) : seed_(seed) {}

    std::vector<int32_t> forward(const Tensor& logits,
                                 const Tensor& temperatures,
                                 const Tensor& top_ks,
                                 const Tensor& top_ps,
                                 const Tensor& penalty_token_ids,
                                 const Tensor& penalty_token_counts,
                                 const Tensor& penalties,
                                 DeviceAllocator& allocator) const;

private:
    uint64_t seed_ = 1;
};

} // namespace nano_vllm