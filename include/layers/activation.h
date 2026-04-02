#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

namespace nano_vllm {

class SiluAndMul {
public:
    Tensor forward(const Tensor& gate_up,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;
};

} // namespace nano_vllm