#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <utility>

namespace nano_vllm {

class RMSNorm {
public:
    RMSNorm(int hidden_size,
            float eps,
            ScalarType dtype,
            Device device,
            DeviceAllocator& allocator);

    Tensor forward(const Tensor& input,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;
    std::pair<Tensor, Tensor> forward(const Tensor& input,
                                      const Tensor& residual,
                                      DeviceAllocator& allocator,
                                      cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);

    int hidden_size() const { return hidden_size_; }
    float eps() const { return eps_; }

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }

private:
    int hidden_size_ = 0;
    float eps_ = 1e-6f;
    Tensor weight_;
};

} // namespace nano_vllm