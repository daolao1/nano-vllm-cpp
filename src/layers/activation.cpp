#include "layers/activation.h"

#include "layers/kernel_ops.h"

#include <stdexcept>

namespace nano_vllm {

Tensor SiluAndMul::forward(const Tensor& gate_up,
                           DeviceAllocator& allocator,
                           cudaStream_t stream) const {
    if (!gate_up.defined()) {
        throw std::invalid_argument("gate_up must be defined");
    }
    if (gate_up.sizes().empty() || gate_up.sizes().back() % 2 != 0) {
        throw std::invalid_argument("gate_up last dimension must be divisible by 2");
    }

    std::vector<int64_t> output_sizes = gate_up.sizes();
    output_sizes.back() /= 2;
    Tensor output = Tensor::empty(output_sizes, gate_up.dtype(), gate_up.device(), allocator);
    silu_and_mul(gate_up, output, stream);
    return output;
}

} // namespace nano_vllm