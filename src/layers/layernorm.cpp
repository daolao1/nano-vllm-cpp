#include "layers/layernorm.h"

#include "layers/kernel_ops.h"
#include "utils/loader.h"

#include <stdexcept>
#include <vector>

namespace nano_vllm {
namespace {

void fill_weight_with_ones(Tensor& weight, DeviceAllocator& allocator) {
    if (weight.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument("RMSNorm currently expects CUDA weights");
    }

    if (weight.dtype() == ScalarType::kFloat32) {
        std::vector<float> host(weight.numel(), 1.0f);
        allocator.copy_to_device_async(weight.data(), host.data(), weight.nbytes(), nullptr);
        allocator.synchronize_stream(nullptr);
        return;
    }
    if (weight.dtype() == ScalarType::kFloat16) {
        std::vector<uint16_t> host(weight.numel(), static_cast<uint16_t>(0x3c00));
        allocator.copy_to_device_async(weight.data(), host.data(), weight.nbytes(), nullptr);
        allocator.synchronize_stream(nullptr);
        return;
    }
    if (weight.dtype() == ScalarType::kBFloat16) {
        std::vector<uint16_t> host(weight.numel(), static_cast<uint16_t>(0x3f80));
        allocator.copy_to_device_async(weight.data(), host.data(), weight.nbytes(), nullptr);
        allocator.synchronize_stream(nullptr);
        return;
    }
    throw std::invalid_argument("unsupported RMSNorm weight dtype");
}

void validate_hidden_size(const char* name, const Tensor& tensor, int hidden_size) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.sizes().empty() || tensor.sizes().back() != hidden_size) {
        throw std::invalid_argument(std::string(name) + " last dimension must equal hidden_size");
    }
}

} // namespace

RMSNorm::RMSNorm(int hidden_size,
                 float eps,
                 ScalarType dtype,
                 Device device,
                 DeviceAllocator& allocator)
    : hidden_size_(hidden_size),
      eps_(eps),
      weight_(Tensor::zeros({hidden_size}, dtype, device, allocator)) {
    if (hidden_size_ <= 0) {
        throw std::invalid_argument("hidden_size must be positive");
    }
    fill_weight_with_ones(weight_, allocator);
}

Tensor RMSNorm::forward(const Tensor& input,
                        DeviceAllocator& allocator,
                        cudaStream_t stream) const {
    validate_hidden_size("input", input, hidden_size_);
    Tensor output = Tensor::empty(input.sizes(), input.dtype(), input.device(), allocator);
    rms_norm(input, weight_, eps_, output, stream);
    return output;
}

std::pair<Tensor, Tensor> RMSNorm::forward(const Tensor& input,
                                           const Tensor& residual,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) const {
    validate_hidden_size("input", input, hidden_size_);
    validate_hidden_size("residual", residual, hidden_size_);
    if (input.sizes() != residual.sizes()) {
        throw std::invalid_argument("input and residual must have the same shape");
    }

    Tensor output = Tensor::empty(input.sizes(), input.dtype(), input.device(), allocator);
    Tensor residual_out = Tensor::empty(residual.sizes(), residual.dtype(), residual.device(), allocator);
    add_rms_norm(input, residual, weight_, eps_, output, residual_out, stream);
    return {output, residual_out};
}

void RMSNorm::weight_loader(const Tensor& source,
                            DeviceAllocator& allocator,
                            cudaStream_t stream) {
    if (source.sizes() != weight_.sizes()) {
        throw std::invalid_argument("RMSNorm weight must have shape [hidden_size]");
    }
    copy_tensor_to_parameter(source, weight_, allocator, stream);
}

} // namespace nano_vllm