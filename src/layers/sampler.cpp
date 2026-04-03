#include "layers/sampler.h"

#include "layers/kernel_ops.h"

#include <stdexcept>
#include <vector>

namespace nano_vllm {
namespace {

std::vector<int32_t> copy_int_tensor_to_host(const Tensor& tensor, const DeviceAllocator& allocator) {
    if (!tensor.defined() || tensor.device().type != DeviceType::kCUDA || !tensor.is_contiguous() ||
        tensor.dtype() != ScalarType::kInt32) {
        throw std::invalid_argument("sampler output must be a contiguous CUDA int32 tensor");
    }
    std::vector<int32_t> host(tensor.numel(), 0);
    allocator.copy_to_host_async(host.data(), tensor.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return host;
}

} // namespace

std::vector<int32_t> Sampler::forward(const Tensor& logits,
                                      const Tensor& temperatures,
                                      const Tensor& top_ks,
                                      const Tensor& top_ps,
                                      const Tensor& penalty_token_ids,
                                      const Tensor& penalty_token_counts,
                                      const Tensor& penalties,
                                      DeviceAllocator& allocator) const {
    // Apply repetition penalty in-place if any row has penalty != 1.0.
    Tensor mutable_logits = logits; // shallow copy — shares data
    if (penalty_token_ids.defined() && penalty_token_counts.defined() && penalties.defined()) {
        apply_repetition_penalty(mutable_logits, penalty_token_ids,
                                 penalty_token_counts, penalties);
    }

    Tensor output = Tensor::zeros({logits.sizes()[0]}, ScalarType::kInt32, logits.device(), allocator);
    sample_tokens(mutable_logits, temperatures, top_ks, top_ps, seed_, output);
    return copy_int_tensor_to_host(output, allocator);
}

} // namespace nano_vllm