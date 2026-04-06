#include "kernels/embedding.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

template <typename IndexT, typename WeightT, typename OutputT>
__global__ void embedding_lookup_kernel(const IndexT* input_ids,
                                        const WeightT* weight,
                                        OutputT* output,
                                        int64_t token_count,
                                        int64_t hidden_dim,
                                        int64_t vocab_size,
                                        int64_t vocab_start,
                                        int64_t local_vocab_size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = token_count * hidden_dim;
    if (index >= total) {
        return;
    }

    const int64_t token_index = index / hidden_dim;
    const int64_t col = index % hidden_dim;
    const int64_t token = static_cast<int64_t>(input_ids[token_index]);
    if (token < 0 || token >= vocab_size) {
        output[index] = from_float<OutputT>(0.0f);
        return;
    }
    if (token < vocab_start || token >= vocab_start + local_vocab_size) {
        output[index] = from_float<OutputT>(0.0f);
        return;
    }

    output[index] = from_float<OutputT>(to_float(weight[(token - vocab_start) * hidden_dim + col]));
}

template <typename IndexT, typename WeightT, typename OutputT>
void launch_embedding_lookup(const Tensor& input_ids,
                             const Tensor& weight,
                             int64_t vocab_size,
                             int64_t vocab_start,
                             Tensor& output,
                             cudaStream_t stream) {
    const int64_t token_count = static_cast<int64_t>(input_ids.numel());
    const int64_t hidden_dim = weight.sizes()[1];
    const int64_t total = token_count * hidden_dim;
    if (total == 0) {
        return;
    }

    const unsigned int blocks = static_cast<unsigned int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    embedding_lookup_kernel<IndexT, WeightT, OutputT><<<blocks, kThreadsPerBlock, 0, stream>>>(
        input_ids.data_as<IndexT>(),
        weight.data_as<WeightT>(),
        output.data_as<OutputT>(),
        token_count,
        hidden_dim,
        vocab_size,
        vocab_start,
        weight.sizes()[0]);
    throw_if_cuda_error(cudaPeekAtLastError(), "embedding_lookup_kernel launch");
}

} // namespace

void embedding_lookup(const Tensor& input_ids,
                      const Tensor& weight,
                      int64_t vocab_size,
                      int64_t vocab_start,
                      Tensor& output,
                      cudaStream_t stream) {
    ensure_cuda_tensor("input_ids", input_ids);
    ensure_cuda_tensor("weight", weight);
    ensure_cuda_tensor("output", output);
    ensure_same_device("input_ids", input_ids, "weight", weight);
    ensure_same_device("input_ids", input_ids, "output", output);
    validate_supported_dtype("weight", weight.dtype());
    if (input_ids.dtype() != ScalarType::kInt32 && input_ids.dtype() != ScalarType::kInt64) {
        throw std::invalid_argument("input_ids must be int32 or int64");
    }
    if (weight.dim() != 2) {
        throw std::invalid_argument("weight must have shape [local_vocab, hidden_dim]");
    }
    validate_supported_dtype("output", output.dtype());
    std::vector<int64_t> expected_sizes = input_ids.sizes();
    expected_sizes.push_back(weight.sizes()[1]);
    if (output.sizes() != expected_sizes) {
        throw std::invalid_argument("output must have shape input_ids.sizes() + [hidden_dim]");
    }
    if (vocab_size <= 0 || weight.sizes()[0] < 0 || vocab_start < 0 ||
        vocab_start + weight.sizes()[0] > vocab_size) {
        throw std::invalid_argument("embedding vocabulary range is invalid");
    }

    if (input_ids.dtype() == ScalarType::kInt32) {
        switch (weight.dtype()) {
        case ScalarType::kFloat16:
            if (output.dtype() == ScalarType::kFloat16)
                launch_embedding_lookup<int32_t, __half, __half>(input_ids, weight, vocab_size, vocab_start, output, stream);
            else
                launch_embedding_lookup<int32_t, __half, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kBFloat16:
            if (output.dtype() == ScalarType::kBFloat16)
                launch_embedding_lookup<int32_t, __nv_bfloat16, __nv_bfloat16>(input_ids, weight, vocab_size, vocab_start, output, stream);
            else
                launch_embedding_lookup<int32_t, __nv_bfloat16, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kFloat32:
            launch_embedding_lookup<int32_t, float, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        default:
            break;
        }
    } else {
        switch (weight.dtype()) {
        case ScalarType::kFloat16:
            if (output.dtype() == ScalarType::kFloat16)
                launch_embedding_lookup<int64_t, __half, __half>(input_ids, weight, vocab_size, vocab_start, output, stream);
            else
                launch_embedding_lookup<int64_t, __half, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kBFloat16:
            if (output.dtype() == ScalarType::kBFloat16)
                launch_embedding_lookup<int64_t, __nv_bfloat16, __nv_bfloat16>(input_ids, weight, vocab_size, vocab_start, output, stream);
            else
                launch_embedding_lookup<int64_t, __nv_bfloat16, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kFloat32:
            launch_embedding_lookup<int64_t, float, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        default:
            break;
        }
    }

    throw std::invalid_argument("unsupported dtype combination for embedding_lookup");
}

} // namespace nano_vllm
