#include "kernels/gather.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

template <typename IndexT, typename DataT>
__global__ void gather_last_tokens_kernel(const DataT* input,
                                          const IndexT* cu_seqlens,
                                          DataT* output,
                                          int64_t batch_size,
                                          int64_t input_rows,
                                          int64_t cols) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * cols;
    if (index >= total) {
        return;
    }

    const int64_t row = index / cols;
    const int64_t col = index % cols;
    const int64_t source_row = static_cast<int64_t>(cu_seqlens[row + 1]) - 1;
    if (source_row < 0 || source_row >= input_rows) {
        return;
    }

    output[index] = input[source_row * cols + col];
}

template <typename IndexT, typename DataT>
void launch_gather_last_tokens(const Tensor& input,
                               const Tensor& cu_seqlens,
                               Tensor& output,
                               cudaStream_t stream) {
    const int64_t batch_size = static_cast<int64_t>(cu_seqlens.numel()) - 1;
    const int64_t cols = input.sizes()[1];
    const int64_t total = batch_size * cols;
    if (total == 0) {
        return;
    }

    const unsigned int blocks = static_cast<unsigned int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    gather_last_tokens_kernel<IndexT, DataT><<<blocks, kThreadsPerBlock, 0, stream>>>(
        input.data_as<DataT>(),
        cu_seqlens.data_as<IndexT>(),
        output.data_as<DataT>(),
        batch_size,
        input.sizes()[0],
        cols);
    throw_if_cuda_error(cudaPeekAtLastError(), "gather_last_tokens_kernel launch");
}

} // namespace

void gather_last_tokens(const Tensor& input,
                        const Tensor& cu_seqlens,
                        Tensor& output,
                        cudaStream_t stream) {
    ensure_cuda_tensor("input", input);
    ensure_cuda_tensor("cu_seqlens", cu_seqlens);
    ensure_cuda_tensor("output", output);
    ensure_same_device("input", input, "cu_seqlens", cu_seqlens);
    ensure_same_device("input", input, "output", output);
    ensure_same_dtype("input", input, "output", output);
    validate_supported_dtype("input", input.dtype());
    if (cu_seqlens.dtype() != ScalarType::kInt32 && cu_seqlens.dtype() != ScalarType::kInt64) {
        throw std::invalid_argument("cu_seqlens must be int32 or int64");
    }
    if (input.dim() != 2 || output.dim() != 2) {
        throw std::invalid_argument("gather_last_tokens expects rank-2 input and output tensors");
    }
    if (cu_seqlens.dim() != 1 || cu_seqlens.numel() < 2) {
        throw std::invalid_argument("cu_seqlens must have shape [batch_size + 1]");
    }
    const int64_t batch_size = static_cast<int64_t>(cu_seqlens.numel()) - 1;
    if (output.sizes()[0] != batch_size || output.sizes()[1] != input.sizes()[1]) {
        throw std::invalid_argument("output must have shape [batch_size, input.size(1)]");
    }

    if (cu_seqlens.dtype() == ScalarType::kInt32) {
        switch (input.dtype()) {
        case ScalarType::kFloat16:
            launch_gather_last_tokens<int32_t, __half>(input, cu_seqlens, output, stream);
            return;
        case ScalarType::kBFloat16:
            launch_gather_last_tokens<int32_t, __nv_bfloat16>(input, cu_seqlens, output, stream);
            return;
        case ScalarType::kFloat32:
            launch_gather_last_tokens<int32_t, float>(input, cu_seqlens, output, stream);
            return;
        default:
            break;
        }
    } else {
        switch (input.dtype()) {
        case ScalarType::kFloat16:
            launch_gather_last_tokens<int64_t, __half>(input, cu_seqlens, output, stream);
            return;
        case ScalarType::kBFloat16:
            launch_gather_last_tokens<int64_t, __nv_bfloat16>(input, cu_seqlens, output, stream);
            return;
        case ScalarType::kFloat32:
            launch_gather_last_tokens<int64_t, float>(input, cu_seqlens, output, stream);
            return;
        default:
            break;
        }
    }

    throw std::invalid_argument("unsupported dtype combination for gather_last_tokens");
}

} // namespace nano_vllm
