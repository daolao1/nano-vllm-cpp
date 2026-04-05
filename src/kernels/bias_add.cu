#include "kernels/bias_add.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

template <typename T>
__global__ void add_bias_kernel(T* output,
                                const T* bias,
                                int64_t total,
                                int64_t cols) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int64_t col = index % cols;
    output[index] = from_float<T>(to_float(output[index]) + to_float(bias[col]));
}

template <typename T>
__global__ void add_inplace_kernel(T* input_output,
                                   const T* input,
                                   int64_t total) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    input_output[index] = from_float<T>(to_float(input_output[index]) + to_float(input[index]));
}

template <typename T>
void launch_add_bias(Tensor& input_output, const Tensor& bias, cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("input_output", input_output);
    const int64_t total = shape.rows * shape.cols;
    if (total == 0) {
        return;
    }

    const unsigned int blocks = static_cast<unsigned int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    add_bias_kernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
        input_output.data_as<T>(),
        bias.data_as<T>(),
        total,
        shape.cols);
    throw_if_cuda_error(cudaPeekAtLastError(), "add_bias_kernel launch");
}

template <typename T>
void launch_add_inplace(Tensor& input_output, const Tensor& input, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(input_output.numel());
    if (total == 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    add_inplace_kernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
        input_output.data_as<T>(),
        input.data_as<T>(),
        total);
    throw_if_cuda_error(cudaPeekAtLastError(), "add_inplace_kernel launch");
}

} // namespace

void add_bias(Tensor& input_output,
              const Tensor& bias,
              cudaStream_t stream) {
    ensure_cuda_tensor("input_output", input_output);
    ensure_cuda_tensor("bias", bias);
    ensure_same_device("input_output", input_output, "bias", bias);
    ensure_same_dtype("input_output", input_output, "bias", bias);
    validate_supported_dtype("input_output", input_output.dtype());
    if (bias.dim() != 1) {
        throw std::invalid_argument("bias must be rank 1");
    }

    const MatrixShape shape = flatten_by_last_dim("input_output", input_output);
    if (bias.sizes()[0] != shape.cols) {
        throw std::invalid_argument("bias length must match the last dimension of input_output");
    }

    switch (input_output.dtype()) {
    case ScalarType::kFloat16:
        launch_add_bias<__half>(input_output, bias, stream);
        return;
    case ScalarType::kBFloat16:
        launch_add_bias<__nv_bfloat16>(input_output, bias, stream);
        return;
    case ScalarType::kFloat32:
        launch_add_bias<float>(input_output, bias, stream);
        return;
    default:
        break;
    }

    throw std::invalid_argument("unsupported dtype for add_bias");
}

void add_inplace(Tensor& input_output,
                 const Tensor& input,
                 cudaStream_t stream) {
    ensure_cuda_tensor("input_output", input_output);
    ensure_cuda_tensor("input", input);
    ensure_same_device("input_output", input_output, "input", input);
    ensure_same_shape("input_output", input_output, "input", input);
    ensure_same_dtype("input_output", input_output, "input", input);
    validate_supported_dtype("input_output", input_output.dtype());

    switch (input_output.dtype()) {
    case ScalarType::kFloat16:
        launch_add_inplace<__half>(input_output, input, stream);
        return;
    case ScalarType::kBFloat16:
        launch_add_inplace<__nv_bfloat16>(input_output, input, stream);
        return;
    case ScalarType::kFloat32:
        launch_add_inplace<float>(input_output, input, stream);
        return;
    default:
        break;
    }

    throw std::invalid_argument("unsupported dtype for add_inplace");
}

} // namespace nano_vllm
