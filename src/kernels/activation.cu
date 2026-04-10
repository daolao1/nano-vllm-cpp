#include "kernels/activation.h"
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
__global__ void silu_and_mul_kernel(const T* gate_up,
                                    T* output,
                                    int64_t rows,
                                    int64_t output_cols) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t input_cols = output_cols * 2;
    const T* row_input = gate_up + row * input_cols;
    T* row_output = output + row * output_cols;

    for (int64_t col = tid; col < output_cols; col += blockDim.x) {
        const float gate = to_float(row_input[col]);
        const float up = to_float(row_input[col + output_cols]);
        const float silu = gate / (1.0f + expf(-gate));
        row_output[col] = from_float<T>(silu * up);
    }
}

template <typename T>
void launch_silu_and_mul(const Tensor& gate_up, Tensor& output, cudaStream_t stream) {
    const MatrixShape input_shape = flatten_by_last_dim("gate_up", gate_up);
    silu_and_mul_kernel<T><<<static_cast<unsigned int>(input_shape.rows), kThreadsPerBlock, 0, stream>>>(
        gate_up.data_as<T>(),
        output.data_as<T>(),
        input_shape.rows,
        input_shape.cols / 2);
    throw_if_cuda_error(cudaPeekAtLastError(), "silu_and_mul_kernel launch");
}

} // namespace

void silu_and_mul(const Tensor& gate_up,
                  Tensor& output,
                  cudaStream_t stream) {
    ensure_cuda_tensor("gate_up", gate_up);
    ensure_cuda_tensor("output", output);
    ensure_same_device("gate_up", gate_up, "output", output);
    validate_supported_dtype("gate_up", gate_up.dtype());
    if (gate_up.dtype() != output.dtype()) {
        throw std::invalid_argument("gate_up and output must have the same dtype");
    }
    if (gate_up.dim() < 1 || output.dim() < 1) {
        throw std::invalid_argument("gate_up and output must have rank >= 1");
    }

    const MatrixShape input_shape = flatten_by_last_dim("gate_up", gate_up);
    const MatrixShape output_shape = flatten_by_last_dim("output", output);
    if (input_shape.cols % 2 != 0) {
        throw std::invalid_argument("gate_up last dimension must be even");
    }
    if (input_shape.rows != output_shape.rows || input_shape.cols / 2 != output_shape.cols) {
        throw std::invalid_argument("output must match gate_up with the last dimension halved");
    }

    switch (gate_up.dtype()) {
    case ScalarType::kFloat16:
        launch_silu_and_mul<__half>(gate_up, output, stream);
        return;
    case ScalarType::kBFloat16:
        launch_silu_and_mul<__nv_bfloat16>(gate_up, output, stream);
        return;
    case ScalarType::kFloat32:
        launch_silu_and_mul<float>(gate_up, output, stream);
        return;
    default:
        break;
    }

    throw std::invalid_argument("unsupported dtype for silu_and_mul");
}

} // namespace nano_vllm
