#include "kernels/rms_norm.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

bool can_vectorize_f32x4(const Tensor& input,
                         const Tensor& weight,
                         const Tensor& output,
                         int64_t cols) {
    return cols % 4 == 0 &&
           is_pointer_aligned(input.data(), 16) &&
           is_pointer_aligned(weight.data(), 16) &&
           is_pointer_aligned(output.data(), 16);
}

bool can_vectorize_f32x4_add(const Tensor& input,
                             const Tensor& residual,
                             const Tensor& weight,
                             const Tensor& output,
                             const Tensor& residual_out,
                             int64_t cols) {
    return cols % 4 == 0 &&
           is_pointer_aligned(input.data(), 16) &&
           is_pointer_aligned(residual.data(), 16) &&
           is_pointer_aligned(weight.data(), 16) &&
           is_pointer_aligned(output.data(), 16) &&
           is_pointer_aligned(residual_out.data(), 16);
}

// ---- kernels ----

template <typename InputT, typename WeightT>
__global__ void rms_norm_kernel(const InputT* input,
                                const WeightT* weight,
                                InputT* output,
                                int64_t rows,
                                int64_t cols,
                                float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const InputT* row_input = input + row * cols;
    InputT* row_output = output + row * cols;

    float sum_sq = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float value = to_float(row_input[col]);
        sum_sq += value * value;
    }

    const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / static_cast<float>(cols) + eps);
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float normalized = to_float(row_input[col]) * inv_rms;
        const float scaled = normalized * to_float(weight[col]);
        row_output[col] = from_float<InputT>(scaled);
    }
}

__global__ void rms_norm_f32x4_kernel(const float* input,
                                      const float* weight,
                                      float* output,
                                      int64_t rows,
                                      int64_t cols,
                                      float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t vec_cols = cols / 4;
    const float4* row_input = reinterpret_cast<const float4*>(input + row * cols);
    const float4* row_weight = reinterpret_cast<const float4*>(weight);
    float4* row_output = reinterpret_cast<float4*>(output + row * cols);

    float sum_sq = 0.0f;
    for (int64_t col = tid; col < vec_cols; col += blockDim.x) {
        const float4 x = row_input[col];
        sum_sq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }

    const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / static_cast<float>(cols) + eps);
    for (int64_t col = tid; col < vec_cols; col += blockDim.x) {
        const float4 x = row_input[col];
        const float4 w = row_weight[col];
        float4 y;
        y.x = x.x * inv_rms * w.x;
        y.y = x.y * inv_rms * w.y;
        y.z = x.z * inv_rms * w.z;
        y.w = x.w * inv_rms * w.w;
        row_output[col] = y;
    }
}

template <typename InputT, typename WeightT>
__global__ void add_rms_norm_kernel(const InputT* input,
                                    const InputT* residual,
                                    const WeightT* weight,
                                    InputT* output,
                                    InputT* residual_out,
                                    int64_t rows,
                                    int64_t cols,
                                    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const InputT* row_input = input + row * cols;
    const InputT* row_residual = residual + row * cols;
    InputT* row_output = output + row * cols;
    InputT* row_residual_out = residual_out + row * cols;

    float sum_sq = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float combined = to_float(row_input[col]) + to_float(row_residual[col]);
        sum_sq += combined * combined;
    }

    const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / static_cast<float>(cols) + eps);
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float combined = to_float(row_input[col]) + to_float(row_residual[col]);
        row_residual_out[col] = from_float<InputT>(combined);
        const float scaled = combined * inv_rms * to_float(weight[col]);
        row_output[col] = from_float<InputT>(scaled);
    }
}

__global__ void add_rms_norm_f32x4_kernel(const float* input,
                                          const float* residual,
                                          const float* weight,
                                          float* output,
                                          float* residual_out,
                                          int64_t rows,
                                          int64_t cols,
                                          float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t vec_cols = cols / 4;
    const float4* row_input = reinterpret_cast<const float4*>(input + row * cols);
    const float4* row_residual = reinterpret_cast<const float4*>(residual + row * cols);
    const float4* row_weight = reinterpret_cast<const float4*>(weight);
    float4* row_output = reinterpret_cast<float4*>(output + row * cols);
    float4* row_residual_out = reinterpret_cast<float4*>(residual_out + row * cols);

    float sum_sq = 0.0f;
    for (int64_t col = tid; col < vec_cols; col += blockDim.x) {
        const float4 x = row_input[col];
        const float4 r = row_residual[col];
        const float4 combined = {x.x + r.x, x.y + r.y, x.z + r.z, x.w + r.w};
        sum_sq += combined.x * combined.x + combined.y * combined.y +
                  combined.z * combined.z + combined.w * combined.w;
    }

    const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / static_cast<float>(cols) + eps);
    for (int64_t col = tid; col < vec_cols; col += blockDim.x) {
        const float4 x = row_input[col];
        const float4 r = row_residual[col];
        const float4 w = row_weight[col];
        const float4 combined = {x.x + r.x, x.y + r.y, x.z + r.z, x.w + r.w};
        row_residual_out[col] = combined;

        float4 y;
        y.x = combined.x * inv_rms * w.x;
        y.y = combined.y * inv_rms * w.y;
        y.z = combined.z * inv_rms * w.z;
        y.w = combined.w * inv_rms * w.w;
        row_output[col] = y;
    }
}

// ---- launch wrappers ----

template <typename InputT, typename WeightT>
void launch_rms_norm(const Tensor& input,
                     const Tensor& weight,
                     float eps,
                     Tensor& output,
                     cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("input", input);
    const int threads = choose_thread_count(shape.cols);
    rms_norm_kernel<InputT, WeightT><<<static_cast<unsigned int>(shape.rows), threads, 0, stream>>>(
        input.data_as<InputT>(),
        weight.data_as<WeightT>(),
        output.data_as<InputT>(),
        shape.rows,
        shape.cols,
        eps);
    throw_if_cuda_error(cudaPeekAtLastError(), "rms_norm_kernel launch");
}

void launch_rms_norm_f32(const Tensor& input,
                         const Tensor& weight,
                         float eps,
                         Tensor& output,
                         cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("input", input);
    if (can_vectorize_f32x4(input, weight, output, shape.cols)) {
        const int threads = choose_thread_count(shape.cols / 4);
        rms_norm_f32x4_kernel<<<static_cast<unsigned int>(shape.rows), threads, 0, stream>>>(
            input.data_as<float>(),
            weight.data_as<float>(),
            output.data_as<float>(),
            shape.rows,
            shape.cols,
            eps);
        throw_if_cuda_error(cudaPeekAtLastError(), "rms_norm_f32x4_kernel launch");
        return;
    }

    launch_rms_norm<float, float>(input, weight, eps, output, stream);
}

template <typename InputT, typename WeightT>
void launch_add_rms_norm(const Tensor& input,
                         const Tensor& residual,
                         const Tensor& weight,
                         float eps,
                         Tensor& output,
                         Tensor& residual_out,
                         cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("input", input);
    const int threads = choose_thread_count(shape.cols);
    add_rms_norm_kernel<InputT, WeightT><<<static_cast<unsigned int>(shape.rows), threads, 0, stream>>>(
        input.data_as<InputT>(),
        residual.data_as<InputT>(),
        weight.data_as<WeightT>(),
        output.data_as<InputT>(),
        residual_out.data_as<InputT>(),
        shape.rows,
        shape.cols,
        eps);
    throw_if_cuda_error(cudaPeekAtLastError(), "add_rms_norm_kernel launch");
}

void launch_add_rms_norm_f32(const Tensor& input,
                             const Tensor& residual,
                             const Tensor& weight,
                             float eps,
                             Tensor& output,
                             Tensor& residual_out,
                             cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("input", input);
    if (can_vectorize_f32x4_add(input, residual, weight, output, residual_out, shape.cols)) {
        const int threads = choose_thread_count(shape.cols / 4);
        add_rms_norm_f32x4_kernel<<<static_cast<unsigned int>(shape.rows), threads, 0, stream>>>(
            input.data_as<float>(),
            residual.data_as<float>(),
            weight.data_as<float>(),
            output.data_as<float>(),
            residual_out.data_as<float>(),
            shape.rows,
            shape.cols,
            eps);
        throw_if_cuda_error(cudaPeekAtLastError(), "add_rms_norm_f32x4_kernel launch");
        return;
    }

    launch_add_rms_norm<float, float>(input, residual, weight, eps, output, residual_out, stream);
}

} // namespace

// ---- public APIs ----

void rms_norm(const Tensor& input,
              const Tensor& weight,
              float eps,
              Tensor& output,
              cudaStream_t stream) {
    ensure_cuda_tensor("input", input);
    ensure_cuda_tensor("weight", weight);
    ensure_cuda_tensor("output", output);
    ensure_same_device("input", input, "weight", weight);
    ensure_same_device("input", input, "output", output);
    ensure_same_shape("input", input, "output", output);
    validate_supported_dtype("input", input.dtype());
    if (weight.dim() != 1) {
        throw std::invalid_argument("weight must be rank 1");
    }
    const MatrixShape shape = flatten_by_last_dim("input", input);
    if (weight.sizes()[0] != shape.cols) {
        throw std::invalid_argument("weight length must match the hidden dimension");
    }
    if (input.dtype() != output.dtype()) {
        throw std::invalid_argument("input and output must have the same dtype");
    }

    switch (input.dtype()) {
    case ScalarType::kFloat16:
        if (weight.dtype() == ScalarType::kFloat16) {
            launch_rms_norm<__half, __half>(input, weight, eps, output, stream);
            return;
        }
        if (weight.dtype() == ScalarType::kFloat32) {
            launch_rms_norm<__half, float>(input, weight, eps, output, stream);
            return;
        }
        break;
    case ScalarType::kBFloat16:
        if (weight.dtype() == ScalarType::kBFloat16) {
            launch_rms_norm<__nv_bfloat16, __nv_bfloat16>(input, weight, eps, output, stream);
            return;
        }
        if (weight.dtype() == ScalarType::kFloat32) {
            launch_rms_norm<__nv_bfloat16, float>(input, weight, eps, output, stream);
            return;
        }
        break;
    case ScalarType::kFloat32:
        if (weight.dtype() == ScalarType::kFloat32) {
            launch_rms_norm_f32(input, weight, eps, output, stream);
            return;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("unsupported dtype combination for rms_norm");
}

void add_rms_norm(const Tensor& input,
                  const Tensor& residual,
                  const Tensor& weight,
                  float eps,
                  Tensor& output,
                  Tensor& residual_out,
                  cudaStream_t stream) {
    ensure_cuda_tensor("input", input);
    ensure_cuda_tensor("residual", residual);
    ensure_cuda_tensor("weight", weight);
    ensure_cuda_tensor("output", output);
    ensure_cuda_tensor("residual_out", residual_out);
    ensure_same_device("input", input, "residual", residual);
    ensure_same_device("input", input, "weight", weight);
    ensure_same_device("input", input, "output", output);
    ensure_same_device("input", input, "residual_out", residual_out);
    ensure_same_shape("input", input, "residual", residual);
    ensure_same_shape("input", input, "output", output);
    ensure_same_shape("input", input, "residual_out", residual_out);
    ensure_same_dtype("input", input, "residual", residual);
    ensure_same_dtype("input", input, "output", output);
    ensure_same_dtype("input", input, "residual_out", residual_out);
    validate_supported_dtype("input", input.dtype());
    if (weight.dim() != 1) {
        throw std::invalid_argument("weight must be rank 1");
    }
    const MatrixShape shape = flatten_by_last_dim("input", input);
    if (weight.sizes()[0] != shape.cols) {
        throw std::invalid_argument("weight length must match the hidden dimension");
    }

    switch (input.dtype()) {
    case ScalarType::kFloat16:
        if (weight.dtype() == ScalarType::kFloat16) {
            launch_add_rms_norm<__half, __half>(input, residual, weight, eps, output, residual_out, stream);
            return;
        }
        if (weight.dtype() == ScalarType::kFloat32) {
            launch_add_rms_norm<__half, float>(input, residual, weight, eps, output, residual_out, stream);
            return;
        }
        break;
    case ScalarType::kBFloat16:
        if (weight.dtype() == ScalarType::kBFloat16) {
            launch_add_rms_norm<__nv_bfloat16, __nv_bfloat16>(input, residual, weight, eps, output, residual_out, stream);
            return;
        }
        if (weight.dtype() == ScalarType::kFloat32) {
            launch_add_rms_norm<__nv_bfloat16, float>(input, residual, weight, eps, output, residual_out, stream);
            return;
        }
        break;
    case ScalarType::kFloat32:
        if (weight.dtype() == ScalarType::kFloat32) {
            launch_add_rms_norm_f32(input, residual, weight, eps, output, residual_out, stream);
            return;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("unsupported dtype combination for add_rms_norm");
}

} // namespace nano_vllm
