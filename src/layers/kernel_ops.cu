#include "layers/kernel_ops.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

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

template <typename IndexT, typename WeightT>
__global__ void embedding_lookup_kernel(const IndexT* input_ids,
                                        const WeightT* weight,
                                        float* output,
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
        output[index] = 0.0f;
        return;
    }
    if (token < vocab_start || token >= vocab_start + local_vocab_size) {
        output[index] = 0.0f;
        return;
    }

    output[index] = to_float(weight[(token - vocab_start) * hidden_dim + col]);
}

__global__ void sample_tokens_kernel(const float* logits,
                                     const float* temperatures,
                                     int32_t* output,
                                     int64_t rows,
                                     int64_t cols,
                                     uint64_t seed) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float shared_scores[kThreadsPerBlock];
    __shared__ int shared_indices[kThreadsPerBlock];

    const float temperature = fmaxf(temperatures[row], 1e-10f);
    float best_score = -FLT_MAX;
    int best_index = 0;
    const float* row_logits = logits + row * cols;

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float scaled_logit = row_logits[col] / temperature;
        const uint64_t rng_state = seed ^
                                   (static_cast<uint64_t>(row) + 1ull) * 0x9e3779b97f4a7c15ull ^
                                   (static_cast<uint64_t>(col) + 1ull);
        const float uniform = uniform01_from_u64(rng_state);
        const float exp_sample = fmaxf(-logf(uniform), 1e-10f);
        const float score = scaled_logit - logf(exp_sample);
        update_best(score, static_cast<int>(col), best_score, best_index);
    }

    shared_scores[tid] = best_score;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_best(shared_scores[tid + stride],
                        shared_indices[tid + stride],
                        shared_scores[tid],
                        shared_indices[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = static_cast<int32_t>(shared_indices[0]);
    }
}

template <typename IndexT, typename DataT>
__global__ void store_kvcache_kernel(const DataT* key,
                                     const DataT* value,
                                     DataT* k_cache,
                                     DataT* v_cache,
                                     const IndexT* slot_mapping,
                                     int64_t token_count,
                                     int64_t row_width,
                                     int64_t cache_rows) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (token >= token_count) {
        return;
    }

    const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
    if (slot < 0 || slot >= cache_rows) {
        return;
    }

    const DataT* key_row = key + token * row_width;
    const DataT* value_row = value + token * row_width;
    DataT* k_cache_row = k_cache + slot * row_width;
    DataT* v_cache_row = v_cache + slot * row_width;

    for (int64_t col = tid; col < row_width; col += blockDim.x) {
        k_cache_row[col] = key_row[col];
        v_cache_row[col] = value_row[col];
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

template <typename IndexT, typename WeightT>
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
    embedding_lookup_kernel<IndexT, WeightT><<<blocks, kThreadsPerBlock, 0, stream>>>(
        input_ids.data_as<IndexT>(),
        weight.data_as<WeightT>(),
        output.data_as<float>(),
        token_count,
        hidden_dim,
        vocab_size,
        vocab_start,
        weight.sizes()[0]);
    throw_if_cuda_error(cudaPeekAtLastError(), "embedding_lookup_kernel launch");
}

void launch_sample_tokens(const Tensor& logits,
                          const Tensor& temperatures,
                          uint64_t seed,
                          Tensor& output,
                          cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("logits", logits);
    if (shape.rows == 0) {
        return;
    }

    sample_tokens_kernel<<<static_cast<unsigned int>(shape.rows), kThreadsPerBlock, 0, stream>>>(
        logits.data_as<float>(),
        temperatures.data_as<float>(),
        output.data_as<int32_t>(),
        shape.rows,
        shape.cols,
        seed);
    throw_if_cuda_error(cudaPeekAtLastError(), "sample_tokens_kernel launch");
}

template <typename IndexT, typename DataT>
void launch_store_kvcache(const Tensor& key,
                          const Tensor& value,
                          Tensor& k_cache,
                          Tensor& v_cache,
                          const Tensor& slot_mapping,
                          cudaStream_t stream) {
    const int64_t token_count = static_cast<int64_t>(slot_mapping.numel());
    const MatrixShape key_shape = flatten_by_leading_count("key", key, token_count);
    const int64_t cache_rows = cache_rows_from_tensor("k_cache", k_cache, key_shape.cols);

    store_kvcache_kernel<IndexT, DataT><<<static_cast<unsigned int>(token_count), kThreadsPerBlock, 0, stream>>>(
        key.data_as<DataT>(),
        value.data_as<DataT>(),
        k_cache.data_as<DataT>(),
        v_cache.data_as<DataT>(),
        slot_mapping.data_as<IndexT>(),
        token_count,
        key_shape.cols,
        cache_rows);
    throw_if_cuda_error(cudaPeekAtLastError(), "store_kvcache_kernel launch");
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
    if (output.dtype() != ScalarType::kFloat32) {
        throw std::invalid_argument("output must be float32");
    }
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
            launch_embedding_lookup<int32_t, __half>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kBFloat16:
            launch_embedding_lookup<int32_t, __nv_bfloat16>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kFloat32:
            launch_embedding_lookup<int32_t, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        default:
            break;
        }
    } else {
        switch (weight.dtype()) {
        case ScalarType::kFloat16:
            launch_embedding_lookup<int64_t, __half>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kBFloat16:
            launch_embedding_lookup<int64_t, __nv_bfloat16>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        case ScalarType::kFloat32:
            launch_embedding_lookup<int64_t, float>(input_ids, weight, vocab_size, vocab_start, output, stream);
            return;
        default:
            break;
        }
    }

    throw std::invalid_argument("unsupported dtype combination for embedding_lookup");
}

void sample_tokens(const Tensor& logits,
                   const Tensor& temperatures,
                   uint64_t seed,
                   Tensor& output,
                   cudaStream_t stream) {
    ensure_cuda_tensor("logits", logits);
    ensure_cuda_tensor("temperatures", temperatures);
    ensure_cuda_tensor("output", output);
    ensure_same_device("logits", logits, "temperatures", temperatures);
    ensure_same_device("logits", logits, "output", output);
    if (logits.dtype() != ScalarType::kFloat32 || temperatures.dtype() != ScalarType::kFloat32) {
        throw std::invalid_argument("sample_tokens currently expects float32 logits and temperatures");
    }
    if (output.dtype() != ScalarType::kInt32) {
        throw std::invalid_argument("output must be int32");
    }
    if (logits.dim() != 2) {
        throw std::invalid_argument("logits must have shape [batch_size, vocab_size]");
    }
    if (temperatures.dim() != 1 || temperatures.numel() != static_cast<size_t>(logits.sizes()[0])) {
        throw std::invalid_argument("temperatures must have shape [batch_size]");
    }
    if (output.dim() != 1 || output.numel() != static_cast<size_t>(logits.sizes()[0])) {
        throw std::invalid_argument("output must have shape [batch_size]");
    }
    if (logits.sizes()[1] <= 0) {
        throw std::invalid_argument("logits vocab dimension must be positive");
    }

    launch_sample_tokens(logits, temperatures, seed, output, stream);
}

void store_kvcache(const Tensor& key,
                   const Tensor& value,
                   Tensor& k_cache,
                   Tensor& v_cache,
                   const Tensor& slot_mapping,
                   cudaStream_t stream) {
    ensure_cuda_tensor("key", key);
    ensure_cuda_tensor("value", value);
    ensure_cuda_tensor("k_cache", k_cache);
    ensure_cuda_tensor("v_cache", v_cache);
    ensure_cuda_tensor("slot_mapping", slot_mapping);
    ensure_same_device("key", key, "value", value);
    ensure_same_device("key", key, "k_cache", k_cache);
    ensure_same_device("key", key, "v_cache", v_cache);
    ensure_same_device("key", key, "slot_mapping", slot_mapping);
    ensure_same_dtype("key", key, "value", value);
    ensure_same_dtype("key", key, "k_cache", k_cache);
    ensure_same_dtype("key", key, "v_cache", v_cache);
    validate_supported_dtype("key", key.dtype());
    if (slot_mapping.dtype() != ScalarType::kInt32 && slot_mapping.dtype() != ScalarType::kInt64) {
        throw std::invalid_argument("slot_mapping must be int32 or int64");
    }

    const int64_t token_count = static_cast<int64_t>(slot_mapping.numel());
    if (token_count <= 0) {
        throw std::invalid_argument("slot_mapping must contain at least one element");
    }

    const MatrixShape key_shape = flatten_by_leading_count("key", key, token_count);
    const MatrixShape value_shape = flatten_by_leading_count("value", value, token_count);
    if (key_shape.cols != value_shape.cols) {
        throw std::invalid_argument("key and value must have the same flattened row width");
    }
    const int64_t k_cache_rows = cache_rows_from_tensor("k_cache", k_cache, key_shape.cols);
    const int64_t v_cache_rows = cache_rows_from_tensor("v_cache", v_cache, key_shape.cols);
    if (k_cache_rows != v_cache_rows) {
        throw std::invalid_argument("k_cache and v_cache must contain the same number of slots");
    }

    if (slot_mapping.dtype() == ScalarType::kInt32) {
        switch (key.dtype()) {
        case ScalarType::kFloat16:
            launch_store_kvcache<int32_t, __half>(key, value, k_cache, v_cache, slot_mapping, stream);
            return;
        case ScalarType::kBFloat16:
            launch_store_kvcache<int32_t, __nv_bfloat16>(key, value, k_cache, v_cache, slot_mapping, stream);
            return;
        case ScalarType::kFloat32:
            launch_store_kvcache<int32_t, float>(key, value, k_cache, v_cache, slot_mapping, stream);
            return;
        default:
            break;
        }
    } else {
        switch (key.dtype()) {
        case ScalarType::kFloat16:
            launch_store_kvcache<int64_t, __half>(key, value, k_cache, v_cache, slot_mapping, stream);
            return;
        case ScalarType::kBFloat16:
            launch_store_kvcache<int64_t, __nv_bfloat16>(key, value, k_cache, v_cache, slot_mapping, stream);
            return;
        case ScalarType::kFloat32:
            launch_store_kvcache<int64_t, float>(key, value, k_cache, v_cache, slot_mapping, stream);
            return;
        default:
            break;
        }
    }

    throw std::invalid_argument("unsupported dtype combination for store_kvcache");
}

// ---------------------------------------------------------------------------
// split_qkv: split packed [rows, q_size + 2*kv_size] into q, k, v in one pass.
// ---------------------------------------------------------------------------

__global__ void split_qkv_kernel(const float* __restrict__ qkv,
                                 float* __restrict__ q,
                                 float* __restrict__ k,
                                 float* __restrict__ v,
                                 int q_size, int kv_size, int total_cols, int rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * total_cols;
    if (idx >= total) return;

    const int row = idx / total_cols;
    const int col = idx % total_cols;
    const float val = qkv[idx];

    if (col < q_size) {
        q[row * q_size + col] = val;
    } else if (col < q_size + kv_size) {
        k[row * kv_size + (col - q_size)] = val;
    } else {
        v[row * kv_size + (col - q_size - kv_size)] = val;
    }
}

void split_qkv(const Tensor& qkv,
               int q_size,
               int kv_size,
               Tensor& q,
               Tensor& k,
               Tensor& v,
               cudaStream_t stream) {
    const int rows = static_cast<int>(qkv.sizes()[0]);
    const int total_cols = q_size + 2 * kv_size;
    const int total = rows * total_cols;
    constexpr int kBlock = 256;
    const int grid = (total + kBlock - 1) / kBlock;
    split_qkv_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const float*>(qkv.data()),
        static_cast<float*>(q.data()),
        static_cast<float*>(k.data()),
        static_cast<float*>(v.data()),
        q_size, kv_size, total_cols, rows);
}

} // namespace nano_vllm