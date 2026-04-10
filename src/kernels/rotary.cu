#include "kernels/rotary.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

int64_t rotary_cache_positions(const Tensor& cache, int64_t head_dim) {
    if (cache.dim() == 2) {
        if (cache.sizes()[1] != head_dim) {
            throw std::invalid_argument("cos_sin_cache last dimension must equal head_dim");
        }
        return cache.sizes()[0];
    }
    if (cache.dim() == 3) {
        if (cache.sizes()[1] != 1 || cache.sizes()[2] != head_dim) {
            throw std::invalid_argument("3D cos_sin_cache must have shape [max_pos, 1, head_dim]");
        }
        return cache.sizes()[0];
    }
    throw std::invalid_argument("cos_sin_cache must be rank 2 or 3");
}

bool can_vectorize_rotary_f32(const Tensor& cache, const Tensor& data, int64_t head_dim) {
    return head_dim % 4 == 0 &&
           is_pointer_aligned(cache.data(), 8) &&
           is_pointer_aligned(data.data(), 8);
}

// ---- kernels ----

template <typename PositionT, typename DataT, typename CacheT>
__global__ void apply_rotary_kernel(const PositionT* positions,
                                    const CacheT* cos_sin_cache,
                                    DataT* data,
                                    int64_t token_count,
                                    int64_t heads_per_token,
                                    int64_t head_dim,
                                    int64_t max_positions) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= token_count * heads_per_token) {
        return;
    }

    const int64_t half_dim = head_dim / 2;
    const int64_t token_index = row / heads_per_token;
    const int64_t cache_position = static_cast<int64_t>(positions[token_index]);
    if (cache_position < 0 || cache_position >= max_positions) {
        return;
    }

    DataT* row_data = data + row * head_dim;
    const CacheT* row_cache = cos_sin_cache + cache_position * head_dim;

    for (int64_t idx = tid; idx < half_dim; idx += blockDim.x) {
        const float cos_value = to_float(row_cache[idx]);
        const float sin_value = to_float(row_cache[idx + half_dim]);
        const float x1 = to_float(row_data[idx]);
        const float x2 = to_float(row_data[idx + half_dim]);
        row_data[idx] = from_float<DataT>(x1 * cos_value - x2 * sin_value);
        row_data[idx + half_dim] = from_float<DataT>(x2 * cos_value + x1 * sin_value);
    }
}

template <typename PositionT>
__global__ void apply_rotary_f32x2_kernel(const PositionT* positions,
                                          const float* cos_sin_cache,
                                          float* data,
                                          int64_t token_count,
                                          int64_t heads_per_token,
                                          int64_t head_dim,
                                          int64_t max_positions) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= token_count * heads_per_token) {
        return;
    }

    const int64_t half_dim = head_dim / 2;
    const int64_t packed_pairs = half_dim / 2;
    const int64_t token_index = row / heads_per_token;
    const int64_t cache_position = static_cast<int64_t>(positions[token_index]);
    if (cache_position < 0 || cache_position >= max_positions) {
        return;
    }

    float* row_data = data + row * head_dim;
    const float* row_cache = cos_sin_cache + cache_position * head_dim;
    float2* row_first_half = reinterpret_cast<float2*>(row_data);
    float2* row_second_half = reinterpret_cast<float2*>(row_data + half_dim);
    const float2* row_cos = reinterpret_cast<const float2*>(row_cache);
    const float2* row_sin = reinterpret_cast<const float2*>(row_cache + half_dim);

    for (int64_t idx = tid; idx < packed_pairs; idx += blockDim.x) {
        const float2 x1 = row_first_half[idx];
        const float2 x2 = row_second_half[idx];
        const float2 cos_v = row_cos[idx];
        const float2 sin_v = row_sin[idx];

        float2 out_first;
        out_first.x = x1.x * cos_v.x - x2.x * sin_v.x;
        out_first.y = x1.y * cos_v.y - x2.y * sin_v.y;
        float2 out_second;
        out_second.x = x2.x * cos_v.x + x1.x * sin_v.x;
        out_second.y = x2.y * cos_v.y + x1.y * sin_v.y;

        row_first_half[idx] = out_first;
        row_second_half[idx] = out_second;
    }
}

// ---- launch wrappers ----

template <typename PositionT, typename DataT, typename CacheT>
void launch_rotary_one(const Tensor& positions,
                       const Tensor& cos_sin_cache,
                       Tensor& data,
                       cudaStream_t stream) {
    const int64_t token_count = static_cast<int64_t>(positions.numel());
    const int64_t head_dim = data.sizes().back();
    const int64_t rows = static_cast<int64_t>(data.numel() / static_cast<size_t>(head_dim));
    const int64_t heads_per_token = rows / token_count;
    const int64_t max_positions = rotary_cache_positions(cos_sin_cache, head_dim);
    const int threads = choose_thread_count(head_dim / 2);

    apply_rotary_kernel<PositionT, DataT, CacheT><<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
        positions.data_as<PositionT>(),
        cos_sin_cache.data_as<CacheT>(),
        data.data_as<DataT>(),
        token_count,
        heads_per_token,
        head_dim,
        max_positions);
    throw_if_cuda_error(cudaPeekAtLastError(), "apply_rotary_kernel launch");
}

template <typename PositionT>
void launch_rotary_one_f32(const Tensor& positions,
                           const Tensor& cos_sin_cache,
                           Tensor& data,
                           cudaStream_t stream) {
    const int64_t token_count = static_cast<int64_t>(positions.numel());
    const int64_t head_dim = data.sizes().back();
    const int64_t rows = static_cast<int64_t>(data.numel() / static_cast<size_t>(head_dim));
    const int64_t heads_per_token = rows / token_count;
    const int64_t max_positions = rotary_cache_positions(cos_sin_cache, head_dim);

    if (can_vectorize_rotary_f32(cos_sin_cache, data, head_dim)) {
        const int threads = choose_thread_count((head_dim / 2) / 2);
        apply_rotary_f32x2_kernel<PositionT><<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
            positions.data_as<PositionT>(),
            cos_sin_cache.data_as<float>(),
            data.data_as<float>(),
            token_count,
            heads_per_token,
            head_dim,
            max_positions);
        throw_if_cuda_error(cudaPeekAtLastError(), "apply_rotary_f32x2_kernel launch");
        return;
    }

    launch_rotary_one<PositionT, float, float>(positions, cos_sin_cache, data, stream);
}

} // namespace

// ---- public API ----

void rotary_embedding(const Tensor& positions,
                      const Tensor& cos_sin_cache,
                      Tensor& query,
                      Tensor& key,
                      cudaStream_t stream) {
    ensure_cuda_tensor("positions", positions);
    ensure_cuda_tensor("cos_sin_cache", cos_sin_cache);
    ensure_cuda_tensor("query", query);
    ensure_cuda_tensor("key", key);
    ensure_same_device("positions", positions, "cos_sin_cache", cos_sin_cache);
    ensure_same_device("positions", positions, "query", query);
    ensure_same_device("positions", positions, "key", key);
    ensure_same_dtype("query", query, "key", key);
    validate_supported_dtype("query", query.dtype());
    if (positions.dtype() != ScalarType::kInt32 && positions.dtype() != ScalarType::kInt64) {
        throw std::invalid_argument("positions must be int32 or int64");
    }
    if (query.dim() < 2 || key.dim() < 2) {
        throw std::invalid_argument("query and key must have rank >= 2");
    }
    if (query.sizes().back() != key.sizes().back()) {
        throw std::invalid_argument("query and key must have the same head_dim");
    }
    const int64_t token_count = static_cast<int64_t>(positions.numel());
    const int64_t head_dim = query.sizes().back();
    if (head_dim <= 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("head_dim must be positive and even");
    }
    if (token_count <= 0) {
        throw std::invalid_argument("positions must contain at least one token index");
    }
    if (static_cast<int64_t>(query.numel()) % (token_count * head_dim) != 0) {
        throw std::invalid_argument("query shape is incompatible with positions");
    }
    if (static_cast<int64_t>(key.numel()) % (token_count * head_dim) != 0) {
        throw std::invalid_argument("key shape is incompatible with positions");
    }
    (void)rotary_cache_positions(cos_sin_cache, head_dim);

    const bool cache_is_float = cos_sin_cache.dtype() == ScalarType::kFloat32;
    const bool cache_matches_data = cos_sin_cache.dtype() == query.dtype();
    if (!cache_is_float && !cache_matches_data) {
        throw std::invalid_argument("cos_sin_cache must be float32 or match query/key dtype");
    }

    if (positions.dtype() == ScalarType::kInt32) {
        switch (query.dtype()) {
        case ScalarType::kFloat16:
            if (cache_is_float) {
                launch_rotary_one<int32_t, __half, float>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int32_t, __half, float>(positions, cos_sin_cache, key, stream);
            } else {
                launch_rotary_one<int32_t, __half, __half>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int32_t, __half, __half>(positions, cos_sin_cache, key, stream);
            }
            return;
        case ScalarType::kBFloat16:
            if (cache_is_float) {
                launch_rotary_one<int32_t, __nv_bfloat16, float>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int32_t, __nv_bfloat16, float>(positions, cos_sin_cache, key, stream);
            } else {
                launch_rotary_one<int32_t, __nv_bfloat16, __nv_bfloat16>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int32_t, __nv_bfloat16, __nv_bfloat16>(positions, cos_sin_cache, key, stream);
            }
            return;
        case ScalarType::kFloat32:
            launch_rotary_one_f32<int32_t>(positions, cos_sin_cache, query, stream);
            launch_rotary_one_f32<int32_t>(positions, cos_sin_cache, key, stream);
            return;
        default:
            break;
        }
    } else {
        switch (query.dtype()) {
        case ScalarType::kFloat16:
            if (cache_is_float) {
                launch_rotary_one<int64_t, __half, float>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int64_t, __half, float>(positions, cos_sin_cache, key, stream);
            } else {
                launch_rotary_one<int64_t, __half, __half>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int64_t, __half, __half>(positions, cos_sin_cache, key, stream);
            }
            return;
        case ScalarType::kBFloat16:
            if (cache_is_float) {
                launch_rotary_one<int64_t, __nv_bfloat16, float>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int64_t, __nv_bfloat16, float>(positions, cos_sin_cache, key, stream);
            } else {
                launch_rotary_one<int64_t, __nv_bfloat16, __nv_bfloat16>(positions, cos_sin_cache, query, stream);
                launch_rotary_one<int64_t, __nv_bfloat16, __nv_bfloat16>(positions, cos_sin_cache, key, stream);
            }
            return;
        case ScalarType::kFloat32:
            launch_rotary_one_f32<int64_t>(positions, cos_sin_cache, query, stream);
            launch_rotary_one_f32<int64_t>(positions, cos_sin_cache, key, stream);
            return;
        default:
            break;
        }
    }

    throw std::invalid_argument("unsupported dtype combination for rotary_embedding");
}

} // namespace nano_vllm
