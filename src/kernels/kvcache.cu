#include "kernels/kvcache.h"
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

// 16-byte vectorized variant: each thread copies one int4 (16 bytes) per step.
// For bf16/fp16 that is 8 elements/step; for fp32 it is 4 elements/step.
// Requires row_bytes % 16 == 0 and all four base pointers 16-byte aligned.
template <typename IndexT, typename DataT>
__global__ void store_kvcache_vec_kernel(const DataT* key,
                                         const DataT* value,
                                         DataT* k_cache,
                                         DataT* v_cache,
                                         const IndexT* slot_mapping,
                                         int64_t token_count,
                                         int64_t row_width,
                                         int64_t vec_per_row,
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

    const int4* key_vec = reinterpret_cast<const int4*>(key + token * row_width);
    const int4* val_vec = reinterpret_cast<const int4*>(value + token * row_width);
    int4* k_out = reinterpret_cast<int4*>(k_cache + slot * row_width);
    int4* v_out = reinterpret_cast<int4*>(v_cache + slot * row_width);

    for (int64_t i = tid; i < vec_per_row; i += blockDim.x) {
        k_out[i] = key_vec[i];
        v_out[i] = val_vec[i];
    }
}

template <typename DataT>
bool can_vectorize_store_kvcache_16B(const Tensor& key,
                                     const Tensor& value,
                                     const Tensor& k_cache,
                                     const Tensor& v_cache,
                                     int64_t row_width) {
    const int64_t row_bytes = row_width * static_cast<int64_t>(sizeof(DataT));
    return row_bytes % 16 == 0 &&
           is_pointer_aligned(key.data(), 16) &&
           is_pointer_aligned(value.data(), 16) &&
           is_pointer_aligned(k_cache.data(), 16) &&
           is_pointer_aligned(v_cache.data(), 16);
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

    if (can_vectorize_store_kvcache_16B<DataT>(key, value, k_cache, v_cache, key_shape.cols)) {
        const int64_t row_bytes = key_shape.cols * static_cast<int64_t>(sizeof(DataT));
        const int64_t vec_per_row = row_bytes / 16;
        store_kvcache_vec_kernel<IndexT, DataT><<<static_cast<unsigned int>(token_count),
                                                   kThreadsPerBlock, 0, stream>>>(
            key.data_as<DataT>(),
            value.data_as<DataT>(),
            k_cache.data_as<DataT>(),
            v_cache.data_as<DataT>(),
            slot_mapping.data_as<IndexT>(),
            token_count,
            key_shape.cols,
            vec_per_row,
            cache_rows);
        throw_if_cuda_error(cudaPeekAtLastError(), "store_kvcache_vec_kernel launch");
        return;
    }

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
// Int8 quantized store: per-head per-token symmetric quantization
// key/value: [token_count, num_kv_heads, head_dim] in compute dtype
// k_cache/v_cache: [total_slots, num_kv_heads, head_dim] int8
// k_scale/v_scale: [total_slots, num_kv_heads] float16
// ---------------------------------------------------------------------------
namespace {

template <typename InputT>
__global__ void store_kvcache_int8_kernel(const InputT* key,
                                          const InputT* value,
                                          int8_t* k_cache,
                                          int8_t* v_cache,
                                          __half* k_scale,
                                          __half* v_scale,
                                          const int32_t* slot_mapping,
                                          int64_t token_count,
                                          int num_kv_heads,
                                          int head_dim,
                                          int64_t cache_rows) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    const int kv_head = blockIdx.y;  // parallelize over heads
    const int tid = threadIdx.x;
    if (token >= token_count || kv_head >= num_kv_heads) return;

    const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
    if (slot < 0 || slot >= cache_rows) return;

    const int64_t row_width = static_cast<int64_t>(num_kv_heads) * head_dim;
    const int64_t head_offset = kv_head * head_dim;

    // --- Key: find absmax ---
    const InputT* key_head = key + token * row_width + head_offset;
    float local_absmax = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = fabsf(to_float(key_head[d]));
        if (v > local_absmax) local_absmax = v;
    }
    // Warp reduce max (block is one warp for head_dim <= 128)
    for (int off = 16; off >= 1; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, local_absmax, off);
        if (other > local_absmax) local_absmax = other;
    }
    float k_sc = local_absmax / 127.0f;
    k_sc = fmaxf(k_sc, 1e-10f);
    float k_inv_sc = 127.0f / fmaxf(local_absmax, 1e-10f);

    // Store quantized key
    int8_t* k_out = k_cache + (slot * num_kv_heads + kv_head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = to_float(key_head[d]);
        int q = __float2int_rn(v * k_inv_sc);
        q = max(-127, min(127, q));
        k_out[d] = static_cast<int8_t>(q);
    }
    if (tid == 0) {
        k_scale[slot * num_kv_heads + kv_head] = __float2half(k_sc);
    }

    // --- Value: find absmax ---
    const InputT* val_head = value + token * row_width + head_offset;
    float v_absmax = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = fabsf(to_float(val_head[d]));
        if (v > v_absmax) v_absmax = v;
    }
    for (int off = 16; off >= 1; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v_absmax, off);
        if (other > v_absmax) v_absmax = other;
    }
    float v_sc = v_absmax / 127.0f;
    v_sc = fmaxf(v_sc, 1e-10f);
    float v_inv_sc = 127.0f / fmaxf(v_absmax, 1e-10f);

    // Store quantized value
    int8_t* v_out = v_cache + (slot * num_kv_heads + kv_head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = to_float(val_head[d]);
        int q = __float2int_rn(v * v_inv_sc);
        q = max(-127, min(127, q));
        v_out[d] = static_cast<int8_t>(q);
    }
    if (tid == 0) {
        v_scale[slot * num_kv_heads + kv_head] = __float2half(v_sc);
    }
}

} // namespace

void store_kvcache_int8(const Tensor& key,
                        const Tensor& value,
                        Tensor& k_cache,
                        Tensor& v_cache,
                        Tensor& k_scale,
                        Tensor& v_scale,
                        const Tensor& slot_mapping,
                        cudaStream_t stream) {
    ensure_cuda_tensor("key", key);
    ensure_cuda_tensor("value", value);
    ensure_cuda_tensor("k_cache", k_cache);
    ensure_cuda_tensor("v_cache", v_cache);
    ensure_cuda_tensor("k_scale", k_scale);
    ensure_cuda_tensor("v_scale", v_scale);
    ensure_cuda_tensor("slot_mapping", slot_mapping);
    validate_supported_dtype("key", key.dtype());

    if (k_cache.dtype() != ScalarType::kInt8 || v_cache.dtype() != ScalarType::kInt8) {
        throw std::invalid_argument("store_kvcache_int8 requires int8 caches");
    }
    if (k_scale.dtype() != ScalarType::kFloat16 || v_scale.dtype() != ScalarType::kFloat16) {
        throw std::invalid_argument("store_kvcache_int8 requires float16 scales");
    }

    const int64_t token_count = static_cast<int64_t>(slot_mapping.numel());
    if (token_count <= 0) return;

    // key shape: [token_count, num_kv_heads, head_dim]
    if (key.dim() != 3) throw std::invalid_argument("key must be rank-3 for int8 store");
    const int num_kv_heads = static_cast<int>(key.sizes()[1]);
    const int head_dim = static_cast<int>(key.sizes()[2]);
    const int64_t cache_rows = static_cast<int64_t>(k_cache.sizes()[0]) *
                               static_cast<int64_t>(k_cache.sizes()[1]);

    // Launch: one block per (token, kv_head), 32 threads per block (one warp)
    dim3 grid(static_cast<unsigned int>(token_count), static_cast<unsigned int>(num_kv_heads));
    const int threads = 32; // one warp — sufficient for head_dim <= 128

    switch (key.dtype()) {
    case ScalarType::kFloat16:
        store_kvcache_int8_kernel<__half><<<grid, threads, 0, stream>>>(
            key.data_as<__half>(), value.data_as<__half>(),
            k_cache.data_as<int8_t>(), v_cache.data_as<int8_t>(),
            k_scale.data_as<__half>(), v_scale.data_as<__half>(),
            slot_mapping.data_as<int32_t>(),
            token_count, num_kv_heads, head_dim, cache_rows);
        break;
    case ScalarType::kBFloat16:
        store_kvcache_int8_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            key.data_as<__nv_bfloat16>(), value.data_as<__nv_bfloat16>(),
            k_cache.data_as<int8_t>(), v_cache.data_as<int8_t>(),
            k_scale.data_as<__half>(), v_scale.data_as<__half>(),
            slot_mapping.data_as<int32_t>(),
            token_count, num_kv_heads, head_dim, cache_rows);
        break;
    case ScalarType::kFloat32:
        store_kvcache_int8_kernel<float><<<grid, threads, 0, stream>>>(
            key.data_as<float>(), value.data_as<float>(),
            k_cache.data_as<int8_t>(), v_cache.data_as<int8_t>(),
            k_scale.data_as<__half>(), v_scale.data_as<__half>(),
            slot_mapping.data_as<int32_t>(),
            token_count, num_kv_heads, head_dim, cache_rows);
        break;
    default:
        throw std::invalid_argument("unsupported input dtype for store_kvcache_int8");
    }
    throw_if_cuda_error(cudaPeekAtLastError(), "store_kvcache_int8_kernel launch");
}

} // namespace nano_vllm
