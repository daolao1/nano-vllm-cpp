#include "kernels/split_qkv.h"
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
__global__ void split_qkv_kernel(const T* __restrict__ qkv,
                                 T* __restrict__ q,
                                 T* __restrict__ k,
                                 T* __restrict__ v,
                                 int q_size, int kv_size, int total_cols, int rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * total_cols;
    if (idx >= total) return;

    const int row = idx / total_cols;
    const int col = idx % total_cols;
    const T val = qkv[idx];

    if (col < q_size) {
        q[row * q_size + col] = val;
    } else if (col < q_size + kv_size) {
        k[row * kv_size + (col - q_size)] = val;
    } else {
        v[row * kv_size + (col - q_size - kv_size)] = val;
    }
}

template <typename T>
void launch_split_qkv(const Tensor& qkv, int q_size, int kv_size,
                      Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const int rows = static_cast<int>(qkv.sizes()[0]);
    const int total_cols = q_size + 2 * kv_size;
    const int total = rows * total_cols;
    constexpr int kBlock = 256;
    const int grid = (total + kBlock - 1) / kBlock;
    split_qkv_kernel<T><<<grid, kBlock, 0, stream>>>(
        qkv.data_as<T>(), q.data_as<T>(), k.data_as<T>(), v.data_as<T>(),
        q_size, kv_size, total_cols, rows);
    throw_if_cuda_error(cudaPeekAtLastError(), "split_qkv_kernel launch");
}

// ---------------------------------------------------------------------------
// Fused kernel: split QKV + RMSNorm(Q,K) + RoPE(Q,K) + store K/V to KV cache
// ---------------------------------------------------------------------------
// Grid: (rows, num_q_heads + 2 * num_kv_heads)
// Block: head_dim threads (e.g. 128)
// Each block processes one (row, head) pair.
template <typename T>
__global__ void fused_split_norm_rope_store_kernel(
    const T* __restrict__ qkv,
    const int32_t* __restrict__ positions,
    const T* __restrict__ q_norm_weight,
    const T* __restrict__ k_norm_weight,
    const float* __restrict__ cos_sin_cache,
    const int32_t* __restrict__ slot_mapping,
    T* __restrict__ q_out,
    T* __restrict__ k_cache,
    T* __restrict__ v_cache,
    int rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_size,
    int kv_size,
    int total_cols,
    float norm_eps,
    int max_positions,
    int cache_row_width)   // num_kv_heads * head_dim
{
    const int row = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (row >= rows || tid >= head_dim) return;

    const int total_heads = num_q_heads + 2 * num_kv_heads;
    if (head_idx >= total_heads) return;

    const bool is_q = (head_idx < num_q_heads);
    const bool is_k = (!is_q && head_idx < num_q_heads + num_kv_heads);
    // is_v = remainder

    // Determine source offset in QKV row
    int src_head;
    int src_offset;
    if (is_q) {
        src_head = head_idx;
        src_offset = src_head * head_dim;
    } else if (is_k) {
        src_head = head_idx - num_q_heads;
        src_offset = q_size + src_head * head_dim;
    } else {
        src_head = head_idx - num_q_heads - num_kv_heads;
        src_offset = q_size + kv_size + src_head * head_dim;
    }

    // Read element from QKV
    float val = to_float(qkv[row * total_cols + src_offset + tid]);

    if (is_q || is_k) {
        // --- RMSNorm ---
        float sum_sq = val * val;
        sum_sq = block_reduce_sum(sum_sq);
        float inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + norm_eps);
        const T* weight = is_q ? q_norm_weight : k_norm_weight;
        val = val * inv_rms * to_float(weight[tid]);

        // --- RoPE ---
        const int pos = positions[row];
        if (pos >= 0 && pos < max_positions) {
            // Store normalized values in shared memory for cross-element RoPE
            extern __shared__ float rope_smem[];
            rope_smem[tid] = val;
            __syncthreads();

            const int half_dim = head_dim / 2;
            const float* cache_row = cos_sin_cache + pos * head_dim;
            const int cos_idx = (tid < half_dim) ? tid : (tid - half_dim);
            const float cos_v = cache_row[cos_idx];
            const float sin_v = cache_row[cos_idx + half_dim];

            if (tid < half_dim) {
                val = rope_smem[tid] * cos_v - rope_smem[tid + half_dim] * sin_v;
            } else {
                val = rope_smem[tid] * cos_v + rope_smem[tid - half_dim] * sin_v;
            }
        }
    }

    // --- Write output ---
    if (is_q) {
        q_out[row * num_q_heads * head_dim + src_head * head_dim + tid] = from_float<T>(val);
    } else {
        // K or V: write directly to paged KV cache
        const int slot = slot_mapping[row];
        if (slot >= 0) {
            T* cache = is_k ? k_cache : v_cache;
            cache[static_cast<int64_t>(slot) * cache_row_width + src_head * head_dim + tid] = from_float<T>(val);
        }
    }
}

template <typename T>
void launch_fused_split_norm_rope_store(
    const Tensor& qkv,
    const Tensor& positions,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    float norm_eps,
    const Tensor& cos_sin_cache,
    const Tensor& slot_mapping,
    Tensor& k_cache,
    Tensor& v_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    Tensor& q_out,
    cudaStream_t stream)
{
    const int rows = static_cast<int>(qkv.sizes()[0]);
    const int q_size = num_q_heads * head_dim;
    const int kv_size = num_kv_heads * head_dim;
    const int total_cols = q_size + 2 * kv_size;
    const int max_positions = static_cast<int>(cos_sin_cache.sizes()[0]);

    // KV cache row width: flatten all dims after the leading (slot) dim
    // k_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
    // or [num_blocks * block_size, num_kv_heads * head_dim] when flattened by slot
    const int cache_row_width = num_kv_heads * head_dim;

    const int total_heads = num_q_heads + 2 * num_kv_heads;
    dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(total_heads));
    const int block_dim = head_dim;  // e.g. 128
    const size_t smem_bytes = static_cast<size_t>(head_dim) * sizeof(float);

    fused_split_norm_rope_store_kernel<T><<<grid, block_dim, smem_bytes, stream>>>(
        qkv.data_as<T>(),
        positions.data_as<int32_t>(),
        q_norm_weight.data_as<T>(),
        k_norm_weight.data_as<T>(),
        cos_sin_cache.data_as<float>(),
        slot_mapping.data_as<int32_t>(),
        q_out.data_as<T>(),
        k_cache.data_as<T>(),
        v_cache.data_as<T>(),
        rows, num_q_heads, num_kv_heads, head_dim,
        q_size, kv_size, total_cols, norm_eps,
        max_positions, cache_row_width);
    throw_if_cuda_error(cudaPeekAtLastError(), "fused_split_norm_rope_store_kernel launch");
}

} // namespace

void split_qkv(const Tensor& qkv,
               int q_size,
               int kv_size,
               Tensor& q,
               Tensor& k,
               Tensor& v,
               cudaStream_t stream) {
    switch (qkv.dtype()) {
    case ScalarType::kFloat32: launch_split_qkv<float>(qkv, q_size, kv_size, q, k, v, stream); return;
    case ScalarType::kFloat16: launch_split_qkv<__half>(qkv, q_size, kv_size, q, k, v, stream); return;
    case ScalarType::kBFloat16: launch_split_qkv<__nv_bfloat16>(qkv, q_size, kv_size, q, k, v, stream); return;
    default: throw std::invalid_argument("unsupported dtype for split_qkv");
    }
}

void fused_split_norm_rope_store(
    const Tensor& qkv,
    const Tensor& positions,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    float norm_eps,
    const Tensor& cos_sin_cache,
    const Tensor& slot_mapping,
    Tensor& k_cache,
    Tensor& v_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    Tensor& q_out,
    cudaStream_t stream) {
    switch (qkv.dtype()) {
    case ScalarType::kBFloat16:
        launch_fused_split_norm_rope_store<__nv_bfloat16>(
            qkv, positions, q_norm_weight, k_norm_weight, norm_eps,
            cos_sin_cache, slot_mapping, k_cache, v_cache,
            num_q_heads, num_kv_heads, head_dim, q_out, stream);
        return;
    case ScalarType::kFloat16:
        launch_fused_split_norm_rope_store<__half>(
            qkv, positions, q_norm_weight, k_norm_weight, norm_eps,
            cos_sin_cache, slot_mapping, k_cache, v_cache,
            num_q_heads, num_kv_heads, head_dim, q_out, stream);
        return;
    default:
        throw std::invalid_argument("fused_split_norm_rope_store only supports bf16/fp16");
    }
}

} // namespace nano_vllm
