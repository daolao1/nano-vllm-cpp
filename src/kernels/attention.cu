#include "kernels/attention.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

constexpr int kMaxAttentionThreads = 256;

template <typename IndexT>
__device__ __forceinline__ int64_t load_index(const IndexT* data, int64_t index) {
    return static_cast<int64_t>(data[index]);
}

template <typename QuerySeqT, typename KeySeqT>
__device__ void locate_prefill_sequence(const QuerySeqT* cu_seqlens_q,
                                        const KeySeqT* cu_seqlens_k,
                                        int64_t num_seqs,
                                        int64_t q_row,
                                        int64_t query_rows,
                                        int64_t key_rows,
                                        int64_t& q_begin,
                                        int64_t& q_end,
                                        int64_t& k_begin,
                                        int64_t& k_end) {
    if (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr || num_seqs <= 0) {
        q_begin = 0;
        q_end = query_rows;
        k_begin = 0;
        k_end = key_rows;
        return;
    }

    int64_t seq = 0;
    while (seq + 1 < num_seqs && q_row >= load_index(cu_seqlens_q, seq + 1)) {
        ++seq;
    }

    q_begin = load_index(cu_seqlens_q, seq);
    q_end = load_index(cu_seqlens_q, seq + 1);
    k_begin = load_index(cu_seqlens_k, seq);
    k_end = load_index(cu_seqlens_k, seq + 1);
}

__device__ __forceinline__ int kv_head_for_query_head(int query_head, int num_heads, int num_kv_heads) {
    return query_head * num_kv_heads / num_heads;
}

inline int choose_attention_threads(int head_dim) {
    int threads = 32;
    while (threads < head_dim && threads < kMaxAttentionThreads) {
        threads <<= 1;
    }
    return threads > kMaxAttentionThreads ? kMaxAttentionThreads : threads;
}

// Warp-level reduction helpers.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction via shared memory for multi-warp blocks.
__device__ float block_reduce_sum(float val, float* smem, int tid, int nthreads) {
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    const int nwarps = nthreads / 32;
    val = (tid < nwarps) ? smem[tid] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_max(float val, float* smem, int tid, int nthreads) {
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    val = warp_reduce_max(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    const int nwarps = nthreads / 32;
    val = (tid < nwarps) ? smem[tid] : -INFINITY;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

// Prefill dense attention: 1 Q-row per block, online softmax.
// Grid: (query_rows, num_heads), Block: head_dim threads.

template <typename T, typename QuerySeqT, typename KeySeqT>
__global__ void prefill_dense_attention_kernel(const T* query,
                                               const T* key,
                                               const T* value,
                                               const QuerySeqT* cu_seqlens_q,
                                               const KeySeqT* cu_seqlens_k,
                                               int64_t num_seqs,
                                               int64_t query_rows,
                                               int64_t key_rows,
                                               int num_heads,
                                               int num_kv_heads,
                                               int head_dim,
                                               float scale,
                                               T* output) {
    const int64_t q_row = static_cast<int64_t>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    if (q_row >= query_rows || head >= num_heads) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem_raw[];
    float* s_query = reinterpret_cast<float*>(smem_raw);
    float* s_reduce = s_query + head_dim;

    __shared__ int64_t s_k_begin;
    __shared__ int64_t s_allowed_k;
    if (tid == 0) {
        int64_t qb, qe, kb, ke;
        locate_prefill_sequence(cu_seqlens_q, cu_seqlens_k, num_seqs,
                                q_row, query_rows, key_rows,
                                qb, qe, kb, ke);
        const int64_t q_index = q_row - qb;
        s_k_begin = kb;
        s_allowed_k = (ke - kb) - (qe - qb) + q_index + 1;
    }
    __syncthreads();
    const int64_t k_begin = s_k_begin;
    const int64_t allowed_k = s_allowed_k;

    if (allowed_k <= 0) {
        for (int d = tid; d < head_dim; d += nthreads) {
            output[(q_row * num_heads + head) * head_dim + d] = from_float<T>(0.0f);
        }
        return;
    }

    const int kv_head = kv_head_for_query_head(head, num_heads, num_kv_heads);
    const T* q_base = query + (q_row * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        s_query[d] = to_float(q_base[d]);
    }
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc = 0.0f;

    for (int64_t k_index = 0; k_index < allowed_k; ++k_index) {
        const int64_t key_row = k_begin + k_index;
        const T* k_ptr = key + (key_row * num_kv_heads + kv_head) * head_dim;

        float partial_dot = 0.0f;
        for (int d = tid; d < head_dim; d += nthreads) {
            partial_dot += s_query[d] * to_float(k_ptr[d]);
        }
        float qk = block_reduce_sum(partial_dot, s_reduce, tid, nthreads);
        if (tid == 0) s_reduce[0] = qk * scale;
        __syncthreads();
        qk = s_reduce[0];

        const float old_max = running_max;
        running_max = fmaxf(running_max, qk);
        const float exp_correction = expf(old_max - running_max);
        const float exp_qk = expf(qk - running_max);

        v_acc = v_acc * exp_correction;
        running_sum = running_sum * exp_correction + exp_qk;

        if (tid < head_dim) {
            const T* v_ptr = value + (key_row * num_kv_heads + kv_head) * head_dim;
            v_acc += exp_qk * to_float(v_ptr[tid]);
        }
    }

    if (tid < head_dim) {
        const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        output[(q_row * num_heads + head) * head_dim + tid] = from_float<T>(v_acc * inv_sum);
    }
}

template <typename T, typename QuerySeqT, typename KeySeqT, typename BlockIndexT>
__global__ void prefill_paged_attention_kernel(const T* query,
                                               const T* k_cache,
                                               const T* v_cache,
                                               const QuerySeqT* cu_seqlens_q,
                                               const KeySeqT* cu_seqlens_k,
                                               const BlockIndexT* block_tables,
                                               int64_t num_seqs,
                                               int64_t query_rows,
                                               int64_t max_blocks,
                                               int num_heads,
                                               int num_kv_heads,
                                               int head_dim,
                                               int block_size,
                                               float scale,
                                               T* output) {
    const int64_t q_row = static_cast<int64_t>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    if (q_row >= query_rows || head >= num_heads) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem_raw[];
    float* s_query = reinterpret_cast<float*>(smem_raw);
    float* s_reduce = s_query + head_dim;

    __shared__ int64_t s_seq;
    __shared__ int64_t s_allowed_k;
    if (tid == 0) {
        int64_t s = 0;
        while (s + 1 < num_seqs && q_row >= load_index(cu_seqlens_q, s + 1)) {
            ++s;
        }
        int64_t qb = load_index(cu_seqlens_q, s);
        int64_t qe = load_index(cu_seqlens_q, s + 1);
        int64_t kb = load_index(cu_seqlens_k, s);
        int64_t ke = load_index(cu_seqlens_k, s + 1);
        const int64_t q_index = q_row - qb;
        s_seq = s;
        s_allowed_k = (ke - kb) - (qe - qb) + q_index + 1;
    }
    __syncthreads();
    const int64_t seq = s_seq;
    const int64_t allowed_k = s_allowed_k;

    if (allowed_k <= 0) {
        for (int d = tid; d < head_dim; d += nthreads) {
            output[(q_row * num_heads + head) * head_dim + d] = from_float<T>(0.0f);
        }
        return;
    }

    const int kv_head = kv_head_for_query_head(head, num_heads, num_kv_heads);
    const T* q_base = query + (q_row * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        s_query[d] = to_float(q_base[d]);
    }
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc = 0.0f;

    for (int64_t k_index = 0; k_index < allowed_k; ++k_index) {
        const int64_t logical_block = k_index / block_size;
        const int64_t block_offset = k_index % block_size;
        const int64_t block_id = load_index(block_tables, seq * max_blocks + logical_block);
        const int64_t cache_row = block_id * block_size + block_offset;
        const T* k_ptr = k_cache + (cache_row * num_kv_heads + kv_head) * head_dim;

        float partial_dot = 0.0f;
        for (int d = tid; d < head_dim; d += nthreads) {
            partial_dot += s_query[d] * to_float(k_ptr[d]);
        }
        float qk = block_reduce_sum(partial_dot, s_reduce, tid, nthreads);
        if (tid == 0) s_reduce[0] = qk * scale;
        __syncthreads();
        qk = s_reduce[0];

        const float old_max = running_max;
        running_max = fmaxf(running_max, qk);
        const float exp_correction = expf(old_max - running_max);
        const float exp_qk = expf(qk - running_max);

        v_acc = v_acc * exp_correction;
        running_sum = running_sum * exp_correction + exp_qk;

        if (tid < head_dim) {
            const T* v_ptr = v_cache + (cache_row * num_kv_heads + kv_head) * head_dim;
            v_acc += exp_qk * to_float(v_ptr[tid]);
        }
    }

    if (tid < head_dim) {
        const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        output[(q_row * num_heads + head) * head_dim + tid] = from_float<T>(v_acc * inv_sum);
    }
}

// Warp-per-head decode attention: each warp handles one (seq, head) pair.
// With head_dim=128, each thread processes 4 elements. Uses warp_reduce_sum
// (no __syncthreads needed) for massive reduction in sync overhead.
// For small head_dim (<32), falls back to fewer active lanes.
// Grid: (ceil(batch_size * num_heads / WARPS_PER_BLOCK), 1)
// Block: WARPS_PER_BLOCK * 32 threads
constexpr int kDecodeWarpsPerBlock = 4;

template <typename T, typename LengthT, typename BlockIndexT>
__global__ void decode_paged_attention_kernel(const T* query,
                                              const T* k_cache,
                                              const T* v_cache,
                                              const LengthT* context_lens,
                                              const BlockIndexT* block_tables,
                                              int64_t batch_size,
                                              int64_t max_blocks,
                                              int num_heads,
                                              int num_kv_heads,
                                              int head_dim,
                                              int block_size,
                                              float scale,
                                              T* output) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int global_warp = blockIdx.x * kDecodeWarpsPerBlock + warp_id;
    const int total_work = static_cast<int>(batch_size) * num_heads;
    if (global_warp >= total_work) return;

    const int seq_idx = global_warp / num_heads;
    const int head = global_warp % num_heads;
    const int kv_head = kv_head_for_query_head(head, num_heads, num_kv_heads);

    // Load query into registers. Each lane handles elements lane, lane+32, lane+64, ...
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const T* q_base = query + (static_cast<int64_t>(seq_idx) * num_heads + head) * head_dim;
    const int elems_per_thread = (head_dim + 31) / 32;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (i < elems_per_thread && d < head_dim) {
            q_reg[i] = to_float(q_base[d]);
        }
    }

    const int64_t k_len = load_index(context_lens, seq_idx);
    if (k_len <= 0) {
        T* out_base = output + (static_cast<int64_t>(seq_idx) * num_heads + head) * head_dim;
        for (int d = lane; d < head_dim; d += 32) {
            out_base[d] = from_float<T>(0.0f);
        }
        return;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const BlockIndexT* seq_block_table = block_tables + static_cast<int64_t>(seq_idx) * max_blocks;

    for (int64_t k_index = 0; k_index < k_len; ++k_index) {
        const int64_t logical_block = k_index / block_size;
        const int64_t block_offset = k_index % block_size;
        const int64_t block_id = load_index(seq_block_table, logical_block);
        const int64_t cache_row = block_id * block_size + block_offset;
        const T* k_ptr = k_cache + (cache_row * num_kv_heads + kv_head) * head_dim;

        // Dot product: each thread handles up to 4 elements.
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (i < elems_per_thread && d < head_dim) {
                partial += q_reg[i] * to_float(k_ptr[d]);
            }
        }
        float qk = warp_reduce_sum(partial) * scale;

        // Online softmax (no sync needed — warp-uniform via shuffle).
        const float old_max = running_max;
        running_max = fmaxf(running_max, qk);
        const float correction = expf(old_max - running_max);
        const float exp_qk = expf(qk - running_max);
        running_sum = running_sum * correction + exp_qk;

        // Value accumulation.
        const T* v_ptr = v_cache + (cache_row * num_kv_heads + kv_head) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (i < elems_per_thread && d < head_dim) {
                v_acc[i] = v_acc[i] * correction + exp_qk * to_float(v_ptr[d]);
            }
        }
    }

    const float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    T* out_base = output + (static_cast<int64_t>(seq_idx) * num_heads + head) * head_dim;
    for (int d = lane; d < head_dim; d += 32) {
        const int i = d / 32;
        out_base[d] = from_float<T>(v_acc[i] * inv);
    }
}

template <typename T, typename QuerySeqT, typename KeySeqT>
void launch_prefill_dense_impl(const Tensor& query,
                               const Tensor* cu_seqlens_q,
                               const Tensor& key,
                               const Tensor& value,
                               const Tensor* cu_seqlens_k,
                               int num_kv_heads,
                               int head_dim,
                               float scale,
                               Tensor& output,
                               cudaStream_t stream) {
    const int64_t query_rows = query.sizes()[0];
    const int64_t key_rows = key.sizes()[0];
    const int64_t num_seqs = cu_seqlens_q != nullptr ? static_cast<int64_t>(cu_seqlens_q->numel() - 1) : 1;
    const QuerySeqT* q_ptr = cu_seqlens_q != nullptr ? cu_seqlens_q->data_as<QuerySeqT>() : nullptr;
    const KeySeqT* k_ptr = cu_seqlens_k != nullptr ? cu_seqlens_k->data_as<KeySeqT>() : nullptr;

    const dim3 grid(static_cast<unsigned int>(query_rows), static_cast<unsigned int>(query.sizes()[1]));
    int threads = ((head_dim + 31) / 32) * 32;
    if (threads > kMaxAttentionThreads) threads = kMaxAttentionThreads;
    // Shared memory: s_query[hd] + s_reduce[nwarps]
    const size_t smem_bytes = (static_cast<size_t>(head_dim) + static_cast<size_t>(threads / 32)) * sizeof(float);
    prefill_dense_attention_kernel<T, QuerySeqT, KeySeqT><<<grid, threads, smem_bytes, stream>>>(query.data_as<T>(),
                                                                                      key.data_as<T>(),
                                                                                      value.data_as<T>(),
                                                                                      q_ptr,
                                                                                      k_ptr,
                                                                                      num_seqs,
                                                                                      query_rows,
                                                                                      key_rows,
                                                                                      static_cast<int>(query.sizes()[1]),
                                                                                      num_kv_heads,
                                                                                      head_dim,
                                                                                      scale,
                                                                                      output.data_as<T>());
    throw_if_cuda_error(cudaPeekAtLastError(), "prefill_dense_attention_kernel launch");
}

template <typename T, typename QuerySeqT, typename KeySeqT, typename BlockIndexT>
void launch_prefill_paged_impl(const Tensor& query,
                               const Tensor* cu_seqlens_q,
                               const Tensor& k_cache,
                               const Tensor& v_cache,
                               const Tensor* cu_seqlens_k,
                               const Tensor& block_tables,
                               int num_kv_heads,
                               int head_dim,
                               int block_size,
                               float scale,
                               Tensor& output,
                               cudaStream_t stream) {
    const int64_t query_rows = query.sizes()[0];
    const int64_t num_seqs = static_cast<int64_t>(cu_seqlens_q->numel() - 1);
    const int64_t max_blocks = block_tables.sizes()[1];

    const dim3 grid(static_cast<unsigned int>(query_rows), static_cast<unsigned int>(query.sizes()[1]));
    int threads = ((head_dim + 31) / 32) * 32;
    if (threads > kMaxAttentionThreads) threads = kMaxAttentionThreads;
    const size_t smem_bytes = (static_cast<size_t>(head_dim) + static_cast<size_t>(threads / 32)) * sizeof(float);
    prefill_paged_attention_kernel<T, QuerySeqT, KeySeqT, BlockIndexT><<<grid, threads, smem_bytes, stream>>>(
        query.data_as<T>(),
        k_cache.data_as<T>(),
        v_cache.data_as<T>(),
        cu_seqlens_q->data_as<QuerySeqT>(),
        cu_seqlens_k->data_as<KeySeqT>(),
        block_tables.data_as<BlockIndexT>(),
        num_seqs,
        query_rows,
        max_blocks,
        static_cast<int>(query.sizes()[1]),
        num_kv_heads,
        head_dim,
        block_size,
        scale,
        output.data_as<T>());
    throw_if_cuda_error(cudaPeekAtLastError(), "prefill_paged_attention_kernel launch");
}

template <typename T, typename LengthT, typename BlockIndexT>
void launch_decode_paged_impl(const Tensor& query,
                              const Tensor& k_cache,
                              const Tensor& v_cache,
                              const Tensor& context_lens,
                              const Tensor& block_tables,
                              int num_kv_heads,
                              int head_dim,
                              int block_size,
                              float scale,
                              Tensor& output,
                              cudaStream_t stream) {
    const int64_t batch_size = query.sizes()[0];
    const int num_heads = static_cast<int>(query.sizes()[1]);
    const int64_t max_blocks = block_tables.sizes()[1];
    const int total_work = static_cast<int>(batch_size) * num_heads;
    const int num_blocks = (total_work + kDecodeWarpsPerBlock - 1) / kDecodeWarpsPerBlock;
    const int threads = kDecodeWarpsPerBlock * 32;
    decode_paged_attention_kernel<T, LengthT, BlockIndexT><<<num_blocks, threads, 0, stream>>>(query.data_as<T>(),
                                                                                       k_cache.data_as<T>(),
                                                                                       v_cache.data_as<T>(),
                                                                                       context_lens.data_as<LengthT>(),
                                                                                       block_tables.data_as<BlockIndexT>(),
                                                                                       batch_size,
                                                                                       max_blocks,
                                                                                       num_heads,
                                                                                       num_kv_heads,
                                                                                       head_dim,
                                                                                       block_size,
                                                                                       scale,
                                                                                       output.data_as<T>());
    throw_if_cuda_error(cudaPeekAtLastError(), "decode_paged_attention_kernel launch");
}

template <typename T, typename QuerySeqT>
void dispatch_prefill_dense_k(const Tensor& query,
                              const Tensor* cu_seqlens_q,
                              const Tensor& key,
                              const Tensor& value,
                              const Tensor* cu_seqlens_k,
                              int num_kv_heads,
                              int head_dim,
                              float scale,
                              Tensor& output,
                              cudaStream_t stream) {
    if (cu_seqlens_k == nullptr || cu_seqlens_k->dtype() == ScalarType::kInt32) {
        launch_prefill_dense_impl<T, QuerySeqT, int32_t>(query,
                                                      cu_seqlens_q,
                                                      key,
                                                      value,
                                                      cu_seqlens_k,
                                                      num_kv_heads,
                                                      head_dim,
                                                      scale,
                                                      output,
                                                      stream);
        return;
    }
    launch_prefill_dense_impl<T, QuerySeqT, int64_t>(query,
                                                  cu_seqlens_q,
                                                  key,
                                                  value,
                                                  cu_seqlens_k,
                                                  num_kv_heads,
                                                  head_dim,
                                                  scale,
                                                  output,
                                                  stream);
}

template <typename T, typename QuerySeqT, typename KeySeqT>
void dispatch_prefill_paged_block(const Tensor& query,
                                  const Tensor* cu_seqlens_q,
                                  const Tensor& k_cache,
                                  const Tensor& v_cache,
                                  const Tensor* cu_seqlens_k,
                                  const Tensor& block_tables,
                                  int num_kv_heads,
                                  int head_dim,
                                  int block_size,
                                  float scale,
                                  Tensor& output,
                                  cudaStream_t stream) {
    if (block_tables.dtype() == ScalarType::kInt32) {
        launch_prefill_paged_impl<T, QuerySeqT, KeySeqT, int32_t>(query,
                                                               cu_seqlens_q,
                                                               k_cache,
                                                               v_cache,
                                                               cu_seqlens_k,
                                                               block_tables,
                                                               num_kv_heads,
                                                               head_dim,
                                                               block_size,
                                                               scale,
                                                               output,
                                                               stream);
        return;
    }
    launch_prefill_paged_impl<T, QuerySeqT, KeySeqT, int64_t>(query,
                                                           cu_seqlens_q,
                                                           k_cache,
                                                           v_cache,
                                                           cu_seqlens_k,
                                                           block_tables,
                                                           num_kv_heads,
                                                           head_dim,
                                                           block_size,
                                                           scale,
                                                           output,
                                                           stream);
}

template <typename T, typename LengthT>
void dispatch_decode_paged_block(const Tensor& query,
                                 const Tensor& k_cache,
                                 const Tensor& v_cache,
                                 const Tensor& context_lens,
                                 const Tensor& block_tables,
                                 int num_kv_heads,
                                 int head_dim,
                                 int block_size,
                                 float scale,
                                 Tensor& output,
                                 cudaStream_t stream) {
    if (block_tables.dtype() == ScalarType::kInt32) {
        launch_decode_paged_impl<T, LengthT, int32_t>(query,
                                                   k_cache,
                                                   v_cache,
                                                   context_lens,
                                                   block_tables,
                                                   num_kv_heads,
                                                   head_dim,
                                                   block_size,
                                                   scale,
                                                   output,
                                                   stream);
        return;
    }
    launch_decode_paged_impl<T, LengthT, int64_t>(query,
                                               k_cache,
                                               v_cache,
                                               context_lens,
                                               block_tables,
                                               num_kv_heads,
                                               head_dim,
                                               block_size,
                                               scale,
                                               output,
                                               stream);
}

} // namespace

void launch_prefill_attention_dense(const Tensor& query,
                                    const Tensor* cu_seqlens_q,
                                    const Tensor& key,
                                    const Tensor& value,
                                    const Tensor* cu_seqlens_k,
                                    int num_kv_heads,
                                    int head_dim,
                                    float scale,
                                    Tensor& output,
                                    cudaStream_t stream) {
    const bool q32 = cu_seqlens_q == nullptr || cu_seqlens_q->dtype() == ScalarType::kInt32;
    switch (query.dtype()) {
    case ScalarType::kFloat32:
        if (q32) dispatch_prefill_dense_k<float, int32_t>(query, cu_seqlens_q, key, value, cu_seqlens_k, num_kv_heads, head_dim, scale, output, stream);
        else dispatch_prefill_dense_k<float, int64_t>(query, cu_seqlens_q, key, value, cu_seqlens_k, num_kv_heads, head_dim, scale, output, stream);
        return;
    case ScalarType::kFloat16:
        if (q32) dispatch_prefill_dense_k<__half, int32_t>(query, cu_seqlens_q, key, value, cu_seqlens_k, num_kv_heads, head_dim, scale, output, stream);
        else dispatch_prefill_dense_k<__half, int64_t>(query, cu_seqlens_q, key, value, cu_seqlens_k, num_kv_heads, head_dim, scale, output, stream);
        return;
    case ScalarType::kBFloat16:
        if (q32) dispatch_prefill_dense_k<__nv_bfloat16, int32_t>(query, cu_seqlens_q, key, value, cu_seqlens_k, num_kv_heads, head_dim, scale, output, stream);
        else dispatch_prefill_dense_k<__nv_bfloat16, int64_t>(query, cu_seqlens_q, key, value, cu_seqlens_k, num_kv_heads, head_dim, scale, output, stream);
        return;
    default:
        throw std::invalid_argument("unsupported dtype for prefill_attention_dense");
    }
}

void launch_prefill_attention_paged(const Tensor& query,
                                    const Tensor* cu_seqlens_q,
                                    const Tensor& k_cache,
                                    const Tensor& v_cache,
                                    const Tensor* cu_seqlens_k,
                                    const Tensor& block_tables,
                                    int num_kv_heads,
                                    int head_dim,
                                    int block_size,
                                    float scale,
                                    Tensor& output,
                                    cudaStream_t stream) {
    if (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr) {
        throw std::invalid_argument("paged prefill attention requires cu_seqlens_q and cu_seqlens_k");
    }
    const bool q32 = cu_seqlens_q->dtype() == ScalarType::kInt32;
    const bool k32 = cu_seqlens_k->dtype() == ScalarType::kInt32;
    switch (query.dtype()) {
    case ScalarType::kFloat32:
        if (q32 && k32) dispatch_prefill_paged_block<float, int32_t, int32_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else if (q32) dispatch_prefill_paged_block<float, int32_t, int64_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else if (k32) dispatch_prefill_paged_block<float, int64_t, int32_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_prefill_paged_block<float, int64_t, int64_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    case ScalarType::kFloat16:
        if (q32 && k32) dispatch_prefill_paged_block<__half, int32_t, int32_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else if (q32) dispatch_prefill_paged_block<__half, int32_t, int64_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else if (k32) dispatch_prefill_paged_block<__half, int64_t, int32_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_prefill_paged_block<__half, int64_t, int64_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    case ScalarType::kBFloat16:
        if (q32 && k32) dispatch_prefill_paged_block<__nv_bfloat16, int32_t, int32_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else if (q32) dispatch_prefill_paged_block<__nv_bfloat16, int32_t, int64_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else if (k32) dispatch_prefill_paged_block<__nv_bfloat16, int64_t, int32_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_prefill_paged_block<__nv_bfloat16, int64_t, int64_t>(query, cu_seqlens_q, k_cache, v_cache, cu_seqlens_k, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    default:
        throw std::invalid_argument("unsupported dtype for prefill_attention_paged");
    }
}

void launch_decode_attention_paged(const Tensor& query,
                                   const Tensor& k_cache,
                                   const Tensor& v_cache,
                                   const Tensor& context_lens,
                                   const Tensor& block_tables,
                                   int num_kv_heads,
                                   int head_dim,
                                   int block_size,
                                   float scale,
                                   Tensor& output,
                                   cudaStream_t stream) {
    const bool len32 = context_lens.dtype() == ScalarType::kInt32;
    switch (query.dtype()) {
    case ScalarType::kFloat32:
        if (len32) dispatch_decode_paged_block<float, int32_t>(query, k_cache, v_cache, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_decode_paged_block<float, int64_t>(query, k_cache, v_cache, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    case ScalarType::kFloat16:
        if (len32) dispatch_decode_paged_block<__half, int32_t>(query, k_cache, v_cache, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_decode_paged_block<__half, int64_t>(query, k_cache, v_cache, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    case ScalarType::kBFloat16:
        if (len32) dispatch_decode_paged_block<__nv_bfloat16, int32_t>(query, k_cache, v_cache, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_decode_paged_block<__nv_bfloat16, int64_t>(query, k_cache, v_cache, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    default:
        throw std::invalid_argument("unsupported dtype for decode_attention_paged");
    }
}

// ---------------------------------------------------------------------------
// Int8 KV cache decode attention kernel
// K/V are int8, scales are fp16 (one per token per head).
// Query and output remain in compute dtype (fp16/bf16/fp32).
// ---------------------------------------------------------------------------
namespace {

template <typename QueryT, typename LengthT, typename BlockIndexT>
__global__ void decode_paged_attention_int8_kernel(const QueryT* query,
                                                    const int8_t* k_cache,
                                                    const int8_t* v_cache,
                                                    const __half* k_scale,
                                                    const __half* v_scale,
                                                    const LengthT* context_lens,
                                                    const BlockIndexT* block_tables,
                                                    int64_t batch_size,
                                                    int64_t max_blocks,
                                                    int num_heads,
                                                    int num_kv_heads,
                                                    int head_dim,
                                                    int block_size,
                                                    float attn_scale,
                                                    QueryT* output) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int global_warp = blockIdx.x * kDecodeWarpsPerBlock + warp_id;
    const int total_work = static_cast<int>(batch_size) * num_heads;
    if (global_warp >= total_work) return;

    const int seq_idx = global_warp / num_heads;
    const int head = global_warp % num_heads;
    const int kv_head = kv_head_for_query_head(head, num_heads, num_kv_heads);

    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const QueryT* q_base = query + (static_cast<int64_t>(seq_idx) * num_heads + head) * head_dim;
    const int elems_per_thread = (head_dim + 31) / 32;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (i < elems_per_thread && d < head_dim) {
            q_reg[i] = to_float(q_base[d]);
        }
    }

    const int64_t k_len = load_index(context_lens, seq_idx);
    if (k_len <= 0) {
        QueryT* out_base = output + (static_cast<int64_t>(seq_idx) * num_heads + head) * head_dim;
        for (int d = lane; d < head_dim; d += 32) {
            out_base[d] = from_float<QueryT>(0.0f);
        }
        return;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const BlockIndexT* seq_block_table = block_tables + static_cast<int64_t>(seq_idx) * max_blocks;

    for (int64_t k_index = 0; k_index < k_len; ++k_index) {
        const int64_t logical_block = k_index / block_size;
        const int64_t block_offset = k_index % block_size;
        const int64_t block_id = load_index(seq_block_table, logical_block);
        const int64_t cache_row = block_id * block_size + block_offset;

        // Read K scale for this token, this kv_head
        const float ks = __half2float(k_scale[cache_row * num_kv_heads + kv_head]);
        const int8_t* k_ptr = k_cache + (cache_row * num_kv_heads + kv_head) * head_dim;

        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (i < elems_per_thread && d < head_dim) {
                float k_val = static_cast<float>(k_ptr[d]) * ks;
                partial += q_reg[i] * k_val;
            }
        }
        float qk = warp_reduce_sum(partial) * attn_scale;

        const float old_max = running_max;
        running_max = fmaxf(running_max, qk);
        const float correction = expf(old_max - running_max);
        const float exp_qk = expf(qk - running_max);
        running_sum = running_sum * correction + exp_qk;

        // Read V scale
        const float vs = __half2float(v_scale[cache_row * num_kv_heads + kv_head]);
        const int8_t* v_ptr = v_cache + (cache_row * num_kv_heads + kv_head) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (i < elems_per_thread && d < head_dim) {
                float v_val = static_cast<float>(v_ptr[d]) * vs;
                v_acc[i] = v_acc[i] * correction + exp_qk * v_val;
            }
        }
    }

    const float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    QueryT* out_base = output + (static_cast<int64_t>(seq_idx) * num_heads + head) * head_dim;
    for (int d = lane; d < head_dim; d += 32) {
        const int i = d / 32;
        out_base[d] = from_float<QueryT>(v_acc[i] * inv);
    }
}

template <typename QueryT, typename LengthT>
void dispatch_decode_paged_int8_block(const Tensor& query,
                                       const Tensor& k_cache,
                                       const Tensor& v_cache,
                                       const Tensor& k_scale,
                                       const Tensor& v_scale,
                                       const Tensor& context_lens,
                                       const Tensor& block_tables,
                                       int num_kv_heads,
                                       int head_dim,
                                       int block_size,
                                       float scale,
                                       Tensor& output,
                                       cudaStream_t stream) {
    const int64_t batch_size = query.sizes()[0];
    const int num_heads = static_cast<int>(query.sizes()[1]);
    const int64_t max_blocks = block_tables.sizes()[1];
    const int total_warps = static_cast<int>(batch_size) * num_heads;
    const int blocks_needed = (total_warps + kDecodeWarpsPerBlock - 1) / kDecodeWarpsPerBlock;

    if (block_tables.dtype() == ScalarType::kInt32) {
        decode_paged_attention_int8_kernel<QueryT, LengthT, int32_t>
            <<<blocks_needed, kDecodeWarpsPerBlock * 32, 0, stream>>>(
                query.data_as<QueryT>(),
                k_cache.data_as<int8_t>(),
                v_cache.data_as<int8_t>(),
                k_scale.data_as<__half>(),
                v_scale.data_as<__half>(),
                context_lens.data_as<LengthT>(),
                block_tables.data_as<int32_t>(),
                batch_size, max_blocks, num_heads, num_kv_heads,
                head_dim, block_size, scale,
                output.data_as<QueryT>());
    } else {
        decode_paged_attention_int8_kernel<QueryT, LengthT, int64_t>
            <<<blocks_needed, kDecodeWarpsPerBlock * 32, 0, stream>>>(
                query.data_as<QueryT>(),
                k_cache.data_as<int8_t>(),
                v_cache.data_as<int8_t>(),
                k_scale.data_as<__half>(),
                v_scale.data_as<__half>(),
                context_lens.data_as<LengthT>(),
                block_tables.data_as<int64_t>(),
                batch_size, max_blocks, num_heads, num_kv_heads,
                head_dim, block_size, scale,
                output.data_as<QueryT>());
    }
    throw_if_cuda_error(cudaPeekAtLastError(), "decode_paged_attention_int8_kernel launch");
}

} // namespace

void launch_decode_attention_paged_int8(const Tensor& query,
                                        const Tensor& k_cache,
                                        const Tensor& v_cache,
                                        const Tensor& k_scale,
                                        const Tensor& v_scale,
                                        const Tensor& context_lens,
                                        const Tensor& block_tables,
                                        int num_kv_heads,
                                        int head_dim,
                                        int block_size,
                                        float scale,
                                        Tensor& output,
                                        cudaStream_t stream) {
    const bool len32 = context_lens.dtype() == ScalarType::kInt32;
    switch (query.dtype()) {
    case ScalarType::kFloat32:
        if (len32) dispatch_decode_paged_int8_block<float, int32_t>(query, k_cache, v_cache, k_scale, v_scale, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_decode_paged_int8_block<float, int64_t>(query, k_cache, v_cache, k_scale, v_scale, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    case ScalarType::kFloat16:
        if (len32) dispatch_decode_paged_int8_block<__half, int32_t>(query, k_cache, v_cache, k_scale, v_scale, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_decode_paged_int8_block<__half, int64_t>(query, k_cache, v_cache, k_scale, v_scale, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    case ScalarType::kBFloat16:
        if (len32) dispatch_decode_paged_int8_block<__nv_bfloat16, int32_t>(query, k_cache, v_cache, k_scale, v_scale, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        else dispatch_decode_paged_int8_block<__nv_bfloat16, int64_t>(query, k_cache, v_cache, k_scale, v_scale, context_lens, block_tables, num_kv_heads, head_dim, block_size, scale, output, stream);
        return;
    default:
        throw std::invalid_argument("unsupported query dtype for decode_attention_paged_int8");
    }
}

} // namespace nano_vllm