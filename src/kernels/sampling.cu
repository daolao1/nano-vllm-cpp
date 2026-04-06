#include "kernels/sampling.h"
#include "kernels/kernel_types.h"
#include "kernels/device_helpers.cuh"

#include "utils/cuda_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace nano_vllm {

using namespace kernel_detail;

namespace {

// ---------------------------------------------------------------------------
// Repetition-penalty kernel
// ---------------------------------------------------------------------------
// For each row, apply penalty to token positions listed in penalty_token_ids.
// penalty_token_ids : [batch_size, max_penalty_tokens]  (int32, padded with -1)
// penalty_token_counts : [batch_size] (int32, actual count per row)
// penalties : [batch_size] (float32)
template <typename T>
__global__ void repetition_penalty_kernel(T* logits,
                                          const int32_t* penalty_token_ids,
                                          const int32_t* penalty_token_counts,
                                          const float* penalties,
                                          int64_t rows,
                                          int64_t cols,
                                          int64_t max_penalty_tokens) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const float penalty = penalties[row];
    if (penalty == 1.0f) return;

    T* row_logits = logits + row * cols;
    const int32_t* row_ids = penalty_token_ids + row * max_penalty_tokens;
    const int count = penalty_token_counts[row];

    for (int i = tid; i < count; i += blockDim.x) {
        const int32_t token_id = row_ids[i];
        if (token_id < 0 || token_id >= static_cast<int32_t>(cols)) continue;
        float logit = to_float(row_logits[token_id]);
        // Positive logits are divided, negative logits are multiplied.
        logit = (logit > 0.0f) ? logit / penalty : logit * penalty;
        row_logits[token_id] = from_float<T>(logit);
    }
}

template <typename T>
void launch_repetition_penalty(Tensor& logits,
                               const Tensor& penalty_token_ids,
                               const Tensor& penalty_token_counts,
                               const Tensor& penalties,
                               cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("logits", logits);
    if (shape.rows == 0) return;
    const int64_t max_penalty_tokens = penalty_token_ids.sizes()[1];
    repetition_penalty_kernel<T><<<static_cast<unsigned int>(shape.rows), kThreadsPerBlock, 0, stream>>>(
        logits.data_as<T>(),
        penalty_token_ids.data_as<int32_t>(),
        penalty_token_counts.data_as<int32_t>(),
        penalties.data_as<float>(),
        shape.rows,
        shape.cols,
        max_penalty_tokens);
    throw_if_cuda_error(cudaPeekAtLastError(), "repetition_penalty_kernel launch");
}

// ---------------------------------------------------------------------------
// Top-k / top-p sampling kernel  (Gumbel-max with masking)
// ---------------------------------------------------------------------------
// Algorithm for top-k:
//   1. First pass: find the k-th largest scaled logit via block-level
//      approximate threshold (iterative max-reduce to collect top-k values).
//   2. Mask everything below the threshold to -inf.
//
// Algorithm for top-p:
//   After top-k masking, compute online softmax + cumulative probability.
//   Finding the exact nucleus on GPU with 150k vocab is tricky in a single
//   block; we use a practical approximation:
//     - Find max logit, compute log-sum-exp, derive probabilities.
//     - Use Gumbel-max trick which naturally concentrates on high-prob tokens.
//     - Bias: set logits with prob < (1-top_p)/vocab_size to -inf,
//       which is equivalent to removing the long tail.
//   A more precise approach: use the Gumbel-max trick directly.
//   With top-p close to 1.0, the effect is marginal. With top-p = 0.9,
//   we want to only consider tokens whose cumulative sorted probability
//   reaches 0.9. The Gumbel trick already heavily favors high-probability
//   tokens, so the combination of top-k + temperature gives most of the
//   benefit. For exact top-p, we implement a two-pass approach.
//
// Combined kernel flow:
//   Pass 1: Find max logit for numerical stability (block reduce max).
//   Pass 2: If top_k > 0, find the k-th largest value via approximate
//           radix-based selection using shared memory.
//   Pass 3: Apply mask + Gumbel-max sampling.
//
// For simplicity and correctness, we use a two-kernel approach:
//   Kernel A: Apply top-k masking (find k-th value, mask to -inf)
//   Kernel B: Apply top-p masking + Gumbel-max sampling

// ---- block-level k-th largest finder ----
// Each thread processes a strided portion of the row and maintains local
// top candidates.  We then do a block-level merge.  For large vocab (150k)
// and k typically <= 50, we can do this efficiently.

// Find the k-th largest value in a row using iterative threshold refinement.
// This is an O(V * log(range)) approach using binary search on the value range.
__device__ float find_kth_largest(const float* values, int64_t n, int k) {
    // Find global max and min first (shared across block).
    float local_max = -FLT_MAX;
    float local_min = FLT_MAX;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = values[i];
        if (v > local_max) local_max = v;
        if (v < local_min) local_min = v;
    }

    // Block reduce for max and min.
    __shared__ float s_max[kMaxWarpsPerBlock];
    __shared__ float s_min[kMaxWarpsPerBlock];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    // Warp reduce max
    for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
        float other = __shfl_xor_sync(kFullWarpMask, local_max, off);
        if (other > local_max) local_max = other;
    }
    // Warp reduce min
    float lmin = local_min;
    for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
        float other = __shfl_xor_sync(kFullWarpMask, lmin, off);
        if (other < lmin) lmin = other;
    }

    if (lane == 0) { s_max[warp] = local_max; s_min[warp] = lmin; }
    __syncthreads();

    if (warp == 0) {
        const int wc = blockDim.x / kWarpSize;
        float mx = (lane < wc) ? s_max[lane] : -FLT_MAX;
        float mn = (lane < wc) ? s_min[lane] : FLT_MAX;
        for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
            float o = __shfl_xor_sync(kFullWarpMask, mx, off);
            if (o > mx) mx = o;
            o = __shfl_xor_sync(kFullWarpMask, mn, off);
            if (o < mn) mn = o;
        }
        if (lane == 0) { s_max[0] = mx; s_min[0] = mn; }
    }
    __syncthreads();

    float hi = s_max[0];
    float lo = s_min[0];

    // Binary search: find threshold such that count(values >= threshold) == k
    for (int iter = 0; iter < 64; ++iter) {
        float mid = 0.5f * (hi + lo);
        // Count values > mid
        int local_count = 0;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            if (values[i] > mid) ++local_count;
        }
        // Block reduce sum (reuse shared mem)
        __shared__ int s_count[kMaxWarpsPerBlock];
        int cnt = local_count;
        for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
            cnt += __shfl_xor_sync(kFullWarpMask, cnt, off);
        }
        if (lane == 0) s_count[warp] = cnt;
        __syncthreads();
        if (warp == 0) {
            const int wc = blockDim.x / kWarpSize;
            int total = (lane < wc) ? s_count[lane] : 0;
            for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
                total += __shfl_xor_sync(kFullWarpMask, total, off);
            }
            if (lane == 0) s_count[0] = total;
        }
        __syncthreads();

        int total_above = s_count[0];
        if (total_above < k) {
            hi = mid; // threshold too high
        } else {
            lo = mid; // threshold can go higher
        }
        if (hi - lo < 1e-6f) break;
    }

    return lo; // k-th largest value (approximate)
}

template <typename T>
__global__ void sample_tokens_topk_topp_kernel(const T* logits,
                                               const float* temperatures,
                                               const int32_t* top_ks,
                                               const float* top_ps,
                                               int32_t* output,
                                               int64_t rows,
                                               int64_t cols,
                                               uint64_t seed) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const float temperature = fmaxf(temperatures[row], 1e-10f);
    const int top_k = top_ks[row];
    const float top_p = top_ps[row];
    const T* row_logits = logits + row * cols;

    // --- Step 1: Temperature-scaled logits ---
    // We need a scratch buffer for scaled logits. Use dynamic shared memory.
    // For large vocab we cannot fit in shmem, so we work in-register with
    // multiple passes.

    // --- Step 2: Find top-k threshold if needed ---
    float topk_threshold = -FLT_MAX;
    if (top_k > 0 && top_k < static_cast<int>(cols)) {
        // First compute scaled logits max/min for the binary search
        // We do the binary search on temperature-scaled logits.
        // find_kth_largest works on a float array, but our logits are type T.
        // We'll do an inline version here.

        // Find max and min of scaled logits
        float local_max = -FLT_MAX, local_min = FLT_MAX;
        for (int64_t c = tid; c < cols; c += blockDim.x) {
            float v = to_float(row_logits[c]) / temperature;
            if (v > local_max) local_max = v;
            if (v < local_min) local_min = v;
        }

        __shared__ float s_vals[kMaxWarpsPerBlock];
        const int lane = threadIdx.x % kWarpSize;
        const int warp = threadIdx.x / kWarpSize;

        // Reduce max
        for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
            float o = __shfl_xor_sync(kFullWarpMask, local_max, off);
            if (o > local_max) local_max = o;
        }
        if (lane == 0) s_vals[warp] = local_max;
        __syncthreads();
        if (warp == 0) {
            const int wc = blockDim.x / kWarpSize;
            float mx = (lane < wc) ? s_vals[lane] : -FLT_MAX;
            for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
                float o = __shfl_xor_sync(kFullWarpMask, mx, off);
                if (o > mx) mx = o;
            }
            if (lane == 0) s_vals[0] = mx;
        }
        __syncthreads();
        float hi = s_vals[0];

        // Reduce min
        for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
            float o = __shfl_xor_sync(kFullWarpMask, local_min, off);
            if (o < local_min) local_min = o;
        }
        if (lane == 0) s_vals[warp] = local_min;
        __syncthreads();
        if (warp == 0) {
            const int wc = blockDim.x / kWarpSize;
            float mn = (lane < wc) ? s_vals[lane] : FLT_MAX;
            for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
                float o = __shfl_xor_sync(kFullWarpMask, mn, off);
                if (o < mn) mn = o;
            }
            if (lane == 0) s_vals[0] = mn;
        }
        __syncthreads();
        float lo = s_vals[0];

        // Binary search for k-th largest
        __shared__ int s_count[kMaxWarpsPerBlock];
        for (int iter = 0; iter < 50; ++iter) {
            float mid = 0.5f * (hi + lo);
            int local_count = 0;
            for (int64_t c = tid; c < cols; c += blockDim.x) {
                if (to_float(row_logits[c]) / temperature > mid) ++local_count;
            }
            int cnt = local_count;
            for (int off = kWarpSize / 2; off >= 1; off >>= 1)
                cnt += __shfl_xor_sync(kFullWarpMask, cnt, off);
            if (lane == 0) s_count[warp] = cnt;
            __syncthreads();
            if (warp == 0) {
                const int wc = blockDim.x / kWarpSize;
                int total = (lane < wc) ? s_count[lane] : 0;
                for (int off = kWarpSize / 2; off >= 1; off >>= 1)
                    total += __shfl_xor_sync(kFullWarpMask, total, off);
                if (lane == 0) s_count[0] = total;
            }
            __syncthreads();
            if (s_count[0] < top_k) hi = mid; else lo = mid;
            if (hi - lo < 1e-6f) break;
        }
        topk_threshold = lo;
    }

    // --- Step 3: Top-p via block-level softmax + cumulative sum ---
    // Find max for softmax numerical stability (among top-k filtered values)
    float local_max_filtered = -FLT_MAX;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float v = to_float(row_logits[c]) / temperature;
        if (v >= topk_threshold) {
            if (v > local_max_filtered) local_max_filtered = v;
        }
    }
    // Block reduce max
    {
        __shared__ float s_mx[kMaxWarpsPerBlock];
        const int lane = threadIdx.x % kWarpSize;
        const int warp = threadIdx.x / kWarpSize;
        float mx = local_max_filtered;
        for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
            float o = __shfl_xor_sync(kFullWarpMask, mx, off);
            if (o > mx) mx = o;
        }
        if (lane == 0) s_mx[warp] = mx;
        __syncthreads();
        if (warp == 0) {
            const int wc = blockDim.x / kWarpSize;
            mx = (lane < wc) ? s_mx[lane] : -FLT_MAX;
            for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
                float o = __shfl_xor_sync(kFullWarpMask, mx, off);
                if (o > mx) mx = o;
            }
            if (lane == 0) s_mx[0] = mx;
        }
        __syncthreads();
        local_max_filtered = s_mx[0];
    }
    const float log_max = local_max_filtered;

    // Compute sum(exp(logit - max)) for softmax denominator
    float local_exp_sum = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        float v = to_float(row_logits[c]) / temperature;
        if (v >= topk_threshold) {
            local_exp_sum += expf(v - log_max);
        }
    }
    const float exp_sum = block_reduce_sum(local_exp_sum);
    const float log_sum = logf(fmaxf(exp_sum, 1e-10f)) + log_max;

    // --- Step 4: top-p threshold ---
    // For exact nucleus sampling, we need sorted cumulative probabilities.
    // An efficient GPU approach: use iterative threshold on probability.
    // Binary search for the smallest probability threshold such that the
    // sum of probabilities above it is >= top_p.
    float topp_logit_threshold = -FLT_MAX;
    if (top_p < 1.0f && top_p > 0.0f) {
        // Binary search on the logit threshold
        float p_hi = log_max;  // highest logit
        float p_lo = log_max - 20.0f; // very low (prob ~ exp(-20) ≈ 2e-9)

        __shared__ float s_psum[kMaxWarpsPerBlock];
        const int lane = threadIdx.x % kWarpSize;
        const int warp = threadIdx.x / kWarpSize;

        for (int iter = 0; iter < 50; ++iter) {
            float mid = 0.5f * (p_hi + p_lo);
            // Sum probabilities of tokens with scaled_logit >= mid (and also >= topk_threshold)
            float local_psum = 0.0f;
            for (int64_t c = tid; c < cols; c += blockDim.x) {
                float v = to_float(row_logits[c]) / temperature;
                if (v >= topk_threshold && v >= mid) {
                    local_psum += expf(v - log_sum);
                }
            }
            // Block reduce
            float psum = local_psum;
            for (int off = kWarpSize / 2; off >= 1; off >>= 1)
                psum += __shfl_xor_sync(kFullWarpMask, psum, off);
            if (lane == 0) s_psum[warp] = psum;
            __syncthreads();
            if (warp == 0) {
                const int wc = blockDim.x / kWarpSize;
                float total = (lane < wc) ? s_psum[lane] : 0.0f;
                for (int off = kWarpSize / 2; off >= 1; off >>= 1)
                    total += __shfl_xor_sync(kFullWarpMask, total, off);
                if (lane == 0) s_psum[0] = total;
            }
            __syncthreads();

            if (s_psum[0] >= top_p) {
                p_lo = mid; // can raise threshold — still have enough probability mass
            } else {
                p_hi = mid; // too aggressive, lower the threshold
            }
            if (p_hi - p_lo < 1e-6f) break;
        }
        topp_logit_threshold = p_lo;
    }

    // --- Step 5: Gumbel-max sampling with combined mask ---
    const float combined_threshold = fmaxf(topk_threshold, topp_logit_threshold);

    __shared__ float shared_scores[kThreadsPerBlock];
    __shared__ int shared_indices[kThreadsPerBlock];

    float best_score = -FLT_MAX;
    int best_index = 0;

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        float scaled_logit = to_float(row_logits[col]) / temperature;

        // Apply combined top-k + top-p mask
        if (scaled_logit < combined_threshold) continue;

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

template <typename T>
void launch_sample_tokens(const Tensor& logits,
                          const Tensor& temperatures,
                          const Tensor& top_ks,
                          const Tensor& top_ps,
                          uint64_t seed,
                          Tensor& output,
                          cudaStream_t stream) {
    const MatrixShape shape = flatten_by_last_dim("logits", logits);
    if (shape.rows == 0) return;

    sample_tokens_topk_topp_kernel<T><<<static_cast<unsigned int>(shape.rows), kThreadsPerBlock, 0, stream>>>(
        logits.data_as<T>(),
        temperatures.data_as<float>(),
        top_ks.data_as<int32_t>(),
        top_ps.data_as<float>(),
        output.data_as<int32_t>(),
        shape.rows,
        shape.cols,
        seed);
    throw_if_cuda_error(cudaPeekAtLastError(), "sample_tokens_topk_topp_kernel launch");
}

} // namespace

// ---------------------------------------------------------------------------
// Public API: apply_repetition_penalty
// ---------------------------------------------------------------------------
void apply_repetition_penalty(Tensor& logits,
                              const Tensor& penalty_token_ids,
                              const Tensor& penalty_token_counts,
                              const Tensor& penalties,
                              cudaStream_t stream) {
    ensure_cuda_tensor("logits", logits);
    ensure_cuda_tensor("penalty_token_ids", penalty_token_ids);
    ensure_cuda_tensor("penalty_token_counts", penalty_token_counts);
    ensure_cuda_tensor("penalties", penalties);
    validate_supported_dtype("logits", logits.dtype());

    const MatrixShape shape = flatten_by_last_dim("logits", logits);
    if (shape.rows == 0) return;
    if (penalty_token_ids.dim() != 2 || penalty_token_ids.sizes()[0] != shape.rows) {
        throw std::invalid_argument("penalty_token_ids must have shape [batch_size, max_penalty_tokens]");
    }
    if (penalty_token_counts.dim() != 1 || penalty_token_counts.numel() != static_cast<size_t>(shape.rows)) {
        throw std::invalid_argument("penalty_token_counts must have shape [batch_size]");
    }
    if (penalties.dim() != 1 || penalties.numel() != static_cast<size_t>(shape.rows)) {
        throw std::invalid_argument("penalties must have shape [batch_size]");
    }

    switch (logits.dtype()) {
    case ScalarType::kFloat32:
        launch_repetition_penalty<float>(logits, penalty_token_ids, penalty_token_counts, penalties, stream);
        return;
    case ScalarType::kFloat16:
        launch_repetition_penalty<__half>(logits, penalty_token_ids, penalty_token_counts, penalties, stream);
        return;
    case ScalarType::kBFloat16:
        launch_repetition_penalty<__nv_bfloat16>(logits, penalty_token_ids, penalty_token_counts, penalties, stream);
        return;
    default:
        throw std::invalid_argument("unsupported dtype for apply_repetition_penalty");
    }
}

// ---------------------------------------------------------------------------
// Public API: sample_tokens  (with top-k / top-p)
// ---------------------------------------------------------------------------
void sample_tokens(const Tensor& logits,
                   const Tensor& temperatures,
                   const Tensor& top_ks,
                   const Tensor& top_ps,
                   uint64_t seed,
                   Tensor& output,
                   cudaStream_t stream) {
    ensure_cuda_tensor("logits", logits);
    ensure_cuda_tensor("temperatures", temperatures);
    ensure_cuda_tensor("top_ks", top_ks);
    ensure_cuda_tensor("top_ps", top_ps);
    ensure_cuda_tensor("output", output);
    ensure_same_device("logits", logits, "temperatures", temperatures);
    ensure_same_device("logits", logits, "top_ks", top_ks);
    ensure_same_device("logits", logits, "top_ps", top_ps);
    ensure_same_device("logits", logits, "output", output);
    validate_supported_dtype("logits", logits.dtype());
    if (temperatures.dtype() != ScalarType::kFloat32) {
        throw std::invalid_argument("temperatures must be float32");
    }
    if (top_ks.dtype() != ScalarType::kInt32) {
        throw std::invalid_argument("top_ks must be int32");
    }
    if (top_ps.dtype() != ScalarType::kFloat32) {
        throw std::invalid_argument("top_ps must be float32");
    }
    if (output.dtype() != ScalarType::kInt32) {
        throw std::invalid_argument("output must be int32");
    }
    if (logits.dim() != 2) {
        throw std::invalid_argument("logits must have shape [batch_size, vocab_size]");
    }
    const auto batch_size = logits.sizes()[0];
    if (temperatures.dim() != 1 || temperatures.numel() != static_cast<size_t>(batch_size)) {
        throw std::invalid_argument("temperatures must have shape [batch_size]");
    }
    if (top_ks.dim() != 1 || top_ks.numel() != static_cast<size_t>(batch_size)) {
        throw std::invalid_argument("top_ks must have shape [batch_size]");
    }
    if (top_ps.dim() != 1 || top_ps.numel() != static_cast<size_t>(batch_size)) {
        throw std::invalid_argument("top_ps must have shape [batch_size]");
    }
    if (output.dim() != 1 || output.numel() != static_cast<size_t>(batch_size)) {
        throw std::invalid_argument("output must have shape [batch_size]");
    }
    if (logits.sizes()[1] <= 0) {
        throw std::invalid_argument("logits vocab dimension must be positive");
    }

    switch (logits.dtype()) {
    case ScalarType::kFloat32:
        launch_sample_tokens<float>(logits, temperatures, top_ks, top_ps, seed, output, stream); return;
    case ScalarType::kFloat16:
        launch_sample_tokens<__half>(logits, temperatures, top_ks, top_ps, seed, output, stream); return;
    case ScalarType::kBFloat16:
        launch_sample_tokens<__nv_bfloat16>(logits, temperatures, top_ks, top_ps, seed, output, stream); return;
    default:
        throw std::invalid_argument("unsupported dtype for sample_tokens");
    }
}

} // namespace nano_vllm
