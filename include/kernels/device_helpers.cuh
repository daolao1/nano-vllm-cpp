#pragma once

// Device-side helpers shared across kernel .cu files.
// This header must only be #included from .cu files (it uses CUDA intrinsics).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace nano_vllm {
namespace kernel_detail {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
constexpr int kMaxWarpsPerBlock = (kThreadsPerBlock + kWarpSize - 1) / kWarpSize;

// ---- dtype conversion ----

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

// ---- reduction ----

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset >= 1; offset >>= 1) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float shared[kMaxWarpsPerBlock];

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = blockDim.x / kWarpSize;
        value = (lane < warp_count) ? shared[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

// ---- sampling helpers ----

__device__ __forceinline__ uint64_t splitmix64(uint64_t state) {
    state += 0x9e3779b97f4a7c15ull;
    state = (state ^ (state >> 30)) * 0xbf58476d1ce4e5b9ull;
    state = (state ^ (state >> 27)) * 0x94d049bb133111ebull;
    return state ^ (state >> 31);
}

__device__ __forceinline__ float uniform01_from_u64(uint64_t state) {
    const uint64_t bits = (splitmix64(state) >> 11) | 1ull;
    return static_cast<float>(static_cast<double>(bits) *
                              (1.0 / static_cast<double>(1ull << 53)));
}

__device__ __forceinline__ void update_best(float candidate_score,
                                            int candidate_index,
                                            float& best_score,
                                            int& best_index) {
    if (candidate_score > best_score ||
        (candidate_score == best_score && candidate_index < best_index)) {
        best_score = candidate_score;
        best_index = candidate_index;
    }
}

} // namespace kernel_detail
} // namespace nano_vllm
