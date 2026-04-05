#pragma once

#include "utils/cuda_common.h"
#include "utils/tensor.h"

#include <cublasLt.h>

#include <mutex>
#include <unordered_map>

namespace nano_vllm {

enum class GemmTranspose {
    kNone,
    kTranspose,
};

struct GemmConfig {
    int m = 0;
    int n = 0;
    int k = 0;
    GemmTranspose trans_a = GemmTranspose::kNone;
    GemmTranspose trans_b = GemmTranspose::kNone;
    float alpha = 1.0f;
    float beta = 0.0f;
};

class CublasHandle {
public:
    CublasHandle();
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept;
    CublasHandle& operator=(CublasHandle&& other) noexcept;

    void set_stream(cudaStream_t stream);
    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
    void* workspace_ = nullptr;
    cudaStream_t current_stream_ = reinterpret_cast<cudaStream_t>(~uintptr_t(0)); // sentinel
    static constexpr size_t kWorkspaceBytes = 32 << 20; // 32 MB
};

/// Lightweight cublasLt wrapper with per-shape heuristic caching.
class CublasLtContext {
public:
    static CublasLtContext& instance();

    /// Row-major GEMM: C[M,N] = A[M,K] * B^T, where B is stored [N,K].
    void gemm_row_major(int m, int n, int k, ScalarType type,
                        const void* a, const void* b, void* c,
                        cudaStream_t stream);

private:
    CublasLtContext();
    ~CublasLtContext();

    struct ShapeKey {
        int m, n, k;
        bool operator==(const ShapeKey& o) const { return m == o.m && n == o.n && k == o.k; }
    };
    struct ShapeKeyHash {
        size_t operator()(const ShapeKey& k) const {
            return std::hash<int>()(k.m) ^ (std::hash<int>()(k.n) << 11) ^ (std::hash<int>()(k.k) << 22);
        }
    };

    cublasLtHandle_t handle_ = nullptr;
    void* workspace_ = nullptr;
    static constexpr size_t kWorkspaceBytes = 32 << 20;

    struct CachedPlan {
        cublasLtMatmulAlgo_t algo{};
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatrixLayout_t layA = nullptr;
        cublasLtMatrixLayout_t layB = nullptr;
        cublasLtMatrixLayout_t layC = nullptr;
    };

    std::mutex cache_mutex_;
    std::unordered_map<ShapeKey, CachedPlan, ShapeKeyHash> plan_cache_;

    const CachedPlan& get_plan(int m, int n, int k, ScalarType type);
};

void gemm_row_major(const CublasHandle& handle,
                    const GemmConfig& config,
                    ScalarType type,
                    const void* a,
                    const void* b,
                    void* c);

} // namespace nano_vllm