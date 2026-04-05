#include "utils/cublas_wrapper.h"

#include "utils/cuda_common.h"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace nano_vllm {

namespace {

cudaDataType_t to_cuda_data_type(ScalarType type) {
    switch (type) {
    case ScalarType::kFloat16:
        return CUDA_R_16F;
    case ScalarType::kBFloat16:
        return CUDA_R_16BF;
    case ScalarType::kFloat32:
        return CUDA_R_32F;
    default:
        throw std::invalid_argument("gemm_row_major supports only float16/bfloat16/float32");
    }
}

cublasOperation_t to_cublas_row_major_op(GemmTranspose transpose) {
    return transpose == GemmTranspose::kNone ? CUBLAS_OP_N : CUBLAS_OP_T;
}

int row_major_leading_dim_for_a(const GemmConfig& config) {
    return config.trans_a == GemmTranspose::kNone ? config.k : config.m;
}

int row_major_leading_dim_for_b(const GemmConfig& config) {
    return config.trans_b == GemmTranspose::kNone ? config.n : config.k;
}

} // namespace

CublasHandle::CublasHandle() {
    throw_if_cublas_error(cublasCreate(&handle_), "cublasCreate");
    // Enable tensor core acceleration when available.
    cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH);
    // Pre-allocate workspace for better GEMM performance.
    if (cudaMalloc(&workspace_, kWorkspaceBytes) == cudaSuccess) {
        cublasSetWorkspace(handle_, workspace_, kWorkspaceBytes);
    }
}

CublasHandle::~CublasHandle() {
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
    }
    if (workspace_ != nullptr) {
        cudaFree(workspace_);
    }
}

CublasHandle::CublasHandle(CublasHandle&& other) noexcept
    : handle_(other.handle_), workspace_(other.workspace_) {
    other.handle_ = nullptr;
    other.workspace_ = nullptr;
}

CublasHandle& CublasHandle::operator=(CublasHandle&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (handle_ != nullptr) {
        cublasDestroy(handle_);
    }
    if (workspace_ != nullptr) {
        cudaFree(workspace_);
    }
    handle_ = other.handle_;
    workspace_ = other.workspace_;
    other.handle_ = nullptr;
    other.workspace_ = nullptr;
    return *this;
}

void CublasHandle::set_stream(cudaStream_t stream) {
    if (stream != current_stream_) {
        throw_if_cublas_error(cublasSetStream(handle_, stream), "cublasSetStream");
        current_stream_ = stream;
    }
}

void gemm_row_major(const CublasHandle& handle,
                    const GemmConfig& config,
                    ScalarType type,
                    const void* a,
                    const void* b,
                    void* c) {
    if (config.m < 0 || config.n < 0 || config.k < 0) {
        throw std::invalid_argument("gemm dimensions must be non-negative");
    }
    if (config.m == 0 || config.n == 0 || config.k == 0) {
        return;
    }

    const cudaDataType_t data_type = to_cuda_data_type(type);
    const cublasOperation_t op_a = to_cublas_row_major_op(config.trans_b);
    const cublasOperation_t op_b = to_cublas_row_major_op(config.trans_a);
    const int lda = row_major_leading_dim_for_b(config);
    const int ldb = row_major_leading_dim_for_a(config);
    const int ldc = config.n;
    const cublasComputeType_t compute_type =
        (data_type == CUDA_R_16BF || data_type == CUDA_R_16F)
            ? CUBLAS_COMPUTE_32F_FAST_16BF
            : CUBLAS_COMPUTE_32F;

    throw_if_cublas_error(
        cublasGemmEx(handle.get(),
                     op_a,
                     op_b,
                     config.n,
                     config.m,
                     config.k,
                     &config.alpha,
                     b,
                     data_type,
                     lda,
                     a,
                     data_type,
                     ldb,
                     &config.beta,
                     c,
                     data_type,
                     ldc,
                     compute_type,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx(row_major)");
}

// ---------------------------------------------------------------------------
// CublasLtContext
// ---------------------------------------------------------------------------

CublasLtContext& CublasLtContext::instance() {
    static CublasLtContext ctx;
    return ctx;
}

CublasLtContext::CublasLtContext() {
    throw_if_cublas_error(cublasLtCreate(&handle_), "cublasLtCreate");
    if (cudaMalloc(&workspace_, kWorkspaceBytes) != cudaSuccess) {
        workspace_ = nullptr;
    }
}

CublasLtContext::~CublasLtContext() {
    if (handle_) cublasLtDestroy(handle_);
    if (workspace_) cudaFree(workspace_);
}

const CublasLtContext::CachedPlan& CublasLtContext::get_plan(int m, int n, int k, ScalarType type) {
    ShapeKey key{m, n, k};
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = plan_cache_.find(key);
        if (it != plan_cache_.end()) return it->second;
    }

    // Row-major C[M,N] = A[M,K] * B^T, B stored [N,K]
    // Col-major equivalent: C'[N,M] = B_T[N,K] * A[K,M]
    // First arg (B): col-major [K,N] ld=K, with TRANSA=T → shape [N,K]
    // Second arg (A): col-major [K,M] ld=K, with TRANSB=N → shape [K,M]
    // Output (C): col-major [N,M] ld=N

    const cudaDataType_t data_type = to_cuda_data_type(type);

    cublasLtMatmulDesc_t desc;
    throw_if_cublas_error(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "ltDescCreate");
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t layB, layA, layC;
    cublasLtMatrixLayoutCreate(&layB, data_type, k, n, k);
    cublasLtMatrixLayoutCreate(&layA, data_type, k, m, k);
    cublasLtMatrixLayoutCreate(&layC, data_type, n, m, n);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = workspace_ ? kWorkspaceBytes : 0;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t results[8];
    int nResults = 0;
    cublasLtMatmulAlgoGetHeuristic(handle_, desc, layB, layA, layC, layC, pref, 8, results, &nResults);

    cublasLtMatmulPreferenceDestroy(pref);

    if (nResults == 0) {
        cublasLtMatrixLayoutDestroy(layA);
        cublasLtMatrixLayoutDestroy(layB);
        cublasLtMatrixLayoutDestroy(layC);
        cublasLtMatmulDescDestroy(desc);
        throw std::runtime_error("cublasLt: no algorithm for M=" + std::to_string(m) +
                                 " N=" + std::to_string(n) + " K=" + std::to_string(k));
    }

    CachedPlan plan;
    plan.algo = results[0].algo;
    plan.desc = desc;
    plan.layA = layA;
    plan.layB = layB;
    plan.layC = layC;

    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto [it, inserted] = plan_cache_.emplace(key, std::move(plan));
    return it->second;
}

void CublasLtContext::gemm_row_major(int m, int n, int k, ScalarType type,
                                     const void* a, const void* b, void* c,
                                     cudaStream_t stream) {
    if (m == 0 || n == 0 || k == 0) return;

    const CachedPlan& plan = get_plan(m, n, k, type);
    float alpha = 1.0f, beta = 0.0f;
    throw_if_cublas_error(
        cublasLtMatmul(handle_, plan.desc, &alpha,
                       b, plan.layB, a, plan.layA,
                       &beta, c, plan.layC, c, plan.layC,
                       &plan.algo,
                       workspace_, workspace_ ? kWorkspaceBytes : 0,
                       stream),
        "cublasLtMatmul");
}

} // namespace nano_vllm