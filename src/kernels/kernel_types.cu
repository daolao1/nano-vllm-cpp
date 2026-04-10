#include "kernels/kernel_types.h"

#include "utils/cuda_common.h"

#include <stdexcept>

namespace nano_vllm {

// ---- phase1_kernel_specs (moved from kernel_ops.cpp) ----

const std::vector<KernelSpec>& phase1_kernel_specs() {
    static const std::vector<KernelSpec> specs = {
        {"rms_norm", "row-wise RMSNorm over the hidden dimension"},
        {"add_rms_norm", "fused residual add plus RMSNorm"},
        {"rotary_embedding", "apply RoPE to query and key tensors"},
        {"silu_and_mul", "split gate/up projection and apply SiLU gating"},
        {"store_kvcache", "write per-token KV vectors into paged cache slots"},
    };
    return specs;
}

// ---- Validation helpers ----

void ensure_cuda_tensor(const char* name, const Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument(std::string(name) + " must live on CUDA");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    }
}

void ensure_same_device(const char* lhs_name,
                        const Tensor& lhs,
                        const char* rhs_name,
                        const Tensor& rhs) {
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must be on the same device");
    }
}

void ensure_same_shape(const char* lhs_name,
                       const Tensor& lhs,
                       const char* rhs_name,
                       const Tensor& rhs) {
    if (lhs.sizes() != rhs.sizes()) {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same shape");
    }
}

void ensure_same_dtype(const char* lhs_name,
                       const Tensor& lhs,
                       const char* rhs_name,
                       const Tensor& rhs) {
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument(std::string(lhs_name) + " and " + rhs_name + " must have the same dtype");
    }
}

void validate_supported_dtype(const char* name, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat16:
    case ScalarType::kBFloat16:
    case ScalarType::kFloat32:
        return;
    default:
        throw std::invalid_argument(std::string(name) + " must be float16, bfloat16, or float32");
    }
}

// ---- Shape helpers ----

MatrixShape flatten_by_last_dim(const char* name, const Tensor& tensor) {
    if (tensor.dim() < 1) {
        throw std::invalid_argument(std::string(name) + " must have rank >= 1");
    }
    const int64_t cols = tensor.sizes().back();
    if (cols <= 0) {
        throw std::invalid_argument(std::string(name) + " must have a positive last dimension");
    }
    const int64_t rows = static_cast<int64_t>(tensor.numel() / static_cast<size_t>(cols));
    if (rows * cols != static_cast<int64_t>(tensor.numel())) {
        throw std::invalid_argument(std::string(name) + " cannot be flattened by last dimension");
    }
    return {rows, cols};
}

MatrixShape flatten_by_leading_count(const char* name, const Tensor& tensor, int64_t leading_count) {
    if (leading_count <= 0) {
        throw std::invalid_argument(std::string(name) + " must have a positive leading count");
    }
    const int64_t cols = static_cast<int64_t>(tensor.numel() / static_cast<size_t>(leading_count));
    if (cols <= 0 || cols * leading_count != static_cast<int64_t>(tensor.numel())) {
        throw std::invalid_argument(std::string(name) + " cannot be flattened by the provided leading count");
    }
    return {leading_count, cols};
}

int64_t cache_rows_from_tensor(const char* name, const Tensor& tensor, int64_t row_width) {
    if (row_width <= 0) {
        throw std::invalid_argument(std::string(name) + " row width must be positive");
    }
    const int64_t rows = static_cast<int64_t>(tensor.numel() / static_cast<size_t>(row_width));
    if (rows <= 0 || rows * row_width != static_cast<int64_t>(tensor.numel())) {
        throw std::invalid_argument(std::string(name) + " does not contain an integer number of cache rows");
    }
    return rows;
}

int choose_thread_count(int64_t work_items) {
    constexpr int kWarpSize = 32;
    constexpr int kMaxThreads = 256;
    int threads = kWarpSize;
    while (threads < work_items && threads < kMaxThreads) {
        threads <<= 1;
    }
    return threads > kMaxThreads ? kMaxThreads : threads;
}

bool is_pointer_aligned(const void* ptr, std::uintptr_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

} // namespace nano_vllm
