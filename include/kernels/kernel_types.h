#pragma once

#include "utils/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nano_vllm {

// ---- Kernel metadata ----
struct KernelSpec {
    const char* name;
    const char* purpose;
};

const std::vector<KernelSpec>& phase1_kernel_specs();

// ---- Validation helpers ----
void ensure_cuda_tensor(const char* name, const Tensor& tensor);
void ensure_same_device(const char* lhs_name, const Tensor& lhs,
                        const char* rhs_name, const Tensor& rhs);
void ensure_same_shape(const char* lhs_name, const Tensor& lhs,
                       const char* rhs_name, const Tensor& rhs);
void ensure_same_dtype(const char* lhs_name, const Tensor& lhs,
                       const char* rhs_name, const Tensor& rhs);
void validate_supported_dtype(const char* name, ScalarType dtype);

// ---- Internal shape helpers (used by kernel .cu files) ----
struct MatrixShape {
    int64_t rows = 0;
    int64_t cols = 0;
};

MatrixShape flatten_by_last_dim(const char* name, const Tensor& tensor);
MatrixShape flatten_by_leading_count(const char* name, const Tensor& tensor, int64_t leading_count);
int64_t cache_rows_from_tensor(const char* name, const Tensor& tensor, int64_t row_width);
int choose_thread_count(int64_t work_items);
bool is_pointer_aligned(const void* ptr, std::uintptr_t alignment);

} // namespace nano_vllm
