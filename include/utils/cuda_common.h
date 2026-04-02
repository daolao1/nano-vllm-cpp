#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <string>

namespace nano_vllm {

std::string cuda_error_string(cudaError_t status);
std::string cublas_error_string(cublasStatus_t status);

void throw_if_cuda_error(cudaError_t status, const char* context);
void throw_if_cublas_error(cublasStatus_t status, const char* context);

} // namespace nano_vllm