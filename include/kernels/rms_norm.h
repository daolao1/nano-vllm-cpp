#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void rms_norm(const Tensor& input,
              const Tensor& weight,
              float eps,
              Tensor& output,
              cudaStream_t stream = nullptr);

void add_rms_norm(const Tensor& input,
                  const Tensor& residual,
                  const Tensor& weight,
                  float eps,
                  Tensor& output,
                  Tensor& residual_out,
                  cudaStream_t stream = nullptr);

} // namespace nano_vllm
