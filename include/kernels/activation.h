#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void silu_and_mul(const Tensor& gate_up,
                  Tensor& output,
                  cudaStream_t stream = nullptr);

} // namespace nano_vllm
