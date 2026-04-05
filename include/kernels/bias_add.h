#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void add_bias(Tensor& input_output,
              const Tensor& bias,
              cudaStream_t stream = nullptr);

void add_inplace(Tensor& input_output,
                 const Tensor& input,
                 cudaStream_t stream = nullptr);

} // namespace nano_vllm
