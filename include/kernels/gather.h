#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void gather_last_tokens(const Tensor& input,
                        const Tensor& cu_seqlens,
                        Tensor& output,
                        cudaStream_t stream = nullptr);

} // namespace nano_vllm
