#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

void rotary_embedding(const Tensor& positions,
                      const Tensor& cos_sin_cache,
                      Tensor& query,
                      Tensor& key,
                      cudaStream_t stream = nullptr);

} // namespace nano_vllm
