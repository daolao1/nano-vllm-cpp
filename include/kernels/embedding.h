#pragma once

#include "utils/tensor.h"

#include <cstdint>

namespace nano_vllm {

void embedding_lookup(const Tensor& input_ids,
                      const Tensor& weight,
                      int64_t vocab_size,
                      int64_t vocab_start,
                      Tensor& output,
                      cudaStream_t stream = nullptr);

} // namespace nano_vllm
