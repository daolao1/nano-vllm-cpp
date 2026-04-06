#pragma once

#include "utils/tensor.h"

#include <cstdint>

namespace nano_vllm {

/// Apply repetition penalty in-place: for each row, scale logits at positions
/// indicated by penalty_token_ids.  Positive logits are divided by the penalty,
/// negative logits are multiplied.  penalty == 1.0 is a no-op.
void apply_repetition_penalty(Tensor& logits,
                              const Tensor& penalty_token_ids,
                              const Tensor& penalty_token_counts,
                              const Tensor& penalties,
                              cudaStream_t stream = nullptr);

/// Sample one token per row from logits with temperature, top-k, and top-p.
///   temperatures : [batch_size]  (float32)
///   top_ks       : [batch_size]  (int32, <= 0 means disabled)
///   top_ps       : [batch_size]  (float32, 1.0 means disabled)
void sample_tokens(const Tensor& logits,
                   const Tensor& temperatures,
                   const Tensor& top_ks,
                   const Tensor& top_ps,
                   uint64_t seed,
                   Tensor& output,
                   cudaStream_t stream = nullptr);

} // namespace nano_vllm
