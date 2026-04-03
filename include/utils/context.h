#pragma once

#include "utils/tensor.h"

namespace nano_vllm {

struct Context {
    bool is_prefill = false;
    Tensor cu_seqlens_q;
    Tensor cu_seqlens_k;
    int max_seqlen_q = 0;
    int max_seqlen_k = 0;
    Tensor slot_mapping;
    Tensor context_lens;
    Tensor block_tables;
};

Context& get_context();
void set_context(const Context& context);
void set_context(bool is_prefill,
                 const Tensor& cu_seqlens_q = Tensor(),
                 const Tensor& cu_seqlens_k = Tensor(),
                 int max_seqlen_q = 0,
                 int max_seqlen_k = 0,
                 const Tensor& slot_mapping = Tensor(),
                 const Tensor& context_lens = Tensor(),
                 const Tensor& block_tables = Tensor());
void reset_context();

} // namespace nano_vllm