#include "utils/context.h"

namespace nano_vllm {
namespace {

thread_local Context g_context;

} // namespace

Context& get_context() {
    return g_context;
}

void set_context(const Context& context) {
    g_context = context;
}

void set_context(bool is_prefill,
                 const Tensor& cu_seqlens_q,
                 const Tensor& cu_seqlens_k,
                 int max_seqlen_q,
                 int max_seqlen_k,
                 const Tensor& slot_mapping,
                 const Tensor& context_lens,
                 const Tensor& block_tables) {
    Context context;
    context.is_prefill = is_prefill;
    context.cu_seqlens_q = cu_seqlens_q;
    context.cu_seqlens_k = cu_seqlens_k;
    context.max_seqlen_q = max_seqlen_q;
    context.max_seqlen_k = max_seqlen_k;
    context.slot_mapping = slot_mapping;
    context.context_lens = context_lens;
    context.block_tables = block_tables;
    g_context = context;
}

void reset_context() {
    g_context = Context{};
}

} // namespace nano_vllm