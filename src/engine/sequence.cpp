#include "engine/sequence.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace nano_vllm {

std::atomic<int64_t> Sequence::next_seq_id_{0};

Sequence::Sequence(std::vector<int32_t> token_ids, SamplingParams params)
    : seq_id(next_seq_id_.fetch_add(1, std::memory_order_relaxed)),
      token_ids(std::move(token_ids)),
      temperature(params.temperature),
      top_k(params.top_k),
      top_p(params.top_p),
      repetition_penalty(params.repetition_penalty),
      max_tokens(params.max_tokens),
      ignore_eos(params.ignore_eos) {
    num_tokens = static_cast<int32_t>(this->token_ids.size());
    num_prompt_tokens = num_tokens;
    if (!this->token_ids.empty()) {
        last_token = this->token_ids.back();
    }
}

int Sequence::last_block_num_tokens() const {
    if (num_tokens == 0) {
        return 0;
    }
    return num_tokens - (num_blocks() - 1) * BLOCK_SIZE;
}

std::vector<int32_t> Sequence::block(int i) const {
    if (i < 0 || i >= num_blocks()) {
        throw std::out_of_range("block index out of range: " + std::to_string(i));
    }

    const int start = i * BLOCK_SIZE;
    const int end = std::min(start + BLOCK_SIZE, num_tokens);
    return {token_ids.begin() + start, token_ids.begin() + end};
}

void Sequence::append_token(int32_t token_id) {
    token_ids.push_back(token_id);
    last_token = token_id;
    ++num_tokens;
}

} // namespace nano_vllm