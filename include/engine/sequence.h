#pragma once

#include "core/sampling_params.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace nano_vllm {

enum class SequenceStatus {
    WAITING,
    RUNNING,
    FINISHED,
};

class Sequence {
public:
    static constexpr int BLOCK_SIZE = 256;

    Sequence(std::vector<int32_t> token_ids, SamplingParams params);

    int len() const { return num_tokens; }
    bool is_finished() const { return status == SequenceStatus::FINISHED; }
    int num_completion_tokens() const { return num_tokens - num_prompt_tokens; }
    int num_cached_blocks() const { return num_cached_tokens / BLOCK_SIZE; }
    int num_blocks() const { return (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE; }
    int last_block_num_tokens() const;
    std::vector<int32_t> block(int i) const;
    void append_token(int32_t token_id);

    int64_t seq_id;
    SequenceStatus status = SequenceStatus::WAITING;
    std::vector<int32_t> token_ids;
    int32_t last_token = -1;
    int32_t num_tokens = 0;
    int32_t num_prompt_tokens = 0;
    int32_t num_cached_tokens = 0;
    // Number of prompt tokens whose KV has been materialized in the cache.
    // Always satisfies num_cached_tokens <= num_computed_tokens <= num_tokens.
    // For chunked prefill: advances chunk-by-chunk until == num_tokens, after
    // which the sequence transitions to decode.
    int32_t num_computed_tokens = 0;
    // Transient chunk size set by the scheduler each step (prefill only).
    // 0 means decode (or not scheduled this step).
    int32_t num_tokens_to_process = 0;
    std::vector<int32_t> block_table;
    float temperature = 1.0f;
    int top_k = -1;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    int max_tokens = 64;
    bool ignore_eos = false;

private:
    static std::atomic<int64_t> next_seq_id_;
};

} // namespace nano_vllm