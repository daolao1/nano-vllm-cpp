#pragma once

#include "core/config.h"
#include "engine/block_manager.h"

#include <cstdint>
#include <deque>
#include <list>
#include <utility>
#include <vector>

namespace nano_vllm {

class Scheduler {
public:
    explicit Scheduler(const Config& config);

    void add(Sequence seq);
    bool is_finished() const;

    /// Returns (scheduled_seqs, is_prefill).
    /// When chunked prefill is enabled, is_prefill means a mixed batch
    /// containing prefill sequences (others may be decoding).
    std::pair<std::vector<Sequence*>, bool> schedule();
    void postprocess(const std::vector<Sequence*>& seqs,
                     const std::vector<int32_t>& token_ids);

private:
    void preempt(Sequence* seq);
    std::pair<std::vector<Sequence*>, bool> schedule_default();
    std::pair<std::vector<Sequence*>, bool> schedule_chunked();

    int max_num_seqs_;
    int max_num_batched_tokens_;
    int max_num_prefill_tokens_;
    int eos_;
    bool enable_chunked_prefill_;
    BlockManager block_manager_;

    // Own sequences separately so queue pointers remain stable.
    std::list<Sequence> sequences_;
    std::deque<Sequence*> waiting_;
    std::deque<Sequence*> running_;
};

} // namespace nano_vllm