#include "engine/scheduler.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace nano_vllm {

Scheduler::Scheduler(const Config& config)
    : max_num_seqs_(config.max_num_seqs),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      max_num_prefill_tokens_(config.max_num_prefill_tokens),
      eos_(config.eos_token_id),
      enable_chunked_prefill_(config.enable_chunked_prefill),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size) {
    assert(config.num_kvcache_blocks > 0);
}

void Scheduler::add(Sequence seq) {
    sequences_.push_back(std::move(seq));
    waiting_.push_back(&sequences_.back());
}

bool Scheduler::is_finished() const {
    return waiting_.empty() && running_.empty();
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() {
    assert(!is_finished());
    if (enable_chunked_prefill_) {
        return schedule_chunked();
    }
    return schedule_default();
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule_default() {

    std::vector<Sequence*> scheduled_seqs;
    int num_seqs = 0;
    int num_batched_tokens = 0;

    while (!waiting_.empty() && num_seqs < max_num_seqs_) {
        Sequence* seq = waiting_.front();
        if (num_batched_tokens + seq->len() > max_num_batched_tokens_ ||
            !block_manager_.can_allocate(*seq)) {
            break;
        }

        ++num_seqs;
        block_manager_.allocate(*seq);
        num_batched_tokens += seq->len() - seq->num_cached_tokens;
        // Default path: single-chunk prefill.  Must process at least one
        // query token even when the entire prompt is prefix-cached, so the
        // model can produce a first decode logit.
        const int32_t computed =
            std::min<int32_t>(seq->num_cached_tokens, seq->num_tokens - 1);
        seq->num_computed_tokens = computed;
        seq->num_tokens_to_process = seq->num_tokens - computed;
        seq->status = SequenceStatus::RUNNING;
        waiting_.pop_front();
        running_.push_back(seq);
        scheduled_seqs.push_back(seq);
    }

    if (!scheduled_seqs.empty()) {
        return {scheduled_seqs, true};
    }

    while (!running_.empty() && num_seqs < max_num_seqs_) {
        Sequence* seq = running_.front();
        running_.pop_front();

        bool was_preempted = false;
        while (!block_manager_.can_append(*seq)) {
            if (!running_.empty()) {
                Sequence* victim = running_.back();
                running_.pop_back();
                preempt(victim);
            } else {
                preempt(seq);
                was_preempted = true;
                break;
            }
        }

        if (was_preempted) {
            continue;
        }

        ++num_seqs;
        block_manager_.may_append(*seq);
        seq->num_tokens_to_process = 0;  // decode marker for postprocess
        scheduled_seqs.push_back(seq);
    }

    assert(!scheduled_seqs.empty());
    for (auto it = scheduled_seqs.rbegin(); it != scheduled_seqs.rend(); ++it) {
        running_.push_front(*it);
    }
    return {scheduled_seqs, false};
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule_chunked() {
    // Minimal chunked prefill: prefill and decode still use separate steps,
    // but long prefills are split across multiple steps under the
    // max_num_prefill_tokens budget.  Prefix-cache accounting is unchanged.
    //
    // Scheduling order:
    //   1. Admit new sequences from waiting_: allocate blocks, seed
    //      num_computed_tokens = num_cached_tokens (bounded to num_tokens-1).
    //   2. If any running sequence still has num_computed_tokens < num_tokens,
    //      emit a prefill-only batch that splits the remaining budget across
    //      those sequences in queue order.
    //   3. Otherwise, run a decode step identical to schedule_default.

    std::vector<Sequence*> scheduled_seqs;
    int num_seqs = 0;

    // Phase 1: admit newcomers.  Seed num_computed_tokens from the prefix
    // cache hit count (bounded by num_tokens - 1 so we always have at least
    // one query token to process on the terminal prefill chunk).
    while (!waiting_.empty() &&
           num_seqs + static_cast<int>(running_.size()) < max_num_seqs_) {
        Sequence* seq = waiting_.front();
        if (!block_manager_.can_allocate(*seq)) {
            break;
        }
        block_manager_.allocate(*seq);
        const int32_t computed =
            std::min<int32_t>(seq->num_cached_tokens, seq->num_tokens - 1);
        seq->num_computed_tokens = computed;
        seq->num_tokens_to_process = 0;  // scheduler will set below
        seq->status = SequenceStatus::RUNNING;
        waiting_.pop_front();
        running_.push_back(seq);
        ++num_seqs;
    }

    // Phase 2: gather running seqs that still need prefill.
    // "Still prefilling" = at least 2 uncomputed positions remain, because
    // the final uncomputed position is handled by a decode step (q_len=1).
    std::vector<Sequence*> prefill_candidates;
    for (Sequence* seq : running_) {
        if (seq->num_computed_tokens < seq->num_tokens - 1) {
            prefill_candidates.push_back(seq);
        }
    }

    if (!prefill_candidates.empty()) {
        const int total_budget =
            std::min(max_num_prefill_tokens_, max_num_batched_tokens_);
        int budget = total_budget;
        for (Sequence* seq : prefill_candidates) {
            if (static_cast<int>(scheduled_seqs.size()) >= max_num_seqs_) break;
            const int remaining = seq->num_tokens - seq->num_computed_tokens;
            if (remaining <= 0) continue;
            const int chunk = std::min(remaining, budget);
            if (chunk <= 0) break;
            seq->num_tokens_to_process = chunk;
            budget -= chunk;
            scheduled_seqs.push_back(seq);
        }
        if (!scheduled_seqs.empty()) {
            return {scheduled_seqs, true};
        }
        // Fell through: nothing fit the budget (very small budget).  Force at
        // least one token on the first candidate so forward progress is made.
        Sequence* seq = prefill_candidates.front();
        seq->num_tokens_to_process = 1;
        scheduled_seqs.push_back(seq);
        return {scheduled_seqs, true};
    }

    // Phase 3: decode (same semantics as schedule_default).
    num_seqs = 0;
    while (!running_.empty() && num_seqs < max_num_seqs_) {
        Sequence* seq = running_.front();
        running_.pop_front();

        bool was_preempted = false;
        while (!block_manager_.can_append(*seq)) {
            if (!running_.empty()) {
                Sequence* victim = running_.back();
                running_.pop_back();
                preempt(victim);
            } else {
                preempt(seq);
                was_preempted = true;
                break;
            }
        }

        if (was_preempted) continue;

        ++num_seqs;
        block_manager_.may_append(*seq);
        seq->num_tokens_to_process = 0;
        scheduled_seqs.push_back(seq);
    }

    assert(!scheduled_seqs.empty());
    for (auto it = scheduled_seqs.rbegin(); it != scheduled_seqs.rend(); ++it) {
        running_.push_front(*it);
    }
    return {scheduled_seqs, false};
}

void Scheduler::postprocess(const std::vector<Sequence*>& seqs,
                            const std::vector<int32_t>& token_ids) {
    assert(seqs.size() == token_ids.size());

    for (size_t i = 0; i < seqs.size(); ++i) {
        Sequence* seq = seqs[i];
        const int32_t token_id = token_ids[i];

        if (seq->num_tokens_to_process > 0) {
            // Prefill step: advance computed watermark and decide whether the
            // sampled token is meaningful (only on the terminal chunk).
            const int32_t new_computed =
                seq->num_computed_tokens + seq->num_tokens_to_process;
            const bool finished_prefill = (new_computed == seq->num_tokens);
            seq->num_computed_tokens = new_computed;
            seq->num_tokens_to_process = 0;

            if (!finished_prefill) {
                // Mid-chunk: sampled token is garbage, ignore it.
                continue;
            }
            // Terminal chunk: fall through to append_token + EOS check below.
        }

        // Decode step (or terminal prefill chunk): append sampled token and
        // check for termination.
        seq->append_token(token_id);

        if ((!seq->ignore_eos && token_id == eos_) ||
            seq->num_completion_tokens() == seq->max_tokens) {
            seq->status = SequenceStatus::FINISHED;
            block_manager_.deallocate(*seq);

            auto it = std::find(running_.begin(), running_.end(), seq);
            if (it != running_.end()) {
                running_.erase(it);
            }
        }
    }
}

void Scheduler::preempt(Sequence* seq) {
    seq->status = SequenceStatus::WAITING;
    block_manager_.deallocate(*seq);
    waiting_.push_front(seq);
}

} // namespace nano_vllm