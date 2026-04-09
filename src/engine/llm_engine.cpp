#include "engine/llm_engine.h"

#include "utils/cuda_allocator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nano_vllm {

LLMEngine::LLMEngine(const std::string& model,
                     int max_model_len,
                     int max_num_seqs,
                     int max_num_batched_tokens,
                     float gpu_memory_utilization,
                     int tensor_parallel_size,
                     bool enforce_eager,
                     int kvcache_block_size,
                     bool kv_cache_int8)
    : config_(Config::from_model_dir(model,
                                     max_model_len,
                                     max_num_seqs,
                                     max_num_batched_tokens,
                                     gpu_memory_utilization,
                                     tensor_parallel_size,
                                     enforce_eager,
                                     kvcache_block_size,
                                     kv_cache_int8)),
      tokenizer_(model),
      allocator_(CudaAllocator::instance()),
      model_runner_(config_, allocator_, /*rank=*/0, ScalarType::kBFloat16) {
    // ModelRunner::allocate_kv_cache() computed num_kvcache_blocks inside its
    // own copy of config.  Copy it back so the Scheduler sees the real value.
    config_.num_kvcache_blocks = model_runner_.num_kvcache_blocks();
    scheduler_ = std::make_unique<Scheduler>(config_);

    // Use tokenizer's eos_token_id if config didn't get one.
    if (config_.eos_token_id < 0) {
        config_.eos_token_id = tokenizer_.eos_token_id();
    }
}

void LLMEngine::add_request(const std::string& prompt, const SamplingParams& params) {
    auto token_ids = tokenizer_.encode(prompt);
    if (token_ids.empty()) {
        throw std::invalid_argument("LLMEngine::add_request: empty prompt after tokenization");
    }
    add_request(token_ids, params);
}

void LLMEngine::add_request(const std::vector<int32_t>& token_ids, const SamplingParams& params) {
    Sequence seq(token_ids, params);
    scheduler_->add(std::move(seq));
}

std::pair<std::vector<std::pair<int64_t, std::vector<int32_t>>>, int>
LLMEngine::step(std::vector<std::pair<int64_t, int32_t>>* step_tokens) {
    auto [seqs, is_prefill] = scheduler_->schedule();
    auto token_ids = model_runner_.run(seqs, is_prefill);

    if (step_tokens != nullptr) {
        step_tokens->clear();
        step_tokens->reserve(seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) {
            step_tokens->emplace_back(seqs[i]->seq_id, token_ids[i]);
        }
    }

    scheduler_->postprocess(seqs, token_ids);

    // Collect finished outputs: (seq_id, completion_token_ids).
    std::vector<std::pair<int64_t, std::vector<int32_t>>> outputs;
    for (auto* seq : seqs) {
        if (seq->is_finished()) {
            // Completion tokens = token_ids[num_prompt_tokens:].
            std::vector<int32_t> completion(
                seq->token_ids.begin() + seq->num_prompt_tokens,
                seq->token_ids.end());
            outputs.emplace_back(seq->seq_id, std::move(completion));
        }
    }

    int num_tokens;
    if (is_prefill) {
        num_tokens = 0;
        for (auto* seq : seqs) {
            num_tokens += seq->len();
        }
    } else {
        num_tokens = -static_cast<int>(seqs.size());
    }

    return {std::move(outputs), num_tokens};
}

bool LLMEngine::is_finished() const {
    return scheduler_->is_finished();
}

std::vector<LLMEngine::Output> LLMEngine::generate(
    const std::vector<std::string>& prompts,
    const SamplingParams& params) {
    return generate_stream(prompts, params, StreamCallback{});
}

std::vector<LLMEngine::Output> LLMEngine::generate(
    const std::vector<std::vector<int32_t>>& prompts,
    const SamplingParams& params) {
    return generate_stream(prompts, params, StreamCallback{});
}

std::vector<LLMEngine::Output> LLMEngine::generate_stream(
    const std::vector<std::string>& prompts,
    const SamplingParams& params,
    const StreamCallback& on_token) {
    std::vector<std::vector<int32_t>> tokenized_prompts;
    tokenized_prompts.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        auto token_ids = tokenizer_.encode(prompt);
        if (token_ids.empty()) {
            throw std::invalid_argument(
                "LLMEngine::generate_stream: empty prompt after tokenization");
        }
        tokenized_prompts.push_back(std::move(token_ids));
    }

    return generate_stream(tokenized_prompts, params, on_token);
}

std::vector<LLMEngine::Output> LLMEngine::generate_stream(
    const std::vector<std::vector<int32_t>>& prompts,
    const SamplingParams& params,
    const StreamCallback& on_token) {
    for (auto& tids : prompts) {
        add_request(tids, params);
    }

    // Collect by seq_id for ordered output.
    std::map<int64_t, std::vector<int32_t>> collected;

    double prefill_throughput = 0.0;
    double decode_throughput = 0.0;

    while (!is_finished()) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::pair<int64_t, int32_t>> step_tokens;
        auto [finished, num_tokens] = step(&step_tokens);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        if (elapsed > 0.0) {
            if (num_tokens > 0) {
                prefill_throughput = num_tokens / elapsed;
            } else if (num_tokens < 0) {
                decode_throughput = (-num_tokens) / elapsed;
            }
        }

        std::unordered_set<int64_t> finished_ids;
        for (auto& [seq_id, tids] : finished) {
            finished_ids.insert(seq_id);
            collected[seq_id] = std::move(tids);
        }

        if (on_token) {
            for (const auto& [seq_id, token_id] : step_tokens) {
                const bool seq_finished = finished_ids.find(seq_id) != finished_ids.end();
                on_token(seq_id,
                         token_id,
                         tokenizer_.decode(std::vector<int32_t>{token_id}),
                         seq_finished);
            }
        }
    }

    std::cerr << "[LLMEngine] prefill: " << static_cast<int>(prefill_throughput)
              << " tok/s, decode: " << static_cast<int>(decode_throughput)
              << " tok/s\n";

    // Output in seq_id order (= input order).
    std::vector<Output> results;
    results.reserve(collected.size());
    for (auto& [seq_id, tids] : collected) {
        Output out;
        out.token_ids = std::move(tids);
        out.text = tokenizer_.decode(out.token_ids);
        results.push_back(std::move(out));
    }

    return results;
}

} // namespace nano_vllm
