#pragma once

#include "core/config.h"
#include "core/sampling_params.h"
#include "engine/model_runner.h"
#include "engine/scheduler.h"
#include "utils/tokenizer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nano_vllm {

class LLMEngine {
public:
    using StreamCallback = std::function<void(int64_t seq_id,
                                              int32_t token_id,
                                              const std::string& token_text,
                                              bool finished)>;

    struct Output {
        std::string text;
        std::vector<int32_t> token_ids;
    };

    /// Construct the engine: loads model, tokenizer, creates scheduler + model_runner.
    LLMEngine(const std::string& model,
              int max_model_len = 4096,
              int max_num_seqs = 512,
              int max_num_batched_tokens = 16384,
              float gpu_memory_utilization = 0.9f,
              int tensor_parallel_size = 1,
              bool enforce_eager = false,
              int kvcache_block_size = 256,
              bool kv_cache_int8 = false);

    /// Add a request (string prompt — will be tokenized).
    void add_request(const std::string& prompt, const SamplingParams& params);

    /// Add a request (pre-tokenized).
    void add_request(const std::vector<int32_t>& token_ids, const SamplingParams& params);

    /// Run one scheduling + model step.
    /// Returns (finished outputs, num_tokens processed).
    /// num_tokens > 0 for prefill, < 0 (-batch_size) for decode.
    /// step_tokens, if provided, receives one sampled token per scheduled seq.
    std::pair<std::vector<std::pair<int64_t, std::vector<int32_t>>>, int> step(
        std::vector<std::pair<int64_t, int32_t>>* step_tokens = nullptr);

    /// Check if all requests are done.
    bool is_finished() const;

    /// Run to completion, returning decoded text for each prompt.
    std::vector<Output> generate(const std::vector<std::string>& prompts,
                                 const SamplingParams& params);

    /// Generate from pre-tokenized prompts.
    std::vector<Output> generate(const std::vector<std::vector<int32_t>>& prompts,
                                 const SamplingParams& params);

    /// Run to completion with per-token stream callback.
    std::vector<Output> generate_stream(const std::vector<std::string>& prompts,
                                        const SamplingParams& params,
                                        const StreamCallback& on_token);

    /// Generate from pre-tokenized prompts with per-token stream callback.
    std::vector<Output> generate_stream(const std::vector<std::vector<int32_t>>& prompts,
                                        const SamplingParams& params,
                                        const StreamCallback& on_token);

    // Accessors.
    const Config& config() const { return config_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }

private:
    Config config_;
    Tokenizer tokenizer_;
    DeviceAllocator& allocator_;
    ModelRunner model_runner_;
    std::unique_ptr<Scheduler> scheduler_;
};

} // namespace nano_vllm
