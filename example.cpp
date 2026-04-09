#include "core/sampling_params.h"
#include "engine/llm_engine.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Minimal chat template example (Qwen3):
//   <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
static std::string apply_chat_template(const std::string& user_message) {
    return "<|im_start|>user\n" + user_message + "<|im_end|>\n<|im_start|>assistant\n";
}

static void print_progress(int done, int total, int gen_tokens,
                           double prefill_tps, double decode_tps) {
    const int bar_width = 30;
    int filled = total > 0 ? done * bar_width / total : 0;
    std::fprintf(stderr, "\r\033[36m[");
    for (int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] %d/%d reqs | %d tokens", done, total, gen_tokens);
    if (prefill_tps > 0)
        std::fprintf(stderr, " | prefill %.0f tok/s", prefill_tps);
    if (decode_tps > 0)
        std::fprintf(stderr, " | decode %.0f tok/s", decode_tps);
    std::fprintf(stderr, "\033[0m");
    std::fflush(stderr);
}

int main(int argc, char** argv) {
    try {
        const std::string model_dir =
            argc > 1 ? argv[1] : "/home/zzy/huggingface/Qwen3-0.6B";

        // Check for --int8-kv flag
        bool kv_cache_int8 = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--int8-kv") kv_cache_int8 = true;
        }

        nano_vllm::LLMEngine engine(model_dir,
                                    /*max_model_len=*/4096,
                                    /*max_num_seqs=*/4,
                                    /*max_num_batched_tokens=*/4096,
                                    /*gpu_memory_utilization=*/0.9f,
                                    /*tensor_parallel_size=*/1,
                                    /*enforce_eager=*/true,
                                    /*kvcache_block_size=*/nano_vllm::Sequence::BLOCK_SIZE,
                                    /*kv_cache_int8=*/kv_cache_int8);

        nano_vllm::SamplingParams params;
        params.temperature = 0.6f;
        params.top_k = 50;
        params.top_p = 0.9f;
        params.repetition_penalty = 1.1f;
        params.max_tokens = 1024;
        params.ignore_eos = false;

        std::vector<std::string> prompts = {
            "介绍一下你自己",
            "列出100以内的所有质数",
        };

        std::vector<std::string> chat_prompts;
        chat_prompts.reserve(prompts.size());
        for (const auto& p : prompts) {
            chat_prompts.push_back(apply_chat_template(p));
        }

        std::map<int64_t, bool> printed_header;
        auto outputs = engine.generate_stream(
            chat_prompts,
            params,
            [&](int64_t seq_id, int32_t token_id, const std::string& token_text, bool finished) {
                if (!printed_header[seq_id]) {
                    std::cout << "\n[seq " << seq_id << "] ";
                    printed_header[seq_id] = true;
                }
                (void)token_id;
                std::cout << token_text << std::flush;
                if (finished) {
                    std::cout << "\n";
                }
            });

        std::cout << "\n===== Final Outputs =====\n";
        for (size_t i = 0; i < outputs.size() && i < prompts.size(); ++i) {
            std::cout << "\nPrompt: " << prompts[i] << '\n';
            std::cout << "Completion: " << outputs[i].text << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "example failed: " << error.what() << '\n';
        return 1;
    }
}
