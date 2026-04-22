#include "core/sampling_params.h"
#include "engine/llm_engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const std::string model_dir =
            argc > 1 ? argv[1] : "/home/zzy/huggingface/Qwen3-0.6B";

        constexpr int num_seqs = 32;
        constexpr int min_input_len = 100;
        constexpr int max_input_len = 512;
        constexpr int min_output_len = 100;
        constexpr int max_output_len = 512;

        std::mt19937 rng(0);

        const bool enforce_eager = argc > 2 && std::string(argv[2]) == "--eager";

        nano_vllm::LLMEngine engine(model_dir,
                                    /*max_model_len=*/4096,
                                    /*max_num_seqs=*/32,
                                    /*max_num_batched_tokens=*/8192,
                                    /*gpu_memory_utilization=*/0.9f,
                                    /*tensor_parallel_size=*/1,
                                    /*enforce_eager=*/enforce_eager,
                                    /*kvcache_block_size=*/nano_vllm::Sequence::BLOCK_SIZE);

        // Generate random prompt token ids and sampling params.
        int total_prompt_tokens = 0;
        int total_expected_output = 0;
        std::uniform_int_distribution<int> input_len_dist(min_input_len, max_input_len);
        std::uniform_int_distribution<int> output_len_dist(min_output_len, max_output_len);
        std::uniform_int_distribution<int> token_dist(0, 10000);

        for (int i = 0; i < num_seqs; ++i) {
            const int input_len = input_len_dist(rng);
            const int output_len = output_len_dist(rng);
            std::vector<int32_t> prompt_ids(input_len);
            for (int j = 0; j < input_len; ++j) {
                prompt_ids[j] = token_dist(rng);
            }
            total_prompt_tokens += input_len;
            total_expected_output += output_len;

            nano_vllm::SamplingParams params;
            params.temperature = 0.6f;
            params.ignore_eos = true;
            params.max_tokens = output_len;
            engine.add_request(prompt_ids, params);
        }

        // Warmup.
        // (engine warms up internally)

        // Run with timing.
        auto t0 = std::chrono::steady_clock::now();
        std::map<int64_t, std::vector<int32_t>> collected;
        int prefill_steps = 0, decode_steps = 0;
        double prefill_time = 0, decode_time = 0;
        int total_decode_tokens = 0;

        while (!engine.is_finished()) {
            auto step_start = std::chrono::steady_clock::now();
            auto [finished, num_tokens] = engine.step();
            auto step_end = std::chrono::steady_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
            if (num_tokens >= 0) { prefill_steps++; prefill_time += step_ms; }
            else { decode_steps++; decode_time += step_ms; total_decode_tokens += -num_tokens; }
            for (auto& [seq_id, tids] : finished) {
                collected[seq_id] = std::move(tids);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Count actual generated tokens.
        int actual_gen = 0;
        for (auto& [_, tids] : collected) actual_gen += static_cast<int>(tids.size());

        std::printf("\n%s\n", std::string(60, '=').c_str());
        std::printf("nano-vllm-cpp (C++) Benchmark [%s]\n", enforce_eager ? "eager" : "cuda_graph");
        std::printf("%s\n", std::string(60, '=').c_str());
        std::printf("Seqs:             %d\n", num_seqs);
        std::printf("Prompt tokens:    %d\n", total_prompt_tokens);
        std::printf("Generated tokens: %d\n", actual_gen);
        std::printf("Total time:       %.2fs\n", elapsed);
        std::printf("Throughput:       %.2f tok/s\n", actual_gen / elapsed);
        std::printf("Prefill:          %d steps, %.1fms total (%.1fms/step)\n",
                    prefill_steps, prefill_time, prefill_steps > 0 ? prefill_time / prefill_steps : 0);
        std::printf("Decode:           %d steps, %.1fms total (%.1fms/step, avg batch=%.1f)\n",
                    decode_steps, decode_time, decode_steps > 0 ? decode_time / decode_steps : 0,
                    decode_steps > 0 ? static_cast<double>(total_decode_tokens) / decode_steps : 0);
        std::printf("%s\n", std::string(60, '=').c_str());

        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "bench_e2e failed: %s\n", error.what());
        return 1;
    }
}
