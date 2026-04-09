#include "core/sampling_params.h"
#include "engine/llm_engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void print_progress(int done, int total, int gen_tokens,
                           double elapsed_s) {
    const int bar_width = 30;
    int filled = total > 0 ? done * bar_width / total : 0;
    double tps = elapsed_s > 0 ? gen_tokens / elapsed_s : 0;
    std::fprintf(stderr, "\r\033[36m[");
    for (int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] %d/%d reqs | %d tokens | %.1f tok/s\033[0m",
                 done, total, gen_tokens, tps);
    std::fflush(stderr);
}

int main(int argc, char** argv) {
    try {
        const std::string model_dir =
            argc > 1 ? argv[1] : "/home/zzy/huggingface/Qwen3-0.6B";
        const int num_seqs = argc > 2 ? std::atoi(argv[2]) : 256;
        const int max_input_len = argc > 3 ? std::atoi(argv[3]) : 1024;
        const int max_output_len = argc > 4 ? std::atoi(argv[4]) : 1024;

        std::fprintf(stderr, "model:      %s\n", model_dir.c_str());
        std::fprintf(stderr, "num_seqs:   %d\n", num_seqs);
        std::fprintf(stderr, "input_len:  100-%d\n", max_input_len);
        std::fprintf(stderr, "output_len: 100-%d\n", max_output_len);

        nano_vllm::LLMEngine engine(model_dir,
                                    /*max_model_len=*/4096,
                                    /*max_num_seqs=*/512,
                                    /*max_num_batched_tokens=*/16384,
                                    /*gpu_memory_utilization=*/0.9f,
                                    /*tensor_parallel_size=*/1,
                                    /*enforce_eager=*/true,
                                    /*kvcache_block_size=*/nano_vllm::Sequence::BLOCK_SIZE);

        // Warmup.
        {
            nano_vllm::SamplingParams wp;
            wp.max_tokens = 1;
            wp.ignore_eos = true;
            std::vector<int32_t> warmup_ids = {0};
            engine.add_request(warmup_ids, wp);
            while (!engine.is_finished()) engine.step();
        }

        // Generate random prompts and sampling params (same seed as Python bench).
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> token_dist(0, 10000);
        std::uniform_int_distribution<int> input_len_dist(100, max_input_len);
        std::uniform_int_distribution<int> output_len_dist(100, max_output_len);

        int total_output_tokens = 0;
        std::vector<nano_vllm::SamplingParams> params_list;

        for (int i = 0; i < num_seqs; ++i) {
            int input_len = input_len_dist(rng);
            std::vector<int32_t> prompt(input_len);
            for (auto& t : prompt) t = token_dist(rng);

            nano_vllm::SamplingParams sp;
            sp.temperature = 0.6f;
            sp.ignore_eos = true;
            sp.max_tokens = output_len_dist(rng);
            total_output_tokens += sp.max_tokens;

            engine.add_request(prompt, sp);
            params_list.push_back(sp);
        }

        std::fprintf(stderr, "total output tokens: %d\n\n", total_output_tokens);

        // Benchmark.
        int done = 0, gen_tokens = 0;
        auto t0 = std::chrono::steady_clock::now();
        print_progress(done, num_seqs, gen_tokens, 0);

        while (!engine.is_finished()) {
            auto [finished, num_tokens] = engine.step();

            if (num_tokens < 0)
                gen_tokens += -num_tokens;

            done += static_cast<int>(finished.size());

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            print_progress(done, num_seqs, gen_tokens, elapsed);
        }

        auto t1 = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double>(t1 - t0).count();
        double throughput = total_output_tokens / total_time;

        std::fprintf(stderr, "\n\n");
        std::printf("Total: %d tok, Time: %.2f s, Throughput: %.2f tok/s\n",
                    total_output_tokens, total_time, throughput);

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench failed: " << error.what() << '\n';
        return 1;
    }
}
