#include "core/sampling_params.h"
#include "engine/llm_engine.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const std::string model_dir = argc > 1 ? argv[1] : "/home/zzy/huggingface/Qwen3-0.6B";
        std::string prompt = argc > 2 ? argv[2] : "Hello, world!";

        nano_vllm::LLMEngine engine(model_dir,
                                    /*max_model_len=*/512,
                                    /*max_num_seqs=*/4,
                                    /*max_num_batched_tokens=*/512,
                                    /*gpu_memory_utilization=*/0.6f,
                                    /*tensor_parallel_size=*/1,
                                    /*enforce_eager=*/true,
                                    /*kvcache_block_size=*/nano_vllm::Sequence::BLOCK_SIZE);

        nano_vllm::SamplingParams params;
        params.temperature = 0.0f;
        params.max_tokens = 32;
        params.ignore_eos = false;

        std::cout << "prompt: " << prompt << '\n';
        auto outputs = engine.generate({prompt}, params);
        for (auto& out : outputs) {
            std::cout << "output: " << out.text << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "run_qwen3_once failed: " << error.what() << '\n';
        return 1;
    }
}