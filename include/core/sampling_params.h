#pragma once

namespace nano_vllm {

struct SamplingParams {
    float temperature = 1.0f;
    int top_k = -1;            // <= 0 means disabled
    float top_p = 1.0f;        // 1.0 means disabled
    float repetition_penalty = 1.0f; // 1.0 means disabled
    int max_tokens = 64;
    bool ignore_eos = false;
};

} // namespace nano_vllm
