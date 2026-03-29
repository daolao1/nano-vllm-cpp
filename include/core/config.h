#pragma once

#include <string>
#include <stdexcept>
#include <vector>

namespace nano_vllm {

// 从 HuggingFace config.json 解析的模型配置
struct HFConfig {
    std::string model_type;
    std::vector<std::string> architectures;
    int hidden_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int max_position_embeddings = 0;
    int head_dim = 0;               // 可选，默认 = hidden_size / num_attention_heads
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool use_qk_norm = false;
    std::string hidden_act = "silu";
    std::string torch_dtype = "bfloat16";
    int eos_token_id = -1;
    int bos_token_id = -1;
};

struct Config {
    std::string model;                      // HF 模型目录路径
    int max_num_batched_tokens = 16384;
    int max_num_seqs = 512;
    int max_model_len = 4096;
    float gpu_memory_utilization = 0.9f;
    int tensor_parallel_size = 1;
    bool enforce_eager = false;
    int kvcache_block_size = 256;
    int num_kvcache_blocks = -1;            // 由 ModelRunner 计算
    int eos_token_id = -1;                  // 由 LLMEngine 从 HFConfig 赋值
    bool kv_cache_int8 = false;             // int8 KV cache quantization
    bool enable_chunked_prefill = false;    // 分块 prefill，允许 prefill+decode 交错
    int max_num_prefill_tokens = 512;       // 每步最多处理的 prefill token 数

    HFConfig hf_config;

    // 从模型目录加载并校验
    static Config from_model_dir(const std::string& model_dir,
                                  int max_model_len = 4096,
                                  int max_num_seqs = 512,
                                  int max_num_batched_tokens = 16384,
                                  float gpu_memory_utilization = 0.9f,
                                  int tensor_parallel_size = 1,
                                  bool enforce_eager = false,
                                  int kvcache_block_size = 256,
                                  bool kv_cache_int8 = false);
};

// 独立函数：从 config.json 文件路径解析 HFConfig
HFConfig load_hf_config(const std::string& json_path);

} // namespace nano_vllm
