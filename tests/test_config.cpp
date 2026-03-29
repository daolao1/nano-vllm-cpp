#include "core/config.h"
#include "core/sampling_params.h"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace nano_vllm;

static const char* MODEL_DIR = "/home/zzy/huggingface/Qwen3-0.6B";

// ============ HFConfig 测试 ============

TEST(HFConfigTest, LoadFromJson) {
    std::string path = std::string(MODEL_DIR) + "/config.json";
    HFConfig hf = load_hf_config(path);

    EXPECT_EQ(hf.hidden_size, 1024);
    EXPECT_EQ(hf.num_attention_heads, 16);
    EXPECT_EQ(hf.num_key_value_heads, 8);
    EXPECT_EQ(hf.num_hidden_layers, 28);
    EXPECT_EQ(hf.intermediate_size, 3072);
    EXPECT_EQ(hf.vocab_size, 151936);
    EXPECT_EQ(hf.max_position_embeddings, 40960);
    EXPECT_EQ(hf.head_dim, 128);
    EXPECT_FLOAT_EQ(hf.rms_norm_eps, 1e-6f);
    EXPECT_FLOAT_EQ(hf.rope_theta, 1000000.0f);
    EXPECT_TRUE(hf.tie_word_embeddings);
    EXPECT_FALSE(hf.attention_bias);
    EXPECT_TRUE(hf.use_qk_norm);
    EXPECT_EQ(hf.hidden_act, "silu");
    EXPECT_EQ(hf.torch_dtype, "bfloat16");
    EXPECT_EQ(hf.eos_token_id, 151645);
    EXPECT_EQ(hf.bos_token_id, 151643);
}

TEST(HFConfigTest, LlamaStyleFallbacks) {
    namespace fs = std::filesystem;
    const fs::path temp_path = fs::temp_directory_path() / "nano_vllm_test_llama_config.json";

    {
        std::ofstream ofs(temp_path);
        ofs << R"({
  "model_type": "llama",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "intermediate_size": 11008,
  "vocab_size": 32000,
  "n_positions": 8192,
  "attention_bias": false,
  "eos_token_id": [2]
})";
    }

    HFConfig hf = load_hf_config(temp_path.string());
    fs::remove(temp_path);

    EXPECT_EQ(hf.model_type, "llama");
    EXPECT_EQ(hf.num_key_value_heads, hf.num_attention_heads);
    EXPECT_EQ(hf.max_position_embeddings, 8192);
    EXPECT_FALSE(hf.use_qk_norm);
    EXPECT_EQ(hf.eos_token_id, 2);
}

TEST(HFConfigTest, FileNotFound) {
    EXPECT_THROW(load_hf_config("/nonexistent/config.json"), std::runtime_error);
}

// ============ Config 测试 ============

TEST(ConfigTest, FromModelDir) {
    Config cfg = Config::from_model_dir(MODEL_DIR);

    EXPECT_EQ(cfg.model, MODEL_DIR);
    EXPECT_EQ(cfg.max_model_len, 4096);
    EXPECT_EQ(cfg.max_num_seqs, 512);
    EXPECT_EQ(cfg.max_num_batched_tokens, 16384);
    EXPECT_EQ(cfg.kvcache_block_size, 256);
    EXPECT_EQ(cfg.eos_token_id, 151645);
    EXPECT_EQ(cfg.hf_config.hidden_size, 1024);
}

TEST(ConfigTest, MaxModelLenClamped) {
    Config cfg = Config::from_model_dir(MODEL_DIR, /*max_model_len=*/999999);
    EXPECT_EQ(cfg.max_model_len, 40960);
}

TEST(ConfigTest, MaxBatchedTokensAdjusted) {
    Config cfg = Config::from_model_dir(MODEL_DIR, /*max_model_len=*/4096,
                                         /*max_num_seqs=*/512,
                                         /*max_num_batched_tokens=*/100);
    EXPECT_GE(cfg.max_num_batched_tokens, cfg.max_model_len);
}

TEST(ConfigTest, InvalidModelDir) {
    EXPECT_THROW(Config::from_model_dir("/nonexistent/model"), std::runtime_error);
}

TEST(ConfigTest, InvalidBlockSize) {
    // 100 不是 2 的幂
    EXPECT_THROW(
        Config::from_model_dir(MODEL_DIR, 4096, 512, 16384, 0.9f, 1, false, 100),
        std::invalid_argument);
    // 8 < 16
    EXPECT_THROW(
        Config::from_model_dir(MODEL_DIR, 4096, 512, 16384, 0.9f, 1, false, 8),
        std::invalid_argument);
    // 合法值不应抛异常
    EXPECT_NO_THROW(
        Config::from_model_dir(MODEL_DIR, 4096, 512, 16384, 0.9f, 1, false, 16));
    EXPECT_NO_THROW(
        Config::from_model_dir(MODEL_DIR, 4096, 512, 16384, 0.9f, 1, false, 256));
}

TEST(ConfigTest, InvalidTPSize) {
    EXPECT_THROW(
        Config::from_model_dir(MODEL_DIR, 4096, 512, 16384, 0.9f, 0),
        std::invalid_argument);
    EXPECT_THROW(
        Config::from_model_dir(MODEL_DIR, 4096, 512, 16384, 0.9f, 9),
        std::invalid_argument);
}

// ============ SamplingParams 测试 ============

TEST(SamplingParamsTest, Defaults) {
    SamplingParams sp{};
    EXPECT_FLOAT_EQ(sp.temperature, 1.0f);
    EXPECT_EQ(sp.max_tokens, 64);
    EXPECT_FALSE(sp.ignore_eos);
}
