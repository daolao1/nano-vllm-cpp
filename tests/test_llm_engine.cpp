#include "engine/llm_engine.h"
#include "utils/tokenizer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace nano_vllm;

// ---------------------------------------------------------------------------
// Tokenizer unit tests (CPU-only, no CUDA needed).
// ---------------------------------------------------------------------------

static const char* test_model_dir() {
    // Default to Qwen3-0.6B; override with env var.
    const char* dir = std::getenv("NANO_VLLM_MODEL_DIR");
    return dir ? dir : "/home/zzy/huggingface/Qwen3-0.6B";
}

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string dir = test_model_dir();
        if (!std::filesystem::is_directory(dir)) {
            GTEST_SKIP() << "Model directory not found: " << dir;
        }
        tok_ = std::make_unique<Tokenizer>(dir);
    }

    std::unique_ptr<Tokenizer> tok_;
};

TEST_F(TokenizerTest, VocabSizeIsReasonable) {
    EXPECT_GT(tok_->vocab_size(), 100000);
}

TEST_F(TokenizerTest, EosTokenIdIsValid) {
    EXPECT_EQ(tok_->eos_token_id(), 151645);
}

TEST_F(TokenizerTest, EncodeDecodeEnglish) {
    std::string text = "Hello, world!";
    auto ids = tok_->encode(text);
    EXPECT_FALSE(ids.empty());
    std::string decoded = tok_->decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, EncodeDecodeChinese) {
    std::string text = "你好世界";
    auto ids = tok_->encode(text);
    EXPECT_FALSE(ids.empty());
    std::string decoded = tok_->decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, EncodeDecodeRoundTrip) {
    std::string text = "The quick brown fox jumps over the lazy dog. 1234";
    auto ids = tok_->encode(text);
    EXPECT_FALSE(ids.empty());
    std::string decoded = tok_->decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, SpecialTokenEncode) {
    // <|im_start|> should encode to a single token.
    auto ids = tok_->encode("<|im_start|>");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 151644);
}

TEST_F(TokenizerTest, EncodeMatchesPythonReference) {
    // Verified against Python: AutoTokenizer.from_pretrained("Qwen3-0.6B")
    {
        auto ids = tok_->encode("Hello, world!");
        std::vector<int32_t> expected = {9707, 11, 1879, 0};
        EXPECT_EQ(ids, expected);
    }
    {
        auto ids = tok_->encode("你好世界");
        std::vector<int32_t> expected = {108386, 99489};
        EXPECT_EQ(ids, expected);
    }
    {
        auto ids = tok_->encode("The quick brown fox");
        std::vector<int32_t> expected = {785, 3974, 13876, 38835};
        EXPECT_EQ(ids, expected);
    }
}

TEST_F(TokenizerTest, EmptyInput) {
    auto ids = tok_->encode("");
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(tok_->decode({}), "");
}

// ---------------------------------------------------------------------------
// LLMEngine tests (require CUDA + model weights).
// ---------------------------------------------------------------------------

class LLMEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_dir_ = test_model_dir();
        if (!std::filesystem::is_directory(model_dir_)) {
            GTEST_SKIP() << "Model directory not found: " << model_dir_;
        }
        // Quick CUDA check.
        int device_count = 0;
        auto err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    std::string model_dir_;
};

TEST_F(LLMEngineTest, GenerateFromTokenIds) {
    LLMEngine engine(model_dir_,
                     /*max_model_len=*/512,
                     /*max_num_seqs=*/1,
                     /*max_num_batched_tokens=*/512,
                     /*gpu_memory_utilization=*/0.5f,
                     /*tensor_parallel_size=*/1,
                     /*enforce_eager=*/true);

    SamplingParams params;
    params.temperature = 0.01f;
    params.max_tokens = 4;

    // Use a simple prompt: just bos token.
    std::vector<std::vector<int32_t>> prompts = {{151644}};
    auto outputs = engine.generate(prompts, params);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_FALSE(outputs[0].token_ids.empty());
    EXPECT_LE(static_cast<int>(outputs[0].token_ids.size()), params.max_tokens);
}

TEST_F(LLMEngineTest, GenerateFromString) {
    LLMEngine engine(model_dir_,
                     /*max_model_len=*/512,
                     /*max_num_seqs=*/2,
                     /*max_num_batched_tokens=*/512,
                     /*gpu_memory_utilization=*/0.5f,
                     /*tensor_parallel_size=*/1,
                     /*enforce_eager=*/true);

    SamplingParams params;
    params.temperature = 0.01f;
    params.max_tokens = 8;

    std::vector<std::string> prompts = {"Hello", "你好"};
    auto outputs = engine.generate(prompts, params);
    ASSERT_EQ(outputs.size(), 2u);
    for (auto& out : outputs) {
        EXPECT_FALSE(out.text.empty());
        EXPECT_FALSE(out.token_ids.empty());
    }
}
