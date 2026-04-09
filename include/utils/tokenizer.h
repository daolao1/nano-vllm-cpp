#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nano_vllm {

/// Byte-level BPE tokenizer compatible with Qwen2/GPT-style tokenizer.json.
class Tokenizer {
public:
    /// Load from a HuggingFace model directory (reads tokenizer.json).
    explicit Tokenizer(const std::string& model_dir);

    /// Encode text to token IDs.
    std::vector<int32_t> encode(const std::string& text) const;

    /// Decode token IDs to text.
    std::string decode(const std::vector<int32_t>& token_ids) const;

    int eos_token_id() const { return eos_token_id_; }
    int vocab_size() const { return static_cast<int>(token_to_id_.size()); }

private:
    /// Pre-tokenize text using the GPT-4/Qwen regex pattern (via PCRE2).
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    /// Apply byte-level encoding: convert a UTF-8 string to GPT-2 unicode tokens.
    std::string bytes_to_unicode_str(const std::string& s) const;

    /// Convert GPT-2 unicode token string back to raw bytes.
    std::string unicode_str_to_bytes(const std::string& s) const;

    /// Apply BPE to a single pre-tokenized word (already byte-encoded).
    std::vector<std::string> bpe(const std::string& word) const;

    // vocab: token string -> id
    std::unordered_map<std::string, int32_t> token_to_id_;
    // reverse vocab: id -> token string
    std::unordered_map<int32_t, std::string> id_to_token_;
    // added/special tokens: string -> id (checked before BPE)
    std::unordered_map<std::string, int32_t> special_tokens_;
    // BPE merge ranks: "token1 token2" -> priority (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;
    // byte -> unicode char mapping (GPT-2 style)
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    // unicode char -> byte reverse mapping
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    int eos_token_id_ = -1;

    // True for GPT-2/Qwen style byte encoding; false for SentencePiece/LLaMA style.
    bool use_byte_level_encoding_ = true;

    // Compiled PCRE2 regex (opaque, cleaned up in destructor)
    void* pcre2_re_ = nullptr;
    void* pcre2_match_data_ = nullptr;

public:
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&& other) noexcept;
    Tokenizer& operator=(Tokenizer&& other) noexcept;
};

} // namespace nano_vllm
