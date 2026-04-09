#include "utils/tokenizer.h"

#include <nlohmann/json.hpp>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace nano_vllm {
namespace {

// GPT-2 / Qwen byte-to-unicode mapping.
// Printable bytes map to themselves; control/extended bytes get offset to 256+.
std::unordered_map<uint8_t, std::string> build_byte_encoder() {
    std::unordered_map<uint8_t, std::string> enc;

    // Bytes that map to themselves (printable ASCII + Latin-1 supplement).
    auto add_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            char32_t cp = static_cast<char32_t>(b);
            // Encode code point as UTF-8.
            std::string s;
            if (cp < 0x80) {
                s.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
                s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            enc[static_cast<uint8_t>(b)] = s;
        }
    };
    add_range(0x21, 0x7E);  // ! to ~
    add_range(0xA1, 0xAC);  // ¡ to ¬
    add_range(0xAE, 0xFF);  // ® to ÿ

    // Remaining bytes get mapped to 256+N.
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (enc.find(static_cast<uint8_t>(b)) == enc.end()) {
            char32_t cp = static_cast<char32_t>(256 + n);
            std::string s;
            if (cp < 0x800) {
                s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            enc[static_cast<uint8_t>(b)] = s;
            ++n;
        }
    }
    return enc;
}

std::unordered_map<std::string, uint8_t> build_byte_decoder(
    const std::unordered_map<uint8_t, std::string>& encoder) {
    std::unordered_map<std::string, uint8_t> dec;
    for (auto& [b, s] : encoder) {
        dec[s] = b;
    }
    return dec;
}

// PCRE2 pre-tokenizer pattern (GPT-4 / Qwen2 style).
static const char* kPreTokenizerPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

// Decode a single UTF-8 code point from a string, returning the codepoint
// and advancing the index. Returns (codepoint, bytes_consumed).
std::pair<char32_t, int> decode_utf8_codepoint(const std::string& s, size_t pos) {
    auto byte = static_cast<uint8_t>(s[pos]);
    if (byte < 0x80) return {byte, 1};
    if ((byte & 0xE0) == 0xC0) {
        if (pos + 1 >= s.size()) return {0xFFFD, 1};
        char32_t cp = (byte & 0x1F) << 6;
        cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F);
        return {cp, 2};
    }
    if ((byte & 0xF0) == 0xE0) {
        if (pos + 2 >= s.size()) return {0xFFFD, 1};
        char32_t cp = (byte & 0x0F) << 12;
        cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 6;
        cp |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F);
        return {cp, 3};
    }
    if ((byte & 0xF8) == 0xF0) {
        if (pos + 3 >= s.size()) return {0xFFFD, 1};
        char32_t cp = (byte & 0x07) << 18;
        cp |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 12;
        cp |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F) << 6;
        cp |= (static_cast<uint8_t>(s[pos + 3]) & 0x3F);
        return {cp, 4};
    }
    return {0xFFFD, 1};
}

} // namespace

// ---- Construction / Destruction -------------------------------------------

Tokenizer::Tokenizer(const std::string& model_dir) {
    if (!fs::is_directory(model_dir)) {
        throw std::runtime_error("Tokenizer: model directory does not exist: " + model_dir);
    }

    // Build byte encoder/decoder.
    byte_encoder_ = build_byte_encoder();
    byte_decoder_ = build_byte_decoder(byte_encoder_);

    // Load tokenizer.json.
    std::string tok_path = (fs::path(model_dir) / "tokenizer.json").string();
    if (!fs::exists(tok_path)) {
        throw std::runtime_error("Tokenizer: tokenizer.json not found in " + model_dir);
    }

    std::ifstream ifs(tok_path);
    json tok_json = json::parse(ifs);

    // Vocab from model section.
    auto& model_section = tok_json.at("model");
    auto& vocab_json = model_section.at("vocab");
    for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        int32_t id = it.value().get<int32_t>();
        token_to_id_[it.key()] = id;
        id_to_token_[id] = it.key();
    }

    // Merges: can be array of strings "a b" or array of [a, b] pairs.
    auto& merges_json = model_section.at("merges");
    int rank = 0;
    for (auto& m : merges_json) {
        std::string merge_key;
        if (m.is_string()) {
            merge_key = m.get<std::string>();
        } else if (m.is_array() && m.size() == 2) {
            merge_key = m[0].get<std::string>() + " " + m[1].get<std::string>();
        } else {
            continue;
        }
        merge_ranks_[merge_key] = rank++;
    }

    // Added tokens (special tokens).
    if (tok_json.contains("added_tokens")) {
        for (auto& t : tok_json["added_tokens"]) {
            std::string content = t.at("content").get<std::string>();
            int32_t id = t.at("id").get<int32_t>();
            special_tokens_[content] = id;
            token_to_id_[content] = id;
            id_to_token_[id] = content;
        }
    }

    // Detect tokenizer style: SentencePiece (LLaMA) vs GPT-2 byte-level (Qwen).
    // SentencePiece tokenizers have a "decoder" section with "ByteFallback" or
    // "Replace" sub-decoders that handle ▁ and <0xNN> tokens.
    use_byte_level_encoding_ = true;
    if (tok_json.contains("decoder")) {
        auto& decoder = tok_json["decoder"];
        std::string dtype = decoder.value("type", "");
        if (dtype == "Sequence" || dtype == "ByteFallback" || dtype == "Replace") {
            // Check for SentencePiece-style sub-decoders.
            if (decoder.contains("decoders")) {
                for (auto& d : decoder["decoders"]) {
                    std::string dt = d.value("type", "");
                    if (dt == "ByteFallback" || dt == "Replace") {
                        use_byte_level_encoding_ = false;
                        break;
                    }
                }
            }
            if (dtype == "ByteFallback") {
                use_byte_level_encoding_ = false;
            }
        }
    }

    // Find eos token id.
    // Check generation_config.json first, then config.json.
    std::string gen_config_path = (fs::path(model_dir) / "generation_config.json").string();
    if (fs::exists(gen_config_path)) {
        std::ifstream gifs(gen_config_path);
        json gc = json::parse(gifs);
        if (gc.contains("eos_token_id")) {
            auto& eos_val = gc["eos_token_id"];
            if (eos_val.is_array() && !eos_val.empty()) {
                eos_token_id_ = eos_val[0].get<int>();
            } else if (eos_val.is_number()) {
                eos_token_id_ = eos_val.get<int>();
            }
        }
    }
    if (eos_token_id_ < 0) {
        // Fallback: look for <|im_end|> or <|endoftext|>.
        if (auto it = special_tokens_.find("<|im_end|>"); it != special_tokens_.end()) {
            eos_token_id_ = it->second;
        } else if (auto it2 = special_tokens_.find("<|endoftext|>"); it2 != special_tokens_.end()) {
            eos_token_id_ = it2->second;
        }
    }

    // Compile PCRE2 regex for pre-tokenizer.
    int errorcode;
    PCRE2_SIZE erroroffset;
    pcre2_re_ = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(kPreTokenizerPattern),
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errorcode,
        &erroroffset,
        nullptr);
    if (!pcre2_re_) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
        throw std::runtime_error(
            std::string("Tokenizer: PCRE2 compile error: ") +
            reinterpret_cast<const char*>(buffer));
    }
    pcre2_match_data_ = pcre2_match_data_create_from_pattern(
        static_cast<pcre2_code*>(pcre2_re_), nullptr);
}

Tokenizer::~Tokenizer() {
    if (pcre2_match_data_) {
        pcre2_match_data_free(static_cast<pcre2_match_data*>(pcre2_match_data_));
    }
    if (pcre2_re_) {
        pcre2_code_free(static_cast<pcre2_code*>(pcre2_re_));
    }
}

Tokenizer::Tokenizer(Tokenizer&& other) noexcept
    : token_to_id_(std::move(other.token_to_id_)),
      id_to_token_(std::move(other.id_to_token_)),
      special_tokens_(std::move(other.special_tokens_)),
      merge_ranks_(std::move(other.merge_ranks_)),
      byte_encoder_(std::move(other.byte_encoder_)),
      byte_decoder_(std::move(other.byte_decoder_)),
      eos_token_id_(other.eos_token_id_),
      pcre2_re_(other.pcre2_re_),
      pcre2_match_data_(other.pcre2_match_data_) {
    other.pcre2_re_ = nullptr;
    other.pcre2_match_data_ = nullptr;
}

Tokenizer& Tokenizer::operator=(Tokenizer&& other) noexcept {
    if (this != &other) {
        if (pcre2_match_data_) {
            pcre2_match_data_free(static_cast<pcre2_match_data*>(pcre2_match_data_));
        }
        if (pcre2_re_) {
            pcre2_code_free(static_cast<pcre2_code*>(pcre2_re_));
        }
        token_to_id_ = std::move(other.token_to_id_);
        id_to_token_ = std::move(other.id_to_token_);
        special_tokens_ = std::move(other.special_tokens_);
        merge_ranks_ = std::move(other.merge_ranks_);
        byte_encoder_ = std::move(other.byte_encoder_);
        byte_decoder_ = std::move(other.byte_decoder_);
        eos_token_id_ = other.eos_token_id_;
        pcre2_re_ = other.pcre2_re_;
        pcre2_match_data_ = other.pcre2_match_data_;
        other.pcre2_re_ = nullptr;
        other.pcre2_match_data_ = nullptr;
    }
    return *this;
}

// ---- Pre-tokenization (PCRE2 regex) ---------------------------------------

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    if (text.empty()) return tokens;

    auto* re = static_cast<pcre2_code*>(pcre2_re_);
    auto* match_data = static_cast<pcre2_match_data*>(pcre2_match_data_);
    auto* subject = reinterpret_cast<PCRE2_SPTR>(text.data());
    PCRE2_SIZE subject_len = text.size();
    PCRE2_SIZE offset = 0;

    while (offset < subject_len) {
        int rc = pcre2_match(re, subject, subject_len, offset, 0, match_data, nullptr);
        if (rc < 0) break; // No more matches.

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE start = ovector[0];
        PCRE2_SIZE end = ovector[1];

        if (end <= start) {
            // Safety: avoid infinite loop on zero-length match.
            offset = end + 1;
            continue;
        }

        tokens.push_back(text.substr(start, end - start));
        offset = end;
    }

    return tokens;
}

// ---- Byte-level encoding/decoding -----------------------------------------

std::string Tokenizer::bytes_to_unicode_str(const std::string& s) const {
    std::string result;
    for (auto c : s) {
        auto it = byte_encoder_.find(static_cast<uint8_t>(c));
        if (it != byte_encoder_.end()) {
            result += it->second;
        }
    }
    return result;
}

std::string Tokenizer::unicode_str_to_bytes(const std::string& s) const {
    std::string result;
    size_t i = 0;
    while (i < s.size()) {
        auto [cp, len] = decode_utf8_codepoint(s, i);
        std::string token_char = s.substr(i, len);
        auto it = byte_decoder_.find(token_char);
        if (it != byte_decoder_.end()) {
            result.push_back(static_cast<char>(it->second));
        } else {
            // Pass through (shouldn't happen with valid tokens).
            result += token_char;
        }
        i += len;
    }
    return result;
}

// ---- BPE ------------------------------------------------------------------

std::vector<std::string> Tokenizer::bpe(const std::string& word) const {
    if (word.empty()) return {};

    // Split word into individual UTF-8 characters.
    std::vector<std::string> symbols;
    {
        size_t i = 0;
        while (i < word.size()) {
            auto [cp, len] = decode_utf8_codepoint(word, i);
            symbols.push_back(word.substr(i, len));
            i += len;
        }
    }

    if (symbols.size() <= 1) return symbols;

    // Iteratively merge the highest-priority pair.
    while (true) {
        // Find the best pair.
        int best_rank = std::numeric_limits<int>::max();
        int best_idx = -1;
        for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
            std::string pair_key = symbols[i] + " " + symbols[i + 1];
            auto it = merge_ranks_.find(pair_key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break; // No more merges possible.

        // Merge at best_idx.
        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);

        if (symbols.size() <= 1) break;
    }

    return symbols;
}

// ---- Encode ---------------------------------------------------------------

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;

    // Check for special tokens at the boundaries.
    // Simple approach: scan for special tokens and split around them.
    struct Segment {
        std::string text;
        bool is_special;
    };
    std::vector<Segment> segments;

    // Build sorted special tokens (longest first for greedy matching).
    std::vector<std::pair<std::string, int32_t>> sorted_specials(
        special_tokens_.begin(), special_tokens_.end());
    std::sort(sorted_specials.begin(), sorted_specials.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    size_t pos = 0;
    while (pos < text.size()) {
        bool found_special = false;
        for (auto& [tok, id] : sorted_specials) {
            if (pos + tok.size() <= text.size() &&
                text.compare(pos, tok.size(), tok) == 0) {
                if (!segments.empty() && !segments.back().is_special &&
                    segments.back().text.empty()) {
                    segments.pop_back();
                }
                segments.push_back({tok, true});
                pos += tok.size();
                found_special = true;
                break;
            }
        }
        if (!found_special) {
            if (segments.empty() || segments.back().is_special) {
                segments.push_back({"", false});
            }
            // Append one UTF-8 character.
            auto [cp, len] = decode_utf8_codepoint(text, pos);
            segments.back().text += text.substr(pos, len);
            pos += len;
        }
    }

    // Process each segment.
    for (auto& seg : segments) {
        if (seg.is_special) {
            auto it = special_tokens_.find(seg.text);
            if (it != special_tokens_.end()) {
                ids.push_back(it->second);
            }
            continue;
        }

        // Pre-tokenize and encode normal text.
        if (use_byte_level_encoding_) {
            // GPT-2 / Qwen style: regex pre-tokenize, byte-encode, BPE.
            auto pre_tokens = pre_tokenize(seg.text);
            for (auto& pt : pre_tokens) {
                std::string encoded = bytes_to_unicode_str(pt);
                auto bpe_tokens = bpe(encoded);
                for (auto& bt : bpe_tokens) {
                    auto it = token_to_id_.find(bt);
                    if (it != token_to_id_.end()) {
                        ids.push_back(it->second);
                    }
                }
            }
        } else {
            // SentencePiece / LLaMA style: replace spaces with ▁, prepend ▁,
            // then run BPE on the whole normalized text without pre-tokenization.
            std::string normalized;
            // Prepend ▁ and replace all spaces with ▁.
            normalized += "\xe2\x96\x81"; // ▁ in UTF-8
            for (char c : seg.text) {
                if (c == ' ') {
                    normalized += "\xe2\x96\x81";
                } else {
                    normalized += c;
                }
            }
            auto bpe_tokens = bpe(normalized);
            for (auto& bt : bpe_tokens) {
                auto it = token_to_id_.find(bt);
                if (it != token_to_id_.end()) {
                    ids.push_back(it->second);
                } else {
                    // Byte fallback: encode unknown characters as <0xNN>.
                    for (uint8_t byte : bt) {
                        char hex_buf[8];
                        std::snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", byte);
                        auto it2 = token_to_id_.find(hex_buf);
                        if (it2 != token_to_id_.end()) {
                            ids.push_back(it2->second);
                        }
                    }
                }
            }
        }
    }

    return ids;
}

// ---- Decode ---------------------------------------------------------------

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string byte_level_str;
    for (auto id : token_ids) {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            byte_level_str += it->second;
        }
    }

    if (use_byte_level_encoding_) {
        // GPT-2 / Qwen style: decode byte-level unicode back to raw bytes.
        return unicode_str_to_bytes(byte_level_str);
    }

    // SentencePiece / LLaMA style:
    //   ▁ (U+2581) -> space
    //   <0xNN>     -> raw byte 0xNN
    std::string result;
    size_t i = 0;
    while (i < byte_level_str.size()) {
        // Check for ▁ (UTF-8: 0xE2 0x96 0x81)
        if (i + 2 < byte_level_str.size() &&
            static_cast<uint8_t>(byte_level_str[i]) == 0xE2 &&
            static_cast<uint8_t>(byte_level_str[i + 1]) == 0x96 &&
            static_cast<uint8_t>(byte_level_str[i + 2]) == 0x81) {
            result += ' ';
            i += 3;
            continue;
        }

        // Check for <0xNN> byte fallback pattern.
        if (i + 5 < byte_level_str.size() &&
            byte_level_str[i] == '<' &&
            byte_level_str[i + 1] == '0' &&
            byte_level_str[i + 2] == 'x' &&
            byte_level_str[i + 5] == '>') {
            char hi = byte_level_str[i + 3];
            char lo = byte_level_str[i + 4];
            auto hex_val = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            int h = hex_val(hi);
            int l = hex_val(lo);
            if (h >= 0 && l >= 0) {
                result += static_cast<char>((h << 4) | l);
                i += 6;
                continue;
            }
        }

        result += byte_level_str[i];
        ++i;
    }

    return result;
}

} // namespace nano_vllm
