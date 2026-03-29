#include "core/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace nano_vllm {

namespace {

int load_int_with_aliases(const json& j,
                          const std::string& name,
                          const std::vector<std::string>& aliases,
                          bool required,
                          int default_value = 0) {
    if (j.contains(name)) {
        return j.at(name).get<int>();
    }
    for (const auto& alias : aliases) {
        if (j.contains(alias)) {
            return j.at(alias).get<int>();
        }
    }
    if (required) {
        throw std::invalid_argument("missing required config field: " + name);
    }
    return default_value;
}

std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool infer_use_qk_norm_default(const std::string& model_type, bool attention_bias) {
    if (model_type.empty()) {
        // Keep legacy behavior for hand-crafted configs in unit tests.
        return !attention_bias;
    }

    static const std::unordered_set<std::string> kFamiliesWithoutQkNorm = {
        "llama", "mistral", "mixtral", "gemma", "phi", "phi3", "yi",
    };
    if (kFamiliesWithoutQkNorm.find(model_type) != kFamiliesWithoutQkNorm.end()) {
        return false;
    }

    // Qwen3-family models use q_norm / k_norm by default.
    if (model_type == "qwen3") {
        return true;
    }

    return !attention_bias;
}

} // namespace

HFConfig load_hf_config(const std::string& json_path) {
    if (!fs::exists(json_path)) {
        throw std::runtime_error("config.json not found: " + json_path);
    }

    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open: " + json_path);
    }

    json j = json::parse(ifs);
    HFConfig hf{};

    if (j.contains("model_type")) {
        hf.model_type = to_lower_ascii(j.at("model_type").get<std::string>());
    }
    if (j.contains("architectures") && j.at("architectures").is_array()) {
        hf.architectures = j.at("architectures").get<std::vector<std::string>>();
    }

    hf.hidden_size = load_int_with_aliases(j, "hidden_size", {"n_embd", "d_model"}, true);
    hf.num_attention_heads = load_int_with_aliases(j, "num_attention_heads", {"n_head"}, true);
    hf.num_hidden_layers = load_int_with_aliases(j, "num_hidden_layers", {"n_layer", "num_layers"}, true);
    hf.intermediate_size = load_int_with_aliases(j, "intermediate_size", {"ffn_dim"}, true);
    hf.vocab_size = load_int_with_aliases(j, "vocab_size", {}, true);
    hf.max_position_embeddings = load_int_with_aliases(
        j, "max_position_embeddings", {"n_positions", "seq_length"}, false, 4096);

    if (j.contains("num_key_value_heads")) {
        hf.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    } else if (j.contains("n_kv_heads")) {
        hf.num_key_value_heads = j.at("n_kv_heads").get<int>();
    } else {
        hf.num_key_value_heads = hf.num_attention_heads;
    }

    // 可选字段
    if (j.contains("head_dim")) {
        hf.head_dim = j["head_dim"].get<int>();
    } else {
        hf.head_dim = hf.hidden_size / hf.num_attention_heads;
    }

    if (j.contains("rms_norm_eps")) {
        hf.rms_norm_eps = j["rms_norm_eps"].get<float>();
    }
    if (j.contains("rope_theta")) {
        hf.rope_theta = j["rope_theta"].get<float>();
    }
    if (j.contains("tie_word_embeddings")) {
        hf.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    }
    if (j.contains("attention_bias")) {
        hf.attention_bias = j["attention_bias"].get<bool>();
    }
    if (j.contains("use_qk_norm")) {
        hf.use_qk_norm = j["use_qk_norm"].get<bool>();
    } else if (j.contains("qk_norm")) {
        hf.use_qk_norm = j["qk_norm"].get<bool>();
    } else {
        hf.use_qk_norm = infer_use_qk_norm_default(hf.model_type, hf.attention_bias);
    }
    if (j.contains("hidden_act")) {
        hf.hidden_act = j["hidden_act"].get<std::string>();
    } else if (j.contains("hidden_activation")) {
        hf.hidden_act = j["hidden_activation"].get<std::string>();
    } else if (j.contains("activation_function")) {
        hf.hidden_act = j["activation_function"].get<std::string>();
    }
    if (j.contains("torch_dtype")) {
        hf.torch_dtype = j["torch_dtype"].get<std::string>();
    }
    if (j.contains("eos_token_id")) {
        if (j["eos_token_id"].is_array() && !j["eos_token_id"].empty()) {
            hf.eos_token_id = j["eos_token_id"][0].get<int>();
        } else {
            hf.eos_token_id = j["eos_token_id"].get<int>();
        }
    }
    if (j.contains("bos_token_id")) {
        if (j["bos_token_id"].is_array() && !j["bos_token_id"].empty()) {
            hf.bos_token_id = j["bos_token_id"][0].get<int>();
        } else {
            hf.bos_token_id = j["bos_token_id"].get<int>();
        }
    }

    if (hf.hidden_size <= 0 || hf.num_attention_heads <= 0 || hf.num_hidden_layers <= 0 ||
        hf.intermediate_size <= 0 || hf.vocab_size <= 0 || hf.max_position_embeddings <= 0 ||
        hf.num_key_value_heads <= 0) {
        throw std::invalid_argument("invalid non-positive model dimensions in config: " + json_path);
    }

    return hf;
}

Config Config::from_model_dir(const std::string& model_dir,
                               int max_model_len,
                               int max_num_seqs,
                               int max_num_batched_tokens,
                               float gpu_memory_utilization,
                               int tensor_parallel_size,
                               bool enforce_eager,
                               int kvcache_block_size,
                               bool kv_cache_int8) {
    // 1. 模型目录存在
    if (!fs::is_directory(model_dir)) {
        throw std::runtime_error("Model directory does not exist: " + model_dir);
    }

    // 2. 解析 HF config.json
    std::string config_path = (fs::path(model_dir) / "config.json").string();
    HFConfig hf = load_hf_config(config_path);

    // 3. 参数校验
    if (kvcache_block_size < 16 || (kvcache_block_size & (kvcache_block_size - 1)) != 0) {
        throw std::invalid_argument("kvcache_block_size must be a power of 2 and >= 16, got "
                                     + std::to_string(kvcache_block_size));
    }
    if (tensor_parallel_size < 1 || tensor_parallel_size > 8) {
        throw std::invalid_argument("tensor_parallel_size must be in [1, 8], got "
                                     + std::to_string(tensor_parallel_size));
    }
    if (gpu_memory_utilization <= 0.0f || gpu_memory_utilization > 1.0f) {
        throw std::invalid_argument("gpu_memory_utilization must be in (0, 1]");
    }

    // 4. 调整 max_model_len
    max_model_len = std::min(max_model_len, hf.max_position_embeddings);

    // 5. max_num_batched_tokens >= max_model_len
    max_num_batched_tokens = std::max(max_num_batched_tokens, max_model_len);

    // 6. 组装 Config
    Config cfg{};
    cfg.model                   = model_dir;
    cfg.max_num_batched_tokens  = max_num_batched_tokens;
    cfg.max_num_seqs            = max_num_seqs;
    cfg.max_model_len           = max_model_len;
    cfg.gpu_memory_utilization  = gpu_memory_utilization;
    cfg.tensor_parallel_size    = tensor_parallel_size;
    cfg.enforce_eager           = enforce_eager;
    cfg.kvcache_block_size      = kvcache_block_size;
    cfg.eos_token_id            = hf.eos_token_id;
    cfg.kv_cache_int8           = kv_cache_int8;
    cfg.hf_config               = std::move(hf);

    return cfg;
}

} // namespace nano_vllm
