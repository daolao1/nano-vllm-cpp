#include "models/transformer.h"

#include "layers/kernel_ops.h"
#include "utils/context.h"
#include "utils/cuda_common.h"

#include <stdexcept>

namespace nano_vllm {
namespace {

std::string join_name(const std::string& prefix, const std::string& suffix) {
    return prefix.empty() ? suffix : prefix + "." + suffix;
}

Tensor slice_last_dim_contiguous(const Tensor& input,
                                 int64_t start,
                                 int64_t size,
                                 DeviceAllocator& allocator,
                                 cudaStream_t stream) {
    if (!input.defined() || input.sizes().empty()) {
        throw std::invalid_argument("slice_last_dim_contiguous expects a defined tensor with rank >= 1");
    }
    if (input.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument("slice_last_dim_contiguous currently expects a CUDA tensor");
    }
    const int64_t full = input.sizes().back();
    if (start < 0 || size < 0 || start + size > full) {
        throw std::out_of_range("slice_last_dim_contiguous range is out of bounds");
    }
    if (start == 0 && size == full && input.is_contiguous()) {
        return input;
    }

    std::vector<int64_t> output_sizes = input.sizes();
    output_sizes.back() = size;
    Tensor output = Tensor::empty(output_sizes, input.dtype(), input.device(), allocator);

    if (output.nbytes() == 0) {
        return output;
    }
    const int64_t outer = static_cast<int64_t>(input.numel() / static_cast<size_t>(full));
    const size_t element_bytes = input.element_size();
    const size_t input_pitch = static_cast<size_t>(full) * element_bytes;
    const size_t output_pitch = static_cast<size_t>(size) * element_bytes;
    const auto* src = static_cast<const std::byte*>(input.data()) + static_cast<size_t>(start) * element_bytes;
    throw_if_cuda_error(cudaMemcpy2DAsync(output.data(),
                                          output_pitch,
                                          src,
                                          input_pitch,
                                          output_pitch,
                                          static_cast<size_t>(outer),
                                          cudaMemcpyDeviceToDevice,
                                          stream),
                         "cudaMemcpy2DAsync(slice_last_dim_contiguous)");
    return output;
}

int resolve_head_dim(const HFConfig& config) {
    return config.head_dim > 0 ? config.head_dim : config.hidden_size / config.num_attention_heads;
}

} // namespace

TransformerAttention::TransformerAttention(const HFConfig& config,
                               ScalarType dtype,
                               Device device,
                               DeviceAllocator& allocator,
                               int tp_size,
                               int rank,
                               int block_size,
                               std::string tp_prefix)
    : hidden_size_(config.hidden_size),
      total_num_heads_(config.num_attention_heads),
      num_heads_(config.num_attention_heads / tp_size),
      total_num_kv_heads_(config.num_key_value_heads),
      num_kv_heads_(config.num_key_value_heads / tp_size),
      head_dim_(resolve_head_dim(config)),
      q_size_(num_heads_ * head_dim_),
      kv_size_(num_kv_heads_ * head_dim_),
      qkv_bias_(config.attention_bias),
      qkv_proj_(config.hidden_size,
                head_dim_,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.attention_bias,
                dtype,
                device,
                allocator,
                tp_size,
                rank),
      o_proj_(config.num_attention_heads * head_dim_,
              config.hidden_size,
              false,
              dtype,
              device,
              allocator,
              tp_size,
              rank,
              join_name(tp_prefix, "o_proj")),
      rotary_emb_(head_dim_,
                  head_dim_,
                  config.max_position_embeddings,
                  config.rope_theta,
                  device,
                  allocator),
            attn_(num_heads_, num_kv_heads_, head_dim_, block_size) {
        if (config.use_qk_norm) {
        q_norm_.emplace(head_dim_, config.rms_norm_eps, dtype, device, allocator);
        k_norm_.emplace(head_dim_, config.rms_norm_eps, dtype, device, allocator);
    }
}

Tensor TransformerAttention::forward(const Tensor& positions,
                               const Tensor& hidden_states,
                               DeviceAllocator& allocator,
                               cudaStream_t stream) {
    Tensor qkv = qkv_proj_.forward(hidden_states, allocator, stream);
    const int64_t rows = qkv.sizes()[0];

    // Fused path for decode (non-bias models only):
    // Combines split_qkv + q_norm + k_norm + rotary + store_kvcache into 1 kernel.
    // Cannot use fused path with int8 cache (fused kernel writes fp16 directly)
    // or float32 models (fused kernel only supports bf16/fp16).
    const Context& ctx = get_context();
    const bool fused_eligible = !ctx.is_prefill && !attn_.is_int8() &&
                                q_norm_.has_value() && k_norm_.has_value() &&
                                ctx.slot_mapping.defined() &&
                                (qkv.dtype() == ScalarType::kBFloat16 || qkv.dtype() == ScalarType::kFloat16);
    if (fused_eligible) {
        Tensor q = Tensor::empty({rows, num_heads_, head_dim_}, qkv.dtype(), qkv.device(), allocator);
        fused_split_norm_rope_store(
            qkv, positions,
            q_norm_->weight(), k_norm_->weight(), q_norm_->eps(),
            rotary_emb_.cos_sin_cache(),
            ctx.slot_mapping,
            attn_.k_cache(), attn_.v_cache(),
            num_heads_, num_kv_heads_, head_dim_,
            q, stream);
        Tensor output = attn_.forward_decode_flash_only(q, allocator, stream);
        return o_proj_.forward(output.reshape({rows, q_size_}), allocator, stream);
    }

    // Prefill / bias path: original separate kernels
    Tensor q = Tensor::empty({rows, num_heads_, head_dim_}, qkv.dtype(), qkv.device(), allocator);
    Tensor k = Tensor::empty({rows, num_kv_heads_, head_dim_}, qkv.dtype(), qkv.device(), allocator);
    Tensor v = Tensor::empty({rows, num_kv_heads_, head_dim_}, qkv.dtype(), qkv.device(), allocator);
    split_qkv(qkv, q_size_, kv_size_, q, k, v, stream);
    if (q_norm_.has_value() && k_norm_.has_value()) {
        q = q_norm_->forward(q, allocator, stream);
        k = k_norm_->forward(k, allocator, stream);
    }
    rotary_emb_.forward(positions, q, k, stream);
    Tensor output = attn_.forward(q, k, v, allocator, stream);
    return o_proj_.forward(output.reshape({rows, q_size_}), allocator, stream);
}

void TransformerAttention::set_kv_cache(const Tensor& k_cache, const Tensor& v_cache) {
    attn_.set_kv_cache(k_cache, v_cache);
}

void TransformerAttention::set_kv_cache_int8(const Tensor& k_cache, const Tensor& v_cache,
                                              const Tensor& k_scale, const Tensor& v_scale) {
    attn_.set_kv_cache_int8(k_cache, v_cache, k_scale, v_scale);
}

void TransformerAttention::register_parameters(ParameterRegistry& registry, const std::string& prefix) {
    registry.register_parameter(join_name(prefix, "qkv_proj.weight"),
                                [this](const Tensor& source,
                                       std::optional<int> shard_id,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    if (shard_id.has_value()) {
                                        qkv_proj_.weight_loader(source, *shard_id, allocator, stream);
                                    } else {
                                        qkv_proj_.weight_loader(source, allocator, stream);
                                    }
                                });
    if (qkv_proj_.bias().defined()) {
        registry.register_parameter(join_name(prefix, "qkv_proj.bias"),
                                    [this](const Tensor& source,
                                           std::optional<int> shard_id,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) {
                                        if (shard_id.has_value()) {
                                            qkv_proj_.bias_loader(source, *shard_id, allocator, stream);
                                        } else {
                                            qkv_proj_.bias_loader(source, allocator, stream);
                                        }
                                    });
    }

    registry.register_parameter(join_name(prefix, "o_proj.weight"),
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    o_proj_.weight_loader(source, allocator, stream);
                                });
    if (o_proj_.bias().defined()) {
        registry.register_parameter(join_name(prefix, "o_proj.bias"),
                                    [this](const Tensor& source,
                                           std::optional<int>,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) {
                                        o_proj_.bias_loader(source, allocator, stream);
                                    });
    }

    if (q_norm_.has_value()) {
        registry.register_parameter(join_name(prefix, "q_norm.weight"),
                                    [this](const Tensor& source,
                                           std::optional<int>,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) {
                                        q_norm_->weight_loader(source, allocator, stream);
                                    });
        registry.register_parameter(join_name(prefix, "k_norm.weight"),
                                    [this](const Tensor& source,
                                           std::optional<int>,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) {
                                        k_norm_->weight_loader(source, allocator, stream);
                                    });
    }
}

TransformerMLP::TransformerMLP(const HFConfig& config,
                   ScalarType dtype,
                   Device device,
                   DeviceAllocator& allocator,
                   int tp_size,
                   int rank,
                   std::string tp_prefix)
    : gate_up_proj_(config.hidden_size,
                    {config.intermediate_size, config.intermediate_size},
                    false,
                    dtype,
                    device,
                    allocator,
                    tp_size,
                    rank),
      down_proj_(config.intermediate_size,
                 config.hidden_size,
                 false,
                 dtype,
                 device,
                 allocator,
                 tp_size,
                 rank,
                 join_name(tp_prefix, "down_proj")) {
    if (config.hidden_act != "silu") {
        throw std::invalid_argument("TransformerMLP currently supports hidden_act=silu only");
    }
}

Tensor TransformerMLP::forward(const Tensor& input,
                         DeviceAllocator& allocator,
                         cudaStream_t stream) const {
    Tensor gate_up = gate_up_proj_.forward(input, allocator, stream);
    Tensor activated = act_fn_.forward(gate_up, allocator, stream);
    return down_proj_.forward(activated, allocator, stream);
}

void TransformerMLP::register_parameters(ParameterRegistry& registry, const std::string& prefix) {
    registry.register_parameter(join_name(prefix, "gate_up_proj.weight"),
                                [this](const Tensor& source,
                                       std::optional<int> shard_id,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    if (shard_id.has_value()) {
                                        gate_up_proj_.weight_loader(source, *shard_id, allocator, stream);
                                    } else {
                                        gate_up_proj_.weight_loader(source, allocator, stream);
                                    }
                                });
    if (gate_up_proj_.bias().defined()) {
        registry.register_parameter(join_name(prefix, "gate_up_proj.bias"),
                                    [this](const Tensor& source,
                                           std::optional<int> shard_id,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) {
                                        if (shard_id.has_value()) {
                                            gate_up_proj_.bias_loader(source, *shard_id, allocator, stream);
                                        } else {
                                            gate_up_proj_.bias_loader(source, allocator, stream);
                                        }
                                    });
    }
    registry.register_parameter(join_name(prefix, "down_proj.weight"),
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    down_proj_.weight_loader(source, allocator, stream);
                                });
    if (down_proj_.bias().defined()) {
        registry.register_parameter(join_name(prefix, "down_proj.bias"),
                                    [this](const Tensor& source,
                                           std::optional<int>,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) {
                                        down_proj_.bias_loader(source, allocator, stream);
                                    });
    }
}

TransformerDecoderLayer::TransformerDecoderLayer(const HFConfig& config,
                                     ScalarType dtype,
                                     Device device,
                                     DeviceAllocator& allocator,
                                     int tp_size,
                                     int rank,
                                                                         int block_size,
                                                                         std::string tp_prefix)
    : input_layernorm_(config.hidden_size, config.rms_norm_eps, dtype, device, allocator),
            self_attn_(config,
                                 dtype,
                                 device,
                                 allocator,
                                 tp_size,
                                 rank,
                                 block_size,
                                 join_name(tp_prefix, "self_attn")),
      post_attention_layernorm_(config.hidden_size, config.rms_norm_eps, dtype, device, allocator),
            mlp_(config,
                     dtype,
                     device,
                     allocator,
                     tp_size,
                     rank,
                     join_name(tp_prefix, "mlp")) {}

std::pair<Tensor, Tensor> TransformerDecoderLayer::forward(const Tensor& positions,
                                                     const Tensor& hidden_states,
                                                     const Tensor& residual,
                                                     DeviceAllocator& allocator,
                                                     cudaStream_t stream) {
    Tensor hidden = hidden_states;
    Tensor next_residual = residual;
    if (!next_residual.defined()) {
        next_residual = hidden_states;
        hidden = input_layernorm_.forward(hidden_states, allocator, stream);
    } else {
        auto norm_pair = input_layernorm_.forward(hidden_states, next_residual, allocator, stream);
        hidden = norm_pair.first;
        next_residual = norm_pair.second;
    }
    hidden = self_attn_.forward(positions, hidden, allocator, stream);
    auto post_attn = post_attention_layernorm_.forward(hidden, next_residual, allocator, stream);
    hidden = mlp_.forward(post_attn.first, allocator, stream);
    return {hidden, post_attn.second};
}

void TransformerDecoderLayer::set_kv_cache(const Tensor& k_cache, const Tensor& v_cache) {
    self_attn_.set_kv_cache(k_cache, v_cache);
}

void TransformerDecoderLayer::set_kv_cache_int8(const Tensor& k_cache, const Tensor& v_cache,
                                                 const Tensor& k_scale, const Tensor& v_scale) {
    self_attn_.set_kv_cache_int8(k_cache, v_cache, k_scale, v_scale);
}

void TransformerDecoderLayer::register_parameters(ParameterRegistry& registry, const std::string& prefix) {
    registry.register_parameter(join_name(prefix, "input_layernorm.weight"),
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    input_layernorm_.weight_loader(source, allocator, stream);
                                });
    self_attn_.register_parameters(registry, join_name(prefix, "self_attn"));
    registry.register_parameter(join_name(prefix, "post_attention_layernorm.weight"),
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    post_attention_layernorm_.weight_loader(source, allocator, stream);
                                });
    mlp_.register_parameters(registry, join_name(prefix, "mlp"));
}

TransformerModel::TransformerModel(const Config& config,
                       ScalarType dtype,
                       Device device,
                       DeviceAllocator& allocator,
                       int rank)
    : config_(config),
      embed_tokens_(config.hf_config.vocab_size,
                    config.hf_config.hidden_size,
                    dtype,
                    device,
                    allocator,
                    config.tensor_parallel_size,
                    rank,
                    "model.embed_tokens"),
      norm_(config.hf_config.hidden_size,
            config.hf_config.rms_norm_eps,
            dtype,
            device,
            allocator) {
    layers_.reserve(static_cast<size_t>(config.hf_config.num_hidden_layers));
    for (int index = 0; index < config.hf_config.num_hidden_layers; ++index) {
        layers_.emplace_back(config.hf_config,
                             dtype,
                             device,
                             allocator,
                             config.tensor_parallel_size,
                             rank,
                             config.kvcache_block_size,
                             join_name("model", "layers." + std::to_string(index)));
    }
}

Tensor TransformerModel::forward(const Tensor& input_ids,
                           const Tensor& positions,
                           DeviceAllocator& allocator,
                           cudaStream_t stream) {
    Tensor hidden_states = embed_tokens_.forward(input_ids, allocator, stream);
    Tensor residual;
    for (TransformerDecoderLayer& layer : layers_) {
        auto layer_output = layer.forward(positions, hidden_states, residual, allocator, stream);
        hidden_states = layer_output.first;
        residual = layer_output.second;
    }
    if (residual.defined()) {
        return norm_.forward(hidden_states, residual, allocator, stream).first;
    }
    return norm_.forward(hidden_states, allocator, stream);
}

void TransformerModel::set_layer_kv_cache(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache) {
    if (layer_index >= layers_.size()) {
        throw std::out_of_range("layer_index is out of range");
    }
    layers_[layer_index].set_kv_cache(k_cache, v_cache);
}

void TransformerModel::set_layer_kv_cache_int8(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache,
                                                const Tensor& k_scale, const Tensor& v_scale) {
    if (layer_index >= layers_.size()) {
        throw std::out_of_range("layer_index is out of range");
    }
    layers_[layer_index].set_kv_cache_int8(k_cache, v_cache, k_scale, v_scale);
}

void TransformerModel::register_parameters(ParameterRegistry& registry, const std::string& prefix) {
    registry.register_parameter(join_name(prefix, "embed_tokens.weight"),
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    embed_tokens_.weight_loader(source, allocator, stream);
                                });
    for (size_t index = 0; index < layers_.size(); ++index) {
        layers_[index].register_parameters(registry, join_name(prefix, "layers." + std::to_string(index)));
    }
    registry.register_parameter(join_name(prefix, "norm.weight"),
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    norm_.weight_loader(source, allocator, stream);
                                });
}

TransformerForCausalLM::TransformerForCausalLM(const Config& config,
                                   DeviceAllocator& allocator,
                                   ScalarType dtype,
                                   Device device,
                                   int rank)
    : tie_word_embeddings_(config.hf_config.tie_word_embeddings),
      model_(config, dtype, device, allocator, rank),
      lm_head_(config.hf_config.vocab_size,
               config.hf_config.hidden_size,
               dtype,
               device,
               allocator,
               config.tensor_parallel_size,
               rank,
               "lm_head") {
    if (tie_word_embeddings_) {
        lm_head_.set_weight(model_.embed_tokens().weight());
    }
}

Tensor TransformerForCausalLM::forward(const Tensor& input_ids,
                                 const Tensor& positions,
                                 DeviceAllocator& allocator,
                                 cudaStream_t stream) {
    return model_.forward(input_ids, positions, allocator, stream);
}

Tensor TransformerForCausalLM::compute_logits(const Tensor& hidden_states,
                                        DeviceAllocator& allocator,
                                        cudaStream_t stream) const {
    return lm_head_.forward(hidden_states, allocator, stream);
}

void TransformerForCausalLM::set_layer_kv_cache(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache) {
    model_.set_layer_kv_cache(layer_index, k_cache, v_cache);
}

void TransformerForCausalLM::set_layer_kv_cache_int8(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache,
                                                      const Tensor& k_scale, const Tensor& v_scale) {
    model_.set_layer_kv_cache_int8(layer_index, k_cache, v_cache, k_scale, v_scale);
}

void TransformerForCausalLM::register_parameters(ParameterRegistry& registry) {
    model_.register_parameters(registry);
    registry.register_parameter("lm_head.weight",
                                [this](const Tensor& source,
                                       std::optional<int>,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
                                    lm_head_.weight_loader(source, allocator, stream);
                                });
}

void TransformerForCausalLM::load_weights(const std::string& path,
                                    DeviceAllocator& allocator,
                                    cudaStream_t stream) {
    ParameterRegistry registry;
    register_parameters(registry);
    registry.load(path, packed_modules_mapping(), allocator, stream);
}

const std::vector<PackedModuleMapping>& TransformerForCausalLM::packed_modules_mapping() {
    static const std::vector<PackedModuleMapping> mapping = {
        {"q_proj", "qkv_proj", 0},
        {"k_proj", "qkv_proj", 1},
        {"v_proj", "qkv_proj", 2},
        {"gate_proj", "gate_up_proj", 0},
        {"up_proj", "gate_up_proj", 1},
    };
    return mapping;
}

} // namespace nano_vllm