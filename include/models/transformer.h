#pragma once

#include "core/config.h"
#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embed_head.h"
#include "layers/layernorm.h"
#include "layers/linear.h"
#include "layers/rotary_embedding.h"
#include "utils/cuda_allocator.h"
#include "utils/loader.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nano_vllm {

class TransformerAttention {
public:
    TransformerAttention(const HFConfig& config,
                   ScalarType dtype,
                   Device device,
                   DeviceAllocator& allocator,
                   int tp_size,
                   int rank,
                   int block_size,
                   std::string tp_prefix = "");

    Tensor forward(const Tensor& positions,
                   const Tensor& hidden_states,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr);

    void set_kv_cache(const Tensor& k_cache, const Tensor& v_cache);
    void set_kv_cache_int8(const Tensor& k_cache, const Tensor& v_cache,
                           const Tensor& k_scale, const Tensor& v_scale);
    void register_parameters(ParameterRegistry& registry, const std::string& prefix);

    QKVParallelLinear& qkv_proj() { return qkv_proj_; }
    const QKVParallelLinear& qkv_proj() const { return qkv_proj_; }
    RowParallelLinear& o_proj() { return o_proj_; }
    const RowParallelLinear& o_proj() const { return o_proj_; }
    RotaryEmbedding& rotary_emb() { return rotary_emb_; }
    const RotaryEmbedding& rotary_emb() const { return rotary_emb_; }
    Attention& attn() { return attn_; }
    const Attention& attn() const { return attn_; }
    bool has_q_norm() const { return q_norm_.has_value(); }
    bool has_k_norm() const { return k_norm_.has_value(); }
    RMSNorm& q_norm() { return *q_norm_; }
    const RMSNorm& q_norm() const { return *q_norm_; }
    RMSNorm& k_norm() { return *k_norm_; }
    const RMSNorm& k_norm() const { return *k_norm_; }

private:
    int hidden_size_ = 0;
    int total_num_heads_ = 0;
    int num_heads_ = 0;
    int total_num_kv_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int q_size_ = 0;
    int kv_size_ = 0;
    bool qkv_bias_ = false;
    QKVParallelLinear qkv_proj_;
    RowParallelLinear o_proj_;
    RotaryEmbedding rotary_emb_;
    Attention attn_;
    std::optional<RMSNorm> q_norm_;
    std::optional<RMSNorm> k_norm_;
};

class TransformerMLP {
public:
    TransformerMLP(const HFConfig& config,
             ScalarType dtype,
             Device device,
             DeviceAllocator& allocator,
             int tp_size,
             int rank,
             std::string tp_prefix = "");

    Tensor forward(const Tensor& input,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void register_parameters(ParameterRegistry& registry, const std::string& prefix);

    MergedColumnParallelLinear& gate_up_proj() { return gate_up_proj_; }
    const MergedColumnParallelLinear& gate_up_proj() const { return gate_up_proj_; }
    RowParallelLinear& down_proj() { return down_proj_; }
    const RowParallelLinear& down_proj() const { return down_proj_; }

private:
    MergedColumnParallelLinear gate_up_proj_;
    RowParallelLinear down_proj_;
    SiluAndMul act_fn_;
};

class TransformerDecoderLayer {
public:
    TransformerDecoderLayer(const HFConfig& config,
                      ScalarType dtype,
                      Device device,
                      DeviceAllocator& allocator,
                      int tp_size,
                      int rank,
                      int block_size,
                      std::string tp_prefix = "");

    std::pair<Tensor, Tensor> forward(const Tensor& positions,
                                      const Tensor& hidden_states,
                                      const Tensor& residual,
                                      DeviceAllocator& allocator,
                                      cudaStream_t stream = nullptr);

    void set_kv_cache(const Tensor& k_cache, const Tensor& v_cache);
    void set_kv_cache_int8(const Tensor& k_cache, const Tensor& v_cache,
                           const Tensor& k_scale, const Tensor& v_scale);
    void register_parameters(ParameterRegistry& registry, const std::string& prefix);

    RMSNorm& input_layernorm() { return input_layernorm_; }
    const RMSNorm& input_layernorm() const { return input_layernorm_; }
    TransformerAttention& self_attn() { return self_attn_; }
    const TransformerAttention& self_attn() const { return self_attn_; }
    RMSNorm& post_attention_layernorm() { return post_attention_layernorm_; }
    const RMSNorm& post_attention_layernorm() const { return post_attention_layernorm_; }
    TransformerMLP& mlp() { return mlp_; }
    const TransformerMLP& mlp() const { return mlp_; }

private:
    RMSNorm input_layernorm_;
    TransformerAttention self_attn_;
    RMSNorm post_attention_layernorm_;
    TransformerMLP mlp_;
};

class TransformerModel {
public:
    TransformerModel(const Config& config,
               ScalarType dtype,
               Device device,
               DeviceAllocator& allocator,
               int rank = 0);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& positions,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr);

    void set_layer_kv_cache(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache);
    void set_layer_kv_cache_int8(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache,
                                 const Tensor& k_scale, const Tensor& v_scale);
    void register_parameters(ParameterRegistry& registry, const std::string& prefix = "model");

    VocabParallelEmbedding& embed_tokens() { return embed_tokens_; }
    const VocabParallelEmbedding& embed_tokens() const { return embed_tokens_; }
    std::vector<TransformerDecoderLayer>& layers() { return layers_; }
    const std::vector<TransformerDecoderLayer>& layers() const { return layers_; }
    RMSNorm& norm() { return norm_; }
    const RMSNorm& norm() const { return norm_; }

private:
    Config config_;
    VocabParallelEmbedding embed_tokens_;
    std::vector<TransformerDecoderLayer> layers_;
    RMSNorm norm_;
};

class TransformerForCausalLM {
public:
    TransformerForCausalLM(const Config& config,
                     DeviceAllocator& allocator,
                     ScalarType dtype = ScalarType::kFloat32,
                     Device device = Device{DeviceType::kCUDA, 0},
                     int rank = 0);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& positions,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr);
    Tensor compute_logits(const Tensor& hidden_states,
                          DeviceAllocator& allocator,
                          cudaStream_t stream = nullptr) const;

    void set_layer_kv_cache(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache);
    void set_layer_kv_cache_int8(size_t layer_index, const Tensor& k_cache, const Tensor& v_cache,
                                 const Tensor& k_scale, const Tensor& v_scale);
    void register_parameters(ParameterRegistry& registry);
    void load_weights(const std::string& path,
                      DeviceAllocator& allocator,
                      cudaStream_t stream = nullptr);

    static const std::vector<PackedModuleMapping>& packed_modules_mapping();

    TransformerModel& model() { return model_; }
    const TransformerModel& model() const { return model_; }
    ParallelLMHead& lm_head() { return lm_head_; }
    const ParallelLMHead& lm_head() const { return lm_head_; }

private:
    bool tie_word_embeddings_ = false;
    TransformerModel model_;
    ParallelLMHead lm_head_;
};

} // namespace nano_vllm