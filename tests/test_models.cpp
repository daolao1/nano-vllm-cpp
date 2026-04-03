#include "core/config.h"
#include "models/transformer.h"
#include "utils/context.h"
#include "utils/cuda_allocator.h"
#include "utils/tensor_parallel.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace nano_vllm {
namespace {

bool has_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

int cuda_device_count() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        return 0;
    }
    return device_count;
}

void throw_if_cuda_runtime_error(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void set_current_device(int device_index) {
    throw_if_cuda_runtime_error(cudaSetDevice(device_index), "cudaSetDevice");
}

Tensor make_device_float_tensor(const std::vector<int64_t>& sizes,
                                const std::vector<float>& host_values,
                                CudaAllocator& allocator,
                                Device device = Device{DeviceType::kCUDA, 0}) {
    set_current_device(device.index);
    Tensor tensor = Tensor::empty(sizes, ScalarType::kFloat32, device, allocator);
    allocator.copy_to_device_async(tensor.data(), host_values.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return tensor;
}

Tensor make_device_int_tensor(const std::vector<int64_t>& sizes,
                              const std::vector<int32_t>& host_values,
                              CudaAllocator& allocator,
                              Device device = Device{DeviceType::kCUDA, 0}) {
    set_current_device(device.index);
    Tensor tensor = Tensor::empty(sizes, ScalarType::kInt32, device, allocator);
    allocator.copy_to_device_async(tensor.data(), host_values.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return tensor;
}

std::vector<float> copy_device_floats(const Tensor& tensor, CudaAllocator& allocator) {
    set_current_device(tensor.device().index);
    std::vector<float> host_values(tensor.numel(), 0.0f);
    allocator.copy_to_host_async(host_values.data(), tensor.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return host_values;
}

Tensor make_cpu_float_tensor(const std::vector<int64_t>& sizes,
                             const std::vector<float>& host_values) {
    const size_t bytes = host_values.size() * sizeof(float);
    void* buf = std::malloc(bytes);
    std::memcpy(buf, host_values.data(), bytes);
    auto memory = Memory::make_owned(buf, bytes, Device{DeviceType::kCPU, 0},
                                     [](void* p) { std::free(p); });
    return Tensor::from_memory(memory, sizes, ScalarType::kFloat32);
}

Config make_tiny_qwen3_config(bool tie_word_embeddings = false) {
    Config config{};
    config.tensor_parallel_size = 1;
    config.kvcache_block_size = 2;
    config.hf_config.hidden_size = 2;
    config.hf_config.num_attention_heads = 1;
    config.hf_config.num_key_value_heads = 1;
    config.hf_config.num_hidden_layers = 1;
    config.hf_config.intermediate_size = 2;
    config.hf_config.vocab_size = 3;
    config.hf_config.max_position_embeddings = 8;
    config.hf_config.head_dim = 2;
    config.hf_config.rms_norm_eps = 1e-6f;
    config.hf_config.rope_theta = 10000.0f;
    config.hf_config.tie_word_embeddings = tie_word_embeddings;
    config.hf_config.attention_bias = false;
    config.hf_config.hidden_act = "silu";
    config.hf_config.use_qk_norm = true;
    return config;
}

Config make_tiny_tp_qwen3_config(int tensor_parallel_size) {
    Config config = make_tiny_qwen3_config(false);
    config.tensor_parallel_size = tensor_parallel_size;
    config.hf_config.hidden_size = 4;
    config.hf_config.num_attention_heads = 2;
    config.hf_config.num_key_value_heads = 2;
    config.hf_config.intermediate_size = 4;
    config.hf_config.vocab_size = 4;
    config.hf_config.head_dim = 2;
    return config;
}

class ScopedTensorParallelCommunicator {
public:
    explicit ScopedTensorParallelCommunicator(std::shared_ptr<TensorParallelCommunicator> communicator) {
        set_tensor_parallel_communicator(std::move(communicator));
    }

    ~ScopedTensorParallelCommunicator() {
        reset_tensor_parallel_communicator();
    }
};

class ScopedContextReset {
public:
    ~ScopedContextReset() {
        reset_context();
    }
};

struct TransformerTPTestWeights {
    std::vector<float> embedding;
    std::vector<float> q_proj;
    std::vector<float> k_proj;
    std::vector<float> v_proj;
    std::vector<float> o_proj;
    std::vector<float> gate_proj;
    std::vector<float> up_proj;
    std::vector<float> down_proj;
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;
    std::vector<float> model_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    std::vector<float> lm_head;
};

TransformerTPTestWeights make_transformer_tp_test_weights() {
    return {
        {0.50f, -1.00f, 1.50f, 0.00f,
         1.00f, 0.25f, -0.75f, 2.00f,
         -1.00f, 0.50f, 0.00f, 1.00f,
         0.25f, 1.50f, -1.25f, 0.75f},
        {1.00f, 0.00f, 0.20f, -0.10f,
         0.00f, 1.00f, -0.30f, 0.40f,
         0.50f, -0.25f, 1.00f, 0.00f,
         -0.20f, 0.30f, 0.00f, 1.00f},
        {0.60f, 0.00f, 0.10f, -0.20f,
         0.00f, 0.70f, 0.25f, 0.15f,
         0.30f, -0.10f, 0.80f, 0.00f,
         0.05f, 0.20f, 0.00f, 0.90f},
        {0.50f, -0.50f, 0.10f, 0.00f,
         1.00f, 0.25f, -0.20f, 0.15f,
         0.00f, 0.30f, -1.00f, 0.50f,
         -0.10f, 0.00f, 0.75f, 1.00f},
        {1.00f, 0.00f, 0.50f, -0.50f,
         0.00f, 1.00f, -0.25f, 0.75f,
         -0.50f, 0.25f, 1.00f, 0.00f,
         0.75f, -0.50f, 0.00f, 1.00f},
        {0.50f, -0.25f, 0.75f, 0.00f,
         0.00f, 0.50f, -0.50f, 1.00f,
         -0.75f, 0.00f, 0.25f, 0.50f,
         1.00f, -1.00f, 0.00f, 0.25f},
        {1.00f, 0.00f, -0.50f, 0.25f,
         -0.25f, 0.75f, 0.50f, 0.00f,
         0.50f, -0.50f, 1.00f, 0.00f,
         0.00f, 0.25f, -0.75f, 1.00f},
        {0.50f, -1.00f, 0.25f, 0.75f,
         -0.50f, 0.50f, 1.00f, -0.25f,
         1.00f, 0.00f, -0.50f, 0.50f,
         0.25f, 0.75f, 0.50f, -1.00f},
        {1.00f, 0.90f, 1.10f, 0.95f},
        {0.80f, 1.20f, 1.00f, 0.85f},
        {1.05f, 0.95f, 1.10f, 0.90f},
        {1.00f, 0.80f},
        {0.90f, 1.10f},
        {1.00f, 0.00f, 0.50f, -0.50f,
         -0.25f, 1.00f, 0.00f, 0.75f,
         0.50f, -0.50f, 1.00f, 0.25f,
         0.00f, 0.25f, -0.75f, 1.00f},
    };
}

void load_qwen3_tensor_parallel_test_weights(TransformerForCausalLM& model,
                                             const TransformerTPTestWeights& weights,
                                             CudaAllocator& allocator) {
    Tensor embedding = make_cpu_float_tensor({4, 4}, weights.embedding);
    Tensor q_proj = make_cpu_float_tensor({4, 4}, weights.q_proj);
    Tensor k_proj = make_cpu_float_tensor({4, 4}, weights.k_proj);
    Tensor v_proj = make_cpu_float_tensor({4, 4}, weights.v_proj);
    Tensor o_proj = make_cpu_float_tensor({4, 4}, weights.o_proj);
    Tensor gate_proj = make_cpu_float_tensor({4, 4}, weights.gate_proj);
    Tensor up_proj = make_cpu_float_tensor({4, 4}, weights.up_proj);
    Tensor down_proj = make_cpu_float_tensor({4, 4}, weights.down_proj);
    Tensor input_layernorm = make_cpu_float_tensor({4}, weights.input_layernorm);
    Tensor post_attention_layernorm = make_cpu_float_tensor({4}, weights.post_attention_layernorm);
    Tensor model_norm = make_cpu_float_tensor({4}, weights.model_norm);
    Tensor q_norm = make_cpu_float_tensor({2}, weights.q_norm);
    Tensor k_norm = make_cpu_float_tensor({2}, weights.k_norm);
    Tensor lm_head = make_cpu_float_tensor({4, 4}, weights.lm_head);

    model.model().embed_tokens().weight_loader(embedding, allocator);
    model.model().layers()[0].self_attn().qkv_proj().weight_loader(q_proj, 0, allocator);
    model.model().layers()[0].self_attn().qkv_proj().weight_loader(k_proj, 1, allocator);
    model.model().layers()[0].self_attn().qkv_proj().weight_loader(v_proj, 2, allocator);
    model.model().layers()[0].self_attn().o_proj().weight_loader(o_proj, allocator);
    model.model().layers()[0].mlp().gate_up_proj().weight_loader(gate_proj, 0, allocator);
    model.model().layers()[0].mlp().gate_up_proj().weight_loader(up_proj, 1, allocator);
    model.model().layers()[0].mlp().down_proj().weight_loader(down_proj, allocator);
    model.model().layers()[0].input_layernorm().weight_loader(input_layernorm, allocator);
    model.model().layers()[0].post_attention_layernorm().weight_loader(post_attention_layernorm, allocator);
    model.model().layers()[0].self_attn().q_norm().weight_loader(q_norm, allocator);
    model.model().layers()[0].self_attn().k_norm().weight_loader(k_norm, allocator);
    model.model().norm().weight_loader(model_norm, allocator);
    model.lm_head().weight_loader(lm_head, allocator);
    allocator.synchronize_stream(nullptr);
}

struct TensorParallelModelOutputs {
    Tensor embedding_rank0;
    Tensor embedding_rank1;
    Tensor hidden_rank0;
    Tensor hidden_rank1;
    Tensor logits_rank0;
    Tensor logits_rank1;
};

std::pair<std::pair<Tensor, Tensor>, std::pair<Tensor, Tensor>> run_tensor_parallel_decoder_layer_in_lockstep(
    TransformerDecoderLayer& layer_rank0,
    TransformerDecoderLayer& layer_rank1,
    const Tensor& positions,
    const Tensor& hidden_rank0,
    const Tensor& hidden_rank1,
    const Tensor& residual_rank0,
    const Tensor& residual_rank1,
    CudaAllocator& allocator) {
    Tensor normalized_rank0;
    Tensor next_residual_rank0;
    if (!residual_rank0.defined()) {
        next_residual_rank0 = hidden_rank0;
        normalized_rank0 = layer_rank0.input_layernorm().forward(hidden_rank0, allocator);
    } else {
        const auto norm_pair_rank0 = layer_rank0.input_layernorm().forward(hidden_rank0, residual_rank0, allocator);
        normalized_rank0 = norm_pair_rank0.first;
        next_residual_rank0 = norm_pair_rank0.second;
    }

    Tensor normalized_rank1;
    Tensor next_residual_rank1;
    if (!residual_rank1.defined()) {
        next_residual_rank1 = hidden_rank1;
        normalized_rank1 = layer_rank1.input_layernorm().forward(hidden_rank1, allocator);
    } else {
        const auto norm_pair_rank1 = layer_rank1.input_layernorm().forward(hidden_rank1, residual_rank1, allocator);
        normalized_rank1 = norm_pair_rank1.first;
        next_residual_rank1 = norm_pair_rank1.second;
    }

    Tensor attn_rank0 = layer_rank0.self_attn().forward(positions, normalized_rank0, allocator);
    Tensor attn_rank1 = layer_rank1.self_attn().forward(positions, normalized_rank1, allocator);

    const auto post_attn_rank0 = layer_rank0.post_attention_layernorm().forward(attn_rank0, next_residual_rank0, allocator);
    const auto post_attn_rank1 = layer_rank1.post_attention_layernorm().forward(attn_rank1, next_residual_rank1, allocator);

    Tensor mlp_rank0 = layer_rank0.mlp().forward(post_attn_rank0.first, allocator);
    Tensor mlp_rank1 = layer_rank1.mlp().forward(post_attn_rank1.first, allocator);
    return {{mlp_rank0, post_attn_rank0.second}, {mlp_rank1, post_attn_rank1.second}};
}

TensorParallelModelOutputs run_tensor_parallel_model_in_lockstep(TransformerForCausalLM& model_rank0,
                                                                 TransformerForCausalLM& model_rank1,
                                                                 const Tensor& input_ids,
                                                                 const Tensor& positions,
                                                                 CudaAllocator& allocator) {
    Tensor embedding_rank0 = model_rank0.model().embed_tokens().forward(input_ids, allocator);
    Tensor embedding_rank1 = model_rank1.model().embed_tokens().forward(input_ids, allocator);

    Tensor hidden_rank0 = embedding_rank0;
    Tensor hidden_rank1 = embedding_rank1;
    Tensor residual_rank0;
    Tensor residual_rank1;
    for (size_t index = 0; index < model_rank0.model().layers().size(); ++index) {
        const auto outputs = run_tensor_parallel_decoder_layer_in_lockstep(model_rank0.model().layers()[index],
                                                                           model_rank1.model().layers()[index],
                                                                           positions,
                                                                           hidden_rank0,
                                                                           hidden_rank1,
                                                                           residual_rank0,
                                                                           residual_rank1,
                                                                           allocator);
        hidden_rank0 = outputs.first.first;
        residual_rank0 = outputs.first.second;
        hidden_rank1 = outputs.second.first;
        residual_rank1 = outputs.second.second;
    }

    if (residual_rank0.defined()) {
        hidden_rank0 = model_rank0.model().norm().forward(hidden_rank0, residual_rank0, allocator).first;
    } else {
        hidden_rank0 = model_rank0.model().norm().forward(hidden_rank0, allocator);
    }
    if (residual_rank1.defined()) {
        hidden_rank1 = model_rank1.model().norm().forward(hidden_rank1, residual_rank1, allocator).first;
    } else {
        hidden_rank1 = model_rank1.model().norm().forward(hidden_rank1, allocator);
    }

    Tensor logits_rank0 = model_rank0.lm_head().forward(hidden_rank0, allocator);
    Tensor logits_rank1 = model_rank1.lm_head().forward(hidden_rank1, allocator);
    return {embedding_rank0, embedding_rank1, hidden_rank0, hidden_rank1, logits_rank0, logits_rank1};
}

void copy_values_to_tensor(Tensor& tensor, const std::vector<float>& values, CudaAllocator& allocator) {
    set_current_device(tensor.device().index);
    allocator.copy_to_device_async(tensor.data(), values.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
}

struct DirectForwardResult {
    Tensor hidden;
    Tensor logits;
    std::string error;
};

void run_model_forward_direct(TransformerForCausalLM& model,
                              const Tensor& input_ids,
                              const Tensor& positions,
                              const Tensor& cu_q,
                              const Tensor& cu_k,
                              CudaAllocator& allocator,
                              DirectForwardResult* result) {
    try {
        set_current_device(input_ids.device().index);
        set_context(true, cu_q, cu_k, 1, 1, Tensor(), Tensor(), Tensor());
        ScopedContextReset context_reset;
        result->hidden = model.forward(input_ids, positions, allocator);
        result->logits = model.compute_logits(result->hidden, allocator);
        allocator.synchronize_stream(nullptr);
    } catch (const std::exception& error) {
        result->error = error.what();
    } catch (...) {
        result->error = "unknown exception";
    }
}

TEST(ModelTest, TransformerForCausalLmForwardAndLogitsMatchReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const Config config = make_tiny_qwen3_config();
    TransformerForCausalLM model(config, allocator);

    copy_values_to_tensor(model.model().embed_tokens().weight(), {1.0f, 2.0f, 0.0f, 1.0f, -1.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.model().layers()[0].self_attn().qkv_proj().weight(),
                          {1.0f, 0.0f,
                           0.0f, 1.0f,
                           1.0f, 0.0f,
                           0.0f, 1.0f,
                           1.0f, 0.0f,
                           0.0f, 1.0f},
                          allocator);
    copy_values_to_tensor(model.model().layers()[0].self_attn().o_proj().weight(), {1.0f, 0.0f, 0.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.model().layers()[0].mlp().gate_up_proj().weight(), std::vector<float>(8, 0.0f), allocator);
    copy_values_to_tensor(model.model().layers()[0].mlp().down_proj().weight(), std::vector<float>(4, 0.0f), allocator);
    copy_values_to_tensor(model.model().layers()[0].input_layernorm().weight(), {1.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.model().layers()[0].post_attention_layernorm().weight(), {1.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.model().norm().weight(), {1.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.model().layers()[0].self_attn().q_norm().weight(), {1.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.model().layers()[0].self_attn().k_norm().weight(), {1.0f, 1.0f}, allocator);
    copy_values_to_tensor(model.lm_head().weight(), {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, allocator);

    Tensor input_ids = make_device_int_tensor({1}, {0}, allocator);
    Tensor positions = make_device_int_tensor({1}, {0}, allocator);
    Tensor cu_q = make_device_int_tensor({2}, {0, 1}, allocator);
    Tensor cu_k = make_device_int_tensor({2}, {0, 1}, allocator);

    set_context(true, cu_q, cu_k, 1, 1, Tensor(), Tensor(), Tensor());
    Tensor hidden_states = model.forward(input_ids, positions, allocator);
    Tensor logits = model.compute_logits(hidden_states, allocator);
    reset_context();

    const std::vector<float> hidden_actual = copy_device_floats(hidden_states, allocator);
    const std::vector<float> logits_actual = copy_device_floats(logits, allocator);
    const float inv_rms = 1.0f / std::sqrt((1.0f * 1.0f + 2.0f * 2.0f) / 2.0f + 1e-6f);
    const std::vector<float> hidden_expected = {1.0f * inv_rms, 2.0f * inv_rms};
    const std::vector<float> logits_expected = {
        hidden_expected[0],
        hidden_expected[1],
        hidden_expected[0] + hidden_expected[1],
    };

    for (size_t index = 0; index < hidden_expected.size(); ++index) {
        EXPECT_NEAR(hidden_actual[index], hidden_expected[index], 1e-4f);
    }
    for (size_t index = 0; index < logits_expected.size(); ++index) {
        EXPECT_NEAR(logits_actual[index], logits_expected[index], 1e-4f);
    }
}

TEST(ModelTest, TransformerTensorParallelModelMatchesSingleRankReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const TransformerTPTestWeights weights = make_transformer_tp_test_weights();

    TransformerForCausalLM reference_model(make_tiny_tp_qwen3_config(1), allocator);
    TransformerForCausalLM tp_model_rank0(make_tiny_tp_qwen3_config(2), allocator, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, 0);
    TransformerForCausalLM tp_model_rank1(make_tiny_tp_qwen3_config(2), allocator, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, 1);

    load_qwen3_tensor_parallel_test_weights(reference_model, weights, allocator);
    load_qwen3_tensor_parallel_test_weights(tp_model_rank0, weights, allocator);
    load_qwen3_tensor_parallel_test_weights(tp_model_rank1, weights, allocator);

    Tensor input_ids = make_device_int_tensor({2}, {3, 1}, allocator);
    Tensor positions = make_device_int_tensor({2}, {0, 1}, allocator);
    Tensor cu_q = make_device_int_tensor({2}, {0, 2}, allocator);
    Tensor cu_k = make_device_int_tensor({2}, {0, 2}, allocator);
    Tensor reference_embedding = reference_model.model().embed_tokens().forward(input_ids, allocator);

    set_context(true, cu_q, cu_k, 1, 1, Tensor(), Tensor(), Tensor());
    ScopedContextReset context_reset;

    Tensor reference_hidden = reference_model.forward(input_ids, positions, allocator);
    Tensor reference_logits = reference_model.compute_logits(reference_hidden, allocator);

    ScopedTensorParallelCommunicator communicator(make_in_process_tensor_parallel_communicator());
    const TensorParallelModelOutputs tp_outputs =
        run_tensor_parallel_model_in_lockstep(tp_model_rank0, tp_model_rank1, input_ids, positions, allocator);

    const std::vector<float> embedding_expected = copy_device_floats(reference_embedding, allocator);
    const std::vector<float> embedding_rank0 = copy_device_floats(tp_outputs.embedding_rank0, allocator);
    const std::vector<float> embedding_rank1 = copy_device_floats(tp_outputs.embedding_rank1, allocator);
    for (size_t index = 0; index < embedding_expected.size(); ++index) {
        EXPECT_NEAR(embedding_rank0[index], embedding_expected[index], 1e-5f);
        EXPECT_NEAR(embedding_rank1[index], embedding_expected[index], 1e-5f);
    }

    const std::vector<float> hidden_expected = copy_device_floats(reference_hidden, allocator);
    const std::vector<float> hidden_rank0 = copy_device_floats(tp_outputs.hidden_rank0, allocator);
    const std::vector<float> hidden_rank1 = copy_device_floats(tp_outputs.hidden_rank1, allocator);
    for (size_t index = 0; index < hidden_expected.size(); ++index) {
        EXPECT_NEAR(hidden_rank0[index], hidden_expected[index], 1e-4f);
        EXPECT_NEAR(hidden_rank1[index], hidden_expected[index], 1e-4f);
    }

    EXPECT_TRUE(tp_outputs.logits_rank0.defined());
    EXPECT_FALSE(tp_outputs.logits_rank1.defined());
    const std::vector<float> logits_expected = copy_device_floats(reference_logits, allocator);
    const std::vector<float> logits_rank0 = copy_device_floats(tp_outputs.logits_rank0, allocator);
    for (size_t index = 0; index < logits_expected.size(); ++index) {
        EXPECT_NEAR(logits_rank0[index], logits_expected[index], 1e-4f);
    }
}

TEST(ModelTest, TransformerTensorParallelNcclDirectForwardMatchesSingleRankReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    if (!nccl_tensor_parallel_available()) {
        GTEST_SKIP() << "nano-vllm-cpp was built without NCCL support";
    }
    if (cuda_device_count() < 2) {
        GTEST_SKIP() << "NCCL direct-forward TP test requires at least 2 CUDA devices";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const TransformerTPTestWeights weights = make_transformer_tp_test_weights();
    const Device rank0_device{DeviceType::kCUDA, 0};
    const Device rank1_device{DeviceType::kCUDA, 1};

    set_current_device(rank0_device.index);
    TransformerForCausalLM reference_model(make_tiny_tp_qwen3_config(1), allocator, ScalarType::kFloat32, rank0_device, 0);
    TransformerForCausalLM tp_model_rank0(make_tiny_tp_qwen3_config(2), allocator, ScalarType::kFloat32, rank0_device, 0);
    set_current_device(rank1_device.index);
    TransformerForCausalLM tp_model_rank1(make_tiny_tp_qwen3_config(2), allocator, ScalarType::kFloat32, rank1_device, 1);

    load_qwen3_tensor_parallel_test_weights(reference_model, weights, allocator);
    load_qwen3_tensor_parallel_test_weights(tp_model_rank0, weights, allocator);
    load_qwen3_tensor_parallel_test_weights(tp_model_rank1, weights, allocator);

    Tensor input_ids_rank0 = make_device_int_tensor({2}, {3, 1}, allocator, rank0_device);
    Tensor positions_rank0 = make_device_int_tensor({2}, {0, 1}, allocator, rank0_device);
    Tensor cu_q_rank0 = make_device_int_tensor({2}, {0, 2}, allocator, rank0_device);
    Tensor cu_k_rank0 = make_device_int_tensor({2}, {0, 2}, allocator, rank0_device);
    Tensor input_ids_rank1 = make_device_int_tensor({2}, {3, 1}, allocator, rank1_device);
    Tensor positions_rank1 = make_device_int_tensor({2}, {0, 1}, allocator, rank1_device);
    Tensor cu_q_rank1 = make_device_int_tensor({2}, {0, 2}, allocator, rank1_device);
    Tensor cu_k_rank1 = make_device_int_tensor({2}, {0, 2}, allocator, rank1_device);

    set_context(true, cu_q_rank0, cu_k_rank0, 1, 1, Tensor(), Tensor(), Tensor());
    ScopedContextReset context_reset;
    Tensor reference_hidden = reference_model.forward(input_ids_rank0, positions_rank0, allocator);
    Tensor reference_logits = reference_model.compute_logits(reference_hidden, allocator);

    ScopedTensorParallelCommunicator communicator(make_nccl_tensor_parallel_communicator({0, 1}));
    DirectForwardResult rank0_result;
    DirectForwardResult rank1_result;
    std::thread rank0_thread(run_model_forward_direct,
                             std::ref(tp_model_rank0),
                             std::cref(input_ids_rank0),
                             std::cref(positions_rank0),
                             std::cref(cu_q_rank0),
                             std::cref(cu_k_rank0),
                             std::ref(allocator),
                             &rank0_result);
    std::thread rank1_thread(run_model_forward_direct,
                             std::ref(tp_model_rank1),
                             std::cref(input_ids_rank1),
                             std::cref(positions_rank1),
                             std::cref(cu_q_rank1),
                             std::cref(cu_k_rank1),
                             std::ref(allocator),
                             &rank1_result);
    rank0_thread.join();
    rank1_thread.join();

    ASSERT_TRUE(rank0_result.error.empty()) << rank0_result.error;
    ASSERT_TRUE(rank1_result.error.empty()) << rank1_result.error;
    ASSERT_TRUE(rank0_result.hidden.defined());
    ASSERT_TRUE(rank1_result.hidden.defined());
    ASSERT_TRUE(rank0_result.logits.defined());
    EXPECT_FALSE(rank1_result.logits.defined());

    const std::vector<float> hidden_expected = copy_device_floats(reference_hidden, allocator);
    const std::vector<float> hidden_rank0 = copy_device_floats(rank0_result.hidden, allocator);
    const std::vector<float> hidden_rank1 = copy_device_floats(rank1_result.hidden, allocator);
    for (size_t index = 0; index < hidden_expected.size(); ++index) {
        EXPECT_NEAR(hidden_rank0[index], hidden_expected[index], 1e-4f);
        EXPECT_NEAR(hidden_rank1[index], hidden_expected[index], 1e-4f);
    }

    const std::vector<float> logits_expected = copy_device_floats(reference_logits, allocator);
    const std::vector<float> logits_rank0 = copy_device_floats(rank0_result.logits, allocator);
    for (size_t index = 0; index < logits_expected.size(); ++index) {
        EXPECT_NEAR(logits_rank0[index], logits_expected[index], 1e-4f);
    }
}

TEST(ModelTest, TransformerRegistersExpectedParametersAndPackedMappings) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    TransformerForCausalLM model(make_tiny_qwen3_config(), allocator);

    ParameterRegistry registry;
    model.register_parameters(registry);
    EXPECT_TRUE(registry.contains("model.embed_tokens.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.self_attn.qkv_proj.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.self_attn.o_proj.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.self_attn.q_norm.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.self_attn.k_norm.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.input_layernorm.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.post_attention_layernorm.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.mlp.gate_up_proj.weight"));
    EXPECT_TRUE(registry.contains("model.layers.0.mlp.down_proj.weight"));
    EXPECT_TRUE(registry.contains("model.norm.weight"));
    EXPECT_TRUE(registry.contains("lm_head.weight"));

    const auto& mapping = TransformerForCausalLM::packed_modules_mapping();
    ASSERT_EQ(mapping.size(), 5u);
    EXPECT_EQ(mapping[0].source_name, "q_proj");
    EXPECT_EQ(mapping[0].target_name, "qkv_proj");
    EXPECT_EQ(mapping[0].shard_id, 0);
}

TEST(ModelTest, TransformerTiesEmbeddingAndLmHeadWeightsWhenRequested) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    TransformerForCausalLM model(make_tiny_qwen3_config(true), allocator);
    EXPECT_EQ(model.model().embed_tokens().weight().data(), model.lm_head().weight().data());
}

} // namespace
} // namespace nano_vllm