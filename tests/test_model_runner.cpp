#include "engine/model_runner.h"

#include "engine/scheduler.h"
#include "utils/context.h"
#include "utils/cuda_allocator.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace nano_vllm {
namespace {

bool has_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

void throw_if_cuda_runtime_error(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void set_current_device(int device_index) {
    throw_if_cuda_runtime_error(cudaSetDevice(device_index), "cudaSetDevice");
}

std::vector<int32_t> copy_device_ints(const Tensor& tensor, CudaAllocator& allocator) {
    set_current_device(tensor.device().index);
    std::vector<int32_t> host_values(tensor.numel(), 0);
    allocator.copy_to_host_async(host_values.data(), tensor.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return host_values;
}

void copy_values_to_tensor(Tensor& tensor, const std::vector<float>& values, CudaAllocator& allocator) {
    set_current_device(tensor.device().index);
    allocator.copy_to_device_async(tensor.data(), values.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
}

Config make_tiny_runner_config() {
    Config config{};
    config.max_num_batched_tokens = 1024;
    config.max_num_seqs = 2;
    config.max_model_len = 512;
    config.gpu_memory_utilization = 0.1f;
    config.tensor_parallel_size = 1;
    config.kvcache_block_size = Sequence::BLOCK_SIZE;
    config.hf_config.hidden_size = 2;
    config.hf_config.num_attention_heads = 1;
    config.hf_config.num_key_value_heads = 1;
    config.hf_config.num_hidden_layers = 1;
    config.hf_config.intermediate_size = 2;
    config.hf_config.vocab_size = 3;
    config.hf_config.max_position_embeddings = 1024;
    config.hf_config.head_dim = 2;
    config.hf_config.rms_norm_eps = 1e-6f;
    config.hf_config.rope_theta = 10000.0f;
    config.hf_config.tie_word_embeddings = false;
    config.hf_config.attention_bias = false;
    config.hf_config.hidden_act = "silu";
    config.hf_config.use_qk_norm = true;
    config.enforce_eager = true;  // Disable CUDA graphs for unit tests.
    config.num_kvcache_blocks = 4;  // Fixed block count for deterministic tests.
    return config;
}

SamplingParams deterministic_sampling_params() {
    SamplingParams params;
    params.temperature = 1e-10f;
    params.max_tokens = 4;
    params.ignore_eos = false;
    return params;
}

void load_tiny_model_weights(TransformerForCausalLM& model, CudaAllocator& allocator) {
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
}

std::vector<int32_t> manual_run(ModelRunner& runner,
                                const std::vector<Sequence*>& seqs,
                                bool is_prefill,
                                CudaAllocator& allocator) {
    const ModelRunner::ModelInputs batch = is_prefill ? runner.prepare_prefill(seqs) : runner.prepare_decode(seqs);
    ModelRunner::SampleInputs sample = runner.prepare_sample(seqs);
    Tensor logits = runner.run_model(batch.input_ids, batch.positions, is_prefill);
    return runner.sampler().forward(logits, sample.temperatures, sample.top_ks, sample.top_ps,
                                    sample.penalty_token_ids, sample.penalty_token_counts,
                                    sample.penalties, allocator);
}

TEST(ModelRunnerTest, PreparePrefillBuildsPagedContextAndMatchesRun) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const Config config = make_tiny_runner_config();
    ModelRunner runner(config, allocator);
    load_tiny_model_weights(runner.model(), allocator);

    EXPECT_EQ(runner.kv_cache().sizes(), (std::vector<int64_t>{2, 1, 4, Sequence::BLOCK_SIZE, 1, 2}));

    Scheduler scheduler(config);
    const SamplingParams params = deterministic_sampling_params();
    std::vector<int32_t> prefix(Sequence::BLOCK_SIZE, 0);
    std::vector<int32_t> seq0_tokens = prefix;
    seq0_tokens.push_back(1);
    std::vector<int32_t> seq1_tokens = prefix;
    seq1_tokens.push_back(2);
    scheduler.add(Sequence(seq0_tokens, params));
    scheduler.add(Sequence(seq1_tokens, params));

    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.second);
    ASSERT_EQ(scheduled.first.size(), 2u);
    EXPECT_EQ(scheduled.first[1]->num_cached_tokens, Sequence::BLOCK_SIZE);

    const ModelRunner::ModelInputs batch = runner.prepare_prefill(scheduled.first);
    Context& context = get_context();
    ASSERT_TRUE(context.is_prefill);

    const std::vector<int32_t> input_ids = copy_device_ints(batch.input_ids, allocator);
    const std::vector<int32_t> positions = copy_device_ints(batch.positions, allocator);
    const std::vector<int32_t> cu_q = copy_device_ints(context.cu_seqlens_q, allocator);
    const std::vector<int32_t> cu_k = copy_device_ints(context.cu_seqlens_k, allocator);
    const std::vector<int32_t> slot_mapping = copy_device_ints(context.slot_mapping, allocator);
    const std::vector<int32_t> block_tables = copy_device_ints(context.block_tables, allocator);

    std::vector<int32_t> expected_input_ids(static_cast<size_t>(Sequence::BLOCK_SIZE), 0);
    expected_input_ids.push_back(1);
    expected_input_ids.push_back(2);
    std::vector<int32_t> expected_positions(static_cast<size_t>(Sequence::BLOCK_SIZE), 0);
    for (int index = 0; index < Sequence::BLOCK_SIZE; ++index) {
        expected_positions[static_cast<size_t>(index)] = index;
    }
    expected_positions.push_back(Sequence::BLOCK_SIZE);
    expected_positions.push_back(Sequence::BLOCK_SIZE);
    std::vector<int32_t> expected_slots(static_cast<size_t>(Sequence::BLOCK_SIZE), 0);
    for (int index = 0; index < Sequence::BLOCK_SIZE; ++index) {
        expected_slots[static_cast<size_t>(index)] = index;
    }
    expected_slots.push_back(Sequence::BLOCK_SIZE);        // seq0 extra token: block 1, offset 0
    expected_slots.push_back(2 * Sequence::BLOCK_SIZE);    // seq1 extra token: block 2, offset 0

    EXPECT_EQ(input_ids, expected_input_ids);
    EXPECT_EQ(positions, expected_positions);
    EXPECT_EQ(cu_q, (std::vector<int32_t>{0, Sequence::BLOCK_SIZE + 1, Sequence::BLOCK_SIZE + 2}));
    EXPECT_EQ(cu_k, (std::vector<int32_t>{0, Sequence::BLOCK_SIZE + 1, 2 * Sequence::BLOCK_SIZE + 2}));
    EXPECT_EQ(slot_mapping, expected_slots);
    EXPECT_EQ(block_tables, (std::vector<int32_t>{0, 1, 0, 2}));

    const std::vector<int32_t> expected_tokens = manual_run(runner, scheduled.first, true, allocator);
    reset_context();

    const std::vector<int32_t> actual_tokens = runner.run(scheduled.first, true);
    EXPECT_EQ(actual_tokens, expected_tokens);
}

TEST(ModelRunnerTest, PreparePrefillKeepsOneQueryTokenForFullyCachedPrompt) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const Config config = make_tiny_runner_config();
    ModelRunner runner(config, allocator);
    load_tiny_model_weights(runner.model(), allocator);

    Scheduler scheduler(config);
    const SamplingParams params = deterministic_sampling_params();
    std::vector<int32_t> shared_prompt(Sequence::BLOCK_SIZE, 0);
    scheduler.add(Sequence(shared_prompt, params));
    scheduler.add(Sequence(shared_prompt, params));

    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.second);
    ASSERT_EQ(scheduled.first.size(), 2u);
    EXPECT_EQ(scheduled.first[1]->num_cached_tokens, Sequence::BLOCK_SIZE);

    const ModelRunner::ModelInputs batch = runner.prepare_prefill(scheduled.first);
    Context& context = get_context();
    ASSERT_TRUE(context.is_prefill);

    const std::vector<int32_t> input_ids = copy_device_ints(batch.input_ids, allocator);
    const std::vector<int32_t> positions = copy_device_ints(batch.positions, allocator);
    const std::vector<int32_t> cu_q = copy_device_ints(context.cu_seqlens_q, allocator);
    const std::vector<int32_t> cu_k = copy_device_ints(context.cu_seqlens_k, allocator);
    const std::vector<int32_t> slot_mapping = copy_device_ints(context.slot_mapping, allocator);
    const std::vector<int32_t> block_tables = copy_device_ints(context.block_tables, allocator);

    std::vector<int32_t> expected_input_ids(static_cast<size_t>(Sequence::BLOCK_SIZE), 0);
    expected_input_ids.push_back(0);
    std::vector<int32_t> expected_positions(static_cast<size_t>(Sequence::BLOCK_SIZE), 0);
    for (int index = 0; index < Sequence::BLOCK_SIZE; ++index) {
        expected_positions[static_cast<size_t>(index)] = index;
    }
    expected_positions.push_back(Sequence::BLOCK_SIZE - 1);
    std::vector<int32_t> expected_slots(static_cast<size_t>(Sequence::BLOCK_SIZE), 0);
    for (int index = 0; index < Sequence::BLOCK_SIZE; ++index) {
        expected_slots[static_cast<size_t>(index)] = index;
    }
    expected_slots.push_back(Sequence::BLOCK_SIZE - 1);

    EXPECT_EQ(input_ids, expected_input_ids);
    EXPECT_EQ(positions, expected_positions);
    EXPECT_EQ(cu_q, (std::vector<int32_t>{0, Sequence::BLOCK_SIZE, Sequence::BLOCK_SIZE + 1}));
    EXPECT_EQ(cu_k, (std::vector<int32_t>{0, Sequence::BLOCK_SIZE, 2 * Sequence::BLOCK_SIZE}));
    EXPECT_EQ(slot_mapping, expected_slots);
    EXPECT_EQ(block_tables, (std::vector<int32_t>{0, 0}));

    const std::vector<int32_t> expected_tokens = manual_run(runner, scheduled.first, true, allocator);
    reset_context();

    const std::vector<int32_t> actual_tokens = runner.run(scheduled.first, true);
    EXPECT_EQ(actual_tokens, expected_tokens);
}

TEST(ModelRunnerTest, PrepareDecodeBuildsContextAndMatchesRun) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const Config config = make_tiny_runner_config();
    ModelRunner runner(config, allocator);
    load_tiny_model_weights(runner.model(), allocator);

    Scheduler scheduler(config);
    const SamplingParams params = deterministic_sampling_params();
    scheduler.add(Sequence({0}, params));

    auto prefill = scheduler.schedule();
    ASSERT_TRUE(prefill.second);
    std::vector<int32_t> prefill_tokens = runner.run(prefill.first, true);
    ASSERT_EQ(prefill_tokens.size(), 1u);
    scheduler.postprocess(prefill.first, prefill_tokens);

    auto decode = scheduler.schedule();
    ASSERT_FALSE(decode.second);
    ASSERT_EQ(decode.first.size(), 1u);

    const ModelRunner::ModelInputs batch = runner.prepare_decode(decode.first);
    Context& context = get_context();
    ASSERT_FALSE(context.is_prefill);

    EXPECT_EQ(copy_device_ints(batch.input_ids, allocator), std::vector<int32_t>{prefill_tokens[0]});
    EXPECT_EQ(copy_device_ints(batch.positions, allocator), std::vector<int32_t>{1});
    EXPECT_EQ(copy_device_ints(context.slot_mapping, allocator), std::vector<int32_t>{1});
    EXPECT_EQ(copy_device_ints(context.context_lens, allocator), std::vector<int32_t>{2});
    EXPECT_EQ(copy_device_ints(context.block_tables, allocator), std::vector<int32_t>{0});

    const std::vector<int32_t> expected_tokens = manual_run(runner, decode.first, false, allocator);
    reset_context();

    const std::vector<int32_t> actual_tokens = runner.run(decode.first, false);
    EXPECT_EQ(actual_tokens, expected_tokens);
}

// ---------------------------------------------------------------------------
// End-to-end chunked prefill: a 3-block prompt processed in 2 chunks should
// produce the same final decode token as a single-shot prefill.
// ---------------------------------------------------------------------------

TEST(ModelRunnerTest, ChunkedPrefillMatchesSingleShotPrefillAndFirstDecodeToken) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const Config base_config = make_tiny_runner_config();
    const SamplingParams params = deterministic_sampling_params();

    // Build the same prompt twice: once for chunked, once for baseline.
    const int prompt_len = Sequence::BLOCK_SIZE + Sequence::BLOCK_SIZE / 2;
    std::vector<int32_t> prompt(static_cast<size_t>(prompt_len), 1);

    // --- Baseline: default scheduler, one-shot prefill -> one decode. ---
    int32_t baseline_decode_token = -1;
    {
        ModelRunner runner(base_config, allocator);
        load_tiny_model_weights(runner.model(), allocator);
        Scheduler scheduler(base_config);
        scheduler.add(Sequence(prompt, params));
        auto sched1 = scheduler.schedule();
        ASSERT_TRUE(sched1.second);
        auto toks1 = runner.run(sched1.first, true);
        ASSERT_EQ(toks1.size(), 1u);
        scheduler.postprocess(sched1.first, toks1);

        auto sched2 = scheduler.schedule();
        ASSERT_FALSE(sched2.second);
        auto toks2 = runner.run(sched2.first, false);
        ASSERT_EQ(toks2.size(), 1u);
        baseline_decode_token = toks2[0];
    }

    // --- Chunked: split prompt into chunks of BLOCK_SIZE tokens. ---
    Config chunked_config = base_config;
    chunked_config.enable_chunked_prefill = true;
    chunked_config.max_num_prefill_tokens = Sequence::BLOCK_SIZE;  // exactly one block per chunk

    ModelRunner runner(chunked_config, allocator);
    load_tiny_model_weights(runner.model(), allocator);
    Scheduler scheduler(chunked_config);
    scheduler.add(Sequence(prompt, params));

    // Chunk 1: first BLOCK_SIZE tokens, mid-chunk (token_id is garbage).
    auto s1 = scheduler.schedule();
    ASSERT_TRUE(s1.second);
    ASSERT_EQ(s1.first.size(), 1u);
    Sequence* seq = s1.first[0];
    ASSERT_EQ(seq->num_tokens_to_process, Sequence::BLOCK_SIZE);
    auto t1 = runner.run(s1.first, true);
    ASSERT_EQ(t1.size(), 1u);
    scheduler.postprocess(s1.first, t1);
    EXPECT_EQ(seq->num_computed_tokens, Sequence::BLOCK_SIZE);
    EXPECT_EQ(seq->num_tokens, prompt_len);  // mid-chunk: no append

    // Chunk 2 (terminal): remaining BLOCK_SIZE/2 tokens, sampled token is appended.
    auto s2 = scheduler.schedule();
    ASSERT_TRUE(s2.second);
    ASSERT_EQ(s2.first.size(), 1u);
    EXPECT_EQ(s2.first[0], seq);
    EXPECT_EQ(seq->num_tokens_to_process, Sequence::BLOCK_SIZE / 2);
    auto t2 = runner.run(s2.first, true);
    ASSERT_EQ(t2.size(), 1u);
    scheduler.postprocess(s2.first, t2);
    EXPECT_EQ(seq->num_computed_tokens, prompt_len);
    EXPECT_EQ(seq->num_tokens, prompt_len + 1);

    // The first-decode-equivalent token produced by the terminal prefill chunk
    // must match the baseline decode token, since chunked prefill only
    // partitions compute without changing per-layer math.
    EXPECT_EQ(t2[0], baseline_decode_token);

    // Next schedule must be a decode step.
    auto s3 = scheduler.schedule();
    EXPECT_FALSE(s3.second);
}

} // namespace
} // namespace nano_vllm