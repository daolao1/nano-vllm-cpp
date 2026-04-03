#include "core/config.h"
#include "layers/attention.h"
#include "layers/embed_head.h"
#include "layers/linear.h"
#include "layers/sampler.h"
#include "utils/context.h"
#include "utils/cuda_allocator.h"
#include "utils/tensor_parallel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace nano_vllm {
namespace {

bool has_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

Tensor make_device_float_tensor(const std::vector<int64_t>& sizes,
                                const std::vector<float>& host_values,
                                CudaAllocator& allocator) {
    Tensor tensor = Tensor::empty(sizes, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    allocator.copy_to_device_async(tensor.data(), host_values.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return tensor;
}

Tensor make_device_int_tensor(const std::vector<int64_t>& sizes,
                              const std::vector<int32_t>& host_values,
                              CudaAllocator& allocator) {
    Tensor tensor = Tensor::empty(sizes, ScalarType::kInt32, Device{DeviceType::kCUDA, 0}, allocator);
    allocator.copy_to_device_async(tensor.data(), host_values.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return tensor;
}

std::vector<float> copy_device_floats(const Tensor& tensor, CudaAllocator& allocator) {
    std::vector<float> host_values(tensor.numel(), 0.0f);
    allocator.copy_to_host_async(host_values.data(), tensor.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return host_values;
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

TEST(LayerTest, ContextSetAndResetWorks) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    Tensor cu_q = make_device_int_tensor({2}, {0, 3}, allocator);
    Tensor slot_mapping = make_device_int_tensor({3}, {0, 1, 2}, allocator);

    set_context(true, cu_q, Tensor(), 3, 3, slot_mapping, Tensor(), Tensor());
    EXPECT_TRUE(get_context().is_prefill);
    EXPECT_TRUE(get_context().cu_seqlens_q.defined());
    EXPECT_TRUE(get_context().slot_mapping.defined());

    reset_context();
    EXPECT_FALSE(get_context().is_prefill);
    EXPECT_FALSE(get_context().cu_seqlens_q.defined());
    EXPECT_FALSE(get_context().slot_mapping.defined());
}

TEST(LayerTest, AttentionPrefillMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    Attention attention(1, 1, 2, 2);
    Tensor query = make_device_float_tensor({3, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, allocator);
    Tensor key = make_device_float_tensor({3, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, allocator);
    Tensor value = make_device_float_tensor({3, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, allocator);
    Tensor cu_q = make_device_int_tensor({2}, {0, 3}, allocator);
    Tensor cu_k = make_device_int_tensor({2}, {0, 3}, allocator);

    set_context(true, cu_q, cu_k, 3, 3, Tensor(), Tensor(), Tensor());
    Tensor output = attention.forward(query, key, value, allocator);
    reset_context();

    const std::vector<float> actual = copy_device_floats(output, allocator);
    const float s = 1.0f / std::sqrt(2.0f);
    const float b0 = std::exp(0.0f * s - 1.0f * s);
    const float b1 = 1.0f;
    const float c0 = std::exp(1.0f * s - 2.0f * s);
    const float c1 = std::exp(1.0f * s - 2.0f * s);
    const float c2 = 1.0f;
    const std::vector<float> expected = {
        1.0f, 2.0f,
        (b0 * 1.0f + b1 * 3.0f) / (b0 + b1), (b0 * 2.0f + b1 * 4.0f) / (b0 + b1),
        (c0 * 1.0f + c1 * 3.0f + c2 * 5.0f) / (c0 + c1 + c2),
        (c0 * 2.0f + c1 * 4.0f + c2 * 6.0f) / (c0 + c1 + c2),
    };

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
    }
}

TEST(LayerTest, AttentionDecodeReadsKvCache) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    Attention attention(1, 1, 2, 2);
    Tensor k_cache = make_device_float_tensor({2, 2, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, allocator);
    Tensor v_cache = make_device_float_tensor({2, 2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f}, allocator);
    attention.set_kv_cache(k_cache, v_cache);

    Tensor query = make_device_float_tensor({1, 1, 2}, {1.0f, 1.0f}, allocator);
    Tensor key = make_device_float_tensor({1, 1, 2}, {1.0f, 1.0f}, allocator);
    Tensor value = make_device_float_tensor({1, 1, 2}, {5.0f, 6.0f}, allocator);
    Tensor slot_mapping = make_device_int_tensor({1}, {2}, allocator);
    Tensor context_lens = make_device_int_tensor({1}, {3}, allocator);
    Tensor block_tables = make_device_int_tensor({1, 2}, {0, 1}, allocator);

    set_context(false, Tensor(), Tensor(), 0, 0, slot_mapping, context_lens, block_tables);
    Tensor output = attention.forward(query, key, value, allocator);
    reset_context();

    const std::vector<float> actual = copy_device_floats(output, allocator);
    const float s = 1.0f / std::sqrt(2.0f);
    const float p0 = std::exp(1.0f * s - 2.0f * s);
    const float p1 = p0;
    const float p2 = 1.0f;
    const std::vector<float> expected = {
        (p0 * 1.0f + p1 * 3.0f + p2 * 5.0f) / (p0 + p1 + p2),
        (p0 * 2.0f + p1 * 4.0f + p2 * 6.0f) / (p0 + p1 + p2),
    };

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
    }
}

TEST(LayerTest, AttentionPrefillWithPrefixCacheReadsPagedKvCache) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    Attention attention(1, 1, 2, 2);
    Tensor k_cache = make_device_float_tensor({2, 2, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, allocator);
    Tensor v_cache = make_device_float_tensor({2, 2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f}, allocator);
    attention.set_kv_cache(k_cache, v_cache);

    Tensor query = make_device_float_tensor({1, 1, 2}, {1.0f, 1.0f}, allocator);
    Tensor key = make_device_float_tensor({1, 1, 2}, {1.0f, 1.0f}, allocator);
    Tensor value = make_device_float_tensor({1, 1, 2}, {5.0f, 6.0f}, allocator);
    Tensor slot_mapping = make_device_int_tensor({1}, {2}, allocator);
    Tensor cu_q = make_device_int_tensor({2}, {0, 1}, allocator);
    Tensor cu_k = make_device_int_tensor({2}, {0, 3}, allocator);
    Tensor block_tables = make_device_int_tensor({1, 2}, {0, 1}, allocator);

    set_context(true, cu_q, cu_k, 1, 3, slot_mapping, Tensor(), block_tables);
    Tensor output = attention.forward(query, key, value, allocator);
    reset_context();

    const std::vector<float> actual = copy_device_floats(output, allocator);
    const float s = 1.0f / std::sqrt(2.0f);
    const float p0 = std::exp(1.0f * s - 2.0f * s);
    const float p1 = p0;
    const float p2 = 1.0f;
    const std::vector<float> expected = {
        (p0 * 1.0f + p1 * 3.0f + p2 * 5.0f) / (p0 + p1 + p2),
        (p0 * 2.0f + p1 * 4.0f + p2 * 6.0f) / (p0 + p1 + p2),
    };

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
    }
}

TEST(LayerTest, ColumnParallelLinearMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    ColumnParallelLinear linear(3, 2, false, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    allocator.copy_to_device_async(linear.weight().data(),
                                   std::vector<float>{1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f}.data(),
                                   linear.weight().nbytes(),
                                   nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor input = make_device_float_tensor({2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 1.0f}, allocator);
    Tensor output = linear.forward(input, allocator);
    const std::vector<float> actual = copy_device_floats(output, allocator);
    const std::vector<float> expected = {7.0f, 7.0f, 8.0f, 6.5f};
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
    }
}

TEST(LayerTest, ColumnParallelLinearBiasMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    ColumnParallelLinear linear(3, 2, true, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    const std::vector<float> weight = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f};
    const std::vector<float> bias = {0.25f, -0.75f};
    allocator.copy_to_device_async(linear.weight().data(), weight.data(), linear.weight().nbytes(), nullptr);
    allocator.copy_to_device_async(linear.bias().data(), bias.data(), linear.bias().nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor input = make_device_float_tensor({2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 1.0f}, allocator);
    Tensor output = linear.forward(input, allocator);
    const std::vector<float> actual = copy_device_floats(output, allocator);
    const std::vector<float> expected = {7.25f, 6.25f, 8.25f, 5.75f};
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
    }
}

TEST(LayerTest, TensorParallelCollectivesMatchPythonSemantics) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    ScopedTensorParallelCommunicator communicator(make_in_process_tensor_parallel_communicator());

    RowParallelLinear row_rank0(4, 2, true, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 0, "row_linear");
    RowParallelLinear row_rank1(4, 2, true, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 1, "row_linear");
    const std::vector<float> row_weight_rank0 = {1.0f, 2.0f, -1.0f, 0.5f};
    const std::vector<float> row_weight_rank1 = {3.0f, 4.0f, 2.0f, -2.0f};
    const std::vector<float> row_bias = {0.25f, -0.75f};
    allocator.copy_to_device_async(row_rank0.weight().data(), row_weight_rank0.data(), row_rank0.weight().nbytes(), nullptr);
    allocator.copy_to_device_async(row_rank1.weight().data(), row_weight_rank1.data(), row_rank1.weight().nbytes(), nullptr);
    allocator.copy_to_device_async(row_rank0.bias().data(), row_bias.data(), row_rank0.bias().nbytes(), nullptr);
    allocator.copy_to_device_async(row_rank1.bias().data(), row_bias.data(), row_rank1.bias().nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor local_input_rank0 = make_device_float_tensor({2, 2}, {1.0f, -1.0f, 0.0f, 2.0f}, allocator);
    Tensor local_input_rank1 = make_device_float_tensor({2, 2}, {2.0f, 0.5f, -3.0f, 1.0f}, allocator);
    Tensor row_output0 = row_rank0.forward(local_input_rank0, allocator);
    Tensor row_output1 = row_rank1.forward(local_input_rank1, allocator);
    const std::vector<float> reduced0 = copy_device_floats(row_output0, allocator);
    const std::vector<float> reduced1 = copy_device_floats(row_output1, allocator);
    const std::vector<float> row_expected = {7.25f, 0.75f, -0.75f, -7.75f};
    for (size_t i = 0; i < row_expected.size(); ++i) {
        EXPECT_NEAR(reduced0[i], row_expected[i], 1e-5f);
        EXPECT_NEAR(reduced1[i], row_expected[i], 1e-5f);
    }

    VocabParallelEmbedding embedding_rank0(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 0, "embedding");
    VocabParallelEmbedding embedding_rank1(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 1, "embedding");
    const std::vector<float> embedding_weight_rank0 = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> embedding_weight_rank1 = {1.0f, 1.0f, 2.0f, -1.0f};
    allocator.copy_to_device_async(embedding_rank0.weight().data(),
                                   embedding_weight_rank0.data(),
                                   embedding_rank0.weight().nbytes(),
                                   nullptr);
    allocator.copy_to_device_async(embedding_rank1.weight().data(),
                                   embedding_weight_rank1.data(),
                                   embedding_rank1.weight().nbytes(),
                                   nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor ids = make_device_int_tensor({2, 2}, {0, 2, 3, 1}, allocator);
    Tensor embedded0 = embedding_rank0.forward(ids, allocator);
    Tensor embedded1 = embedding_rank1.forward(ids, allocator);
    const std::vector<float> embed0 = copy_device_floats(embedded0, allocator);
    const std::vector<float> embed1 = copy_device_floats(embedded1, allocator);
    const std::vector<float> embed_expected = {1.0f, 0.0f, 1.0f, 1.0f, 2.0f, -1.0f, 0.0f, 1.0f};
    for (size_t i = 0; i < embed_expected.size(); ++i) {
        EXPECT_NEAR(embed0[i], embed_expected[i], 1e-5f);
        EXPECT_NEAR(embed1[i], embed_expected[i], 1e-5f);
    }

    ParallelLMHead lm_head_rank0(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 0, "lm_head");
    ParallelLMHead lm_head_rank1(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 1, "lm_head");
    allocator.copy_to_device_async(lm_head_rank0.weight().data(),
                                   embedding_weight_rank0.data(),
                                   lm_head_rank0.weight().nbytes(),
                                   nullptr);
    allocator.copy_to_device_async(lm_head_rank1.weight().data(),
                                   embedding_weight_rank1.data(),
                                   lm_head_rank1.weight().nbytes(),
                                   nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor hidden = make_device_float_tensor({2, 2}, {1.0f, 0.0f, 0.5f, 1.5f}, allocator);
    Tensor logits_rank0 = lm_head_rank0.forward(hidden, allocator);
    Tensor logits_rank1 = lm_head_rank1.forward(hidden, allocator);
    EXPECT_TRUE(logits_rank0.defined());
    EXPECT_FALSE(logits_rank1.defined());

    const std::vector<float> gathered_logits = copy_device_floats(logits_rank0, allocator);
    const std::vector<float> logits_expected = {1.0f, 0.0f, 1.0f, 2.0f, 0.5f, 1.5f, 2.0f, -0.5f};
    for (size_t i = 0; i < logits_expected.size(); ++i) {
        EXPECT_NEAR(gathered_logits[i], logits_expected[i], 1e-5f);
    }
}

TEST(LayerTest, QkvAndMergedColumnLinearProduceExpectedOutputs) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    QKVParallelLinear qkv(2, 1, 2, 1, false, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    allocator.copy_to_device_async(qkv.weight().data(),
                                   std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, 2.0f}.data(),
                                   qkv.weight().nbytes(),
                                   nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor input = make_device_float_tensor({1, 2}, {2.0f, 3.0f}, allocator);
    Tensor qkv_out = qkv.forward(input, allocator);
    const std::vector<float> qkv_actual = copy_device_floats(qkv_out, allocator);
    const std::vector<float> qkv_expected = {2.0f, 3.0f, 5.0f, 4.0f};
    for (size_t i = 0; i < qkv_actual.size(); ++i) {
        EXPECT_NEAR(qkv_actual[i], qkv_expected[i], 1e-5f);
    }

    MergedColumnParallelLinear merged(2, {1, 2}, false, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    allocator.copy_to_device_async(merged.weight().data(),
                                   std::vector<float>{1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f}.data(),
                                   merged.weight().nbytes(),
                                   nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor merged_out = merged.forward(input, allocator);
    const std::vector<float> merged_actual = copy_device_floats(merged_out, allocator);
    const std::vector<float> merged_expected = {5.0f, 4.0f, 6.0f};
    for (size_t i = 0; i < merged_actual.size(); ++i) {
        EXPECT_NEAR(merged_actual[i], merged_expected[i], 1e-5f);
    }
}

TEST(LayerTest, EmbeddingAndLmHeadMatchReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    VocabParallelEmbedding embedding(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    const std::vector<float> weight_values = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, -1.0f};
    allocator.copy_to_device_async(embedding.weight().data(), weight_values.data(), embedding.weight().nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    Tensor ids = make_device_int_tensor({2, 2}, {0, 2, 3, 1}, allocator);
    Tensor embedded = embedding.forward(ids, allocator);
    const std::vector<float> embedded_actual = copy_device_floats(embedded, allocator);
    const std::vector<float> embedded_expected = {1.0f, 0.0f, 1.0f, 1.0f, 2.0f, -1.0f, 0.0f, 1.0f};
    for (size_t i = 0; i < embedded_actual.size(); ++i) {
        EXPECT_NEAR(embedded_actual[i], embedded_expected[i], 1e-5f);
    }

    ParallelLMHead lm_head(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    lm_head.set_weight(embedding.weight());
    Tensor hidden = make_device_float_tensor({5, 2}, {1.0f, 0.0f, 0.5f, 0.5f, 1.0f, 1.0f, -1.0f, 2.0f, 0.0f, 1.0f}, allocator);
    Tensor cu_q = make_device_int_tensor({3}, {0, 2, 5}, allocator);
    set_context(true, cu_q, Tensor(), 0, 0, Tensor(), Tensor(), Tensor());
    Tensor logits = lm_head.forward(hidden, allocator);
    reset_context();
    const std::vector<float> logits_actual = copy_device_floats(logits, allocator);
    const std::vector<float> logits_expected = {0.5f, 0.5f, 1.0f, 0.5f, 0.0f, 1.0f, 1.0f, -1.0f};
    for (size_t i = 0; i < logits_actual.size(); ++i) {
        EXPECT_NEAR(logits_actual[i], logits_expected[i], 1e-5f);
    }
}

TEST(LayerTest, SamplerUsesDeterministicSeed) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    Sampler sampler(123);
    Tensor logits = make_device_float_tensor({2, 3}, {2.0f, 1.0f, 0.0f, 0.5f, 3.0f, 1.5f}, allocator);
    Tensor temperatures = make_device_float_tensor({2}, {1.0f, 0.5f}, allocator);
    Tensor top_ks = make_device_int_tensor({2}, {-1, -1}, allocator);
    Tensor top_ps = make_device_float_tensor({2}, {1.0f, 1.0f}, allocator);

    const std::vector<int32_t> first = sampler.forward(logits, temperatures, top_ks, top_ps,
                                                       Tensor(), Tensor(), Tensor(), allocator);
    const std::vector<int32_t> second = sampler.forward(logits, temperatures, top_ks, top_ps,
                                                        Tensor(), Tensor(), Tensor(), allocator);

    EXPECT_EQ(first, second);
    ASSERT_EQ(first.size(), 2u);
    for (int32_t token : first) {
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 3);
    }
}

} // namespace
} // namespace nano_vllm