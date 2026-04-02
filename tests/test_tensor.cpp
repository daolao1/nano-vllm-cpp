#include "layers/kernel_ops.h"
#include "utils/cublas_wrapper.h"
#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
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

float silu_ref(float value) {
    return value / (1.0f + std::exp(-value));
}

TEST(TensorTest, MetadataTracksContiguousLayout) {
    std::vector<float> host(24, 0.0f);
    auto memory = Memory::make_borrowed(host.data(), host.size() * sizeof(float), Device{});
    Tensor tensor = Tensor::from_memory(memory, {2, 3, 4}, ScalarType::kFloat32);

    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.dim(), 3);
    EXPECT_EQ(tensor.numel(), 24u);
    EXPECT_EQ(tensor.nbytes(), 24u * sizeof(float));
    EXPECT_EQ(tensor.sizes(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_TRUE(tensor.is_contiguous());
}

TEST(TensorTest, ReshapeAndSliceCreateViews) {
    std::vector<float> host(24, 0.0f);
    auto memory = Memory::make_borrowed(host.data(), host.size() * sizeof(float), Device{});
    Tensor tensor = Tensor::from_memory(memory, {2, 3, 4}, ScalarType::kFloat32);

    Tensor reshaped = tensor.reshape({6, 4});
    EXPECT_EQ(reshaped.memory().get(), tensor.memory().get());
    EXPECT_EQ(reshaped.strides(), (std::vector<int64_t>{4, 1}));
    EXPECT_TRUE(reshaped.is_contiguous());

    Tensor sliced = tensor.slice(1, 1, 3);
    EXPECT_EQ(sliced.memory().get(), tensor.memory().get());
    EXPECT_EQ(sliced.sizes(), (std::vector<int64_t>{2, 2, 4}));
    EXPECT_EQ(sliced.memory_offset(), 4u);
    EXPECT_EQ(sliced.data_as<float>(), host.data() + 4);
    EXPECT_FALSE(sliced.is_contiguous());
    EXPECT_THROW(sliced.reshape({4, 4}), std::invalid_argument);
}

TEST(TensorTest, Phase1KernelListMatchesStepFourScope) {
    const std::vector<KernelSpec>& specs = phase1_kernel_specs();
    ASSERT_EQ(specs.size(), 5u);
    EXPECT_STREQ(specs[0].name, "rms_norm");
    EXPECT_STREQ(specs[1].name, "add_rms_norm");
    EXPECT_STREQ(specs[2].name, "rotary_embedding");
    EXPECT_STREQ(specs[3].name, "silu_and_mul");
    EXPECT_STREQ(specs[4].name, "store_kvcache");
}

TEST(TensorTest, CudaAllocatorZeroInitializesTensor) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    Tensor tensor = Tensor::zeros({8}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    std::array<float, 8> host{};

    allocator.copy_to_host_async(host.data(), tensor.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](float value) { return value == 0.0f; }));
}

TEST(TensorTest, CublasWrapperRunsRowMajorGemm) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    const std::array<float, 6> a = {1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f};
    const std::array<float, 6> b = {7.0f, 8.0f,
                                    9.0f, 10.0f,
                                    11.0f, 12.0f};
    const std::array<float, 4> expected = {58.0f, 64.0f,
                                           139.0f, 154.0f};

    CudaAllocator& allocator = CudaAllocator::instance();
    Tensor a_tensor = Tensor::empty({2, 3}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    Tensor b_tensor = Tensor::empty({3, 2}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    Tensor c_tensor = Tensor::zeros({2, 2}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);

    allocator.copy_to_device_async(a_tensor.data(), a.data(), a_tensor.nbytes(), nullptr);
    allocator.copy_to_device_async(b_tensor.data(), b.data(), b_tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    CublasHandle handle;
    gemm_row_major(handle, GemmConfig{2, 2, 3}, ScalarType::kFloat32, a_tensor.data(), b_tensor.data(), c_tensor.data());
    allocator.synchronize_stream(nullptr);

    std::array<float, 4> host{};
    allocator.copy_to_host_async(host.data(), c_tensor.data(), c_tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    EXPECT_EQ(host, expected);
}

TEST(TensorTest, RmsNormMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<float> input_values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.0f, -3.0f,
    };
    const std::vector<float> weight_values = {1.0f, 1.5f, 0.5f, 2.0f};

    Tensor input = make_device_float_tensor({2, 4}, input_values, allocator);
    Tensor weight = make_device_float_tensor({4}, weight_values, allocator);
    Tensor output = Tensor::zeros({2, 4}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);

    rms_norm(input, weight, 1e-6f, output);

    const std::vector<float> actual = copy_device_floats(output, allocator);
    std::vector<float> expected(actual.size(), 0.0f);
    for (int row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < 4; ++col) {
            const float value = input_values[row * 4 + col];
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / 4.0f + 1e-6f);
        for (int col = 0; col < 4; ++col) {
            expected[row * 4 + col] = input_values[row * 4 + col] * inv_rms * weight_values[col];
        }
    }

    for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}

TEST(TensorTest, RmsNormFallbackMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<float> input_values = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        -1.0f, -2.0f, 0.5f, 1.0f, 2.5f, -3.0f,
    };
    const std::vector<float> weight_values = {0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.75f};

    Tensor input = make_device_float_tensor({2, 6}, input_values, allocator);
    Tensor weight = make_device_float_tensor({6}, weight_values, allocator);
    Tensor output = Tensor::zeros({2, 6}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);

    rms_norm(input, weight, 1e-6f, output);

    const std::vector<float> actual = copy_device_floats(output, allocator);
    std::vector<float> expected(actual.size(), 0.0f);
    for (int row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < 6; ++col) {
            const float value = input_values[row * 6 + col];
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / 6.0f + 1e-6f);
        for (int col = 0; col < 6; ++col) {
            expected[row * 6 + col] = input_values[row * 6 + col] * inv_rms * weight_values[col];
        }
    }

    for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}

TEST(TensorTest, AddRmsNormMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<float> input_values = {
        1.0f, -1.0f, 2.0f, 0.5f,
        0.25f, 0.75f, -0.5f, 1.5f,
    };
    const std::vector<float> residual_values = {
        0.5f, 1.0f, -0.5f, 1.5f,
        -1.0f, 0.25f, 0.5f, -0.25f,
    };
    const std::vector<float> weight_values = {1.0f, 0.5f, 2.0f, 1.5f};

    Tensor input = make_device_float_tensor({2, 4}, input_values, allocator);
    Tensor residual = make_device_float_tensor({2, 4}, residual_values, allocator);
    Tensor weight = make_device_float_tensor({4}, weight_values, allocator);
    Tensor output = Tensor::zeros({2, 4}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    Tensor residual_out = Tensor::zeros({2, 4}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);

    add_rms_norm(input, residual, weight, 1e-6f, output, residual_out);

    const std::vector<float> actual_output = copy_device_floats(output, allocator);
    const std::vector<float> actual_residual = copy_device_floats(residual_out, allocator);
    std::vector<float> expected_output(actual_output.size(), 0.0f);
    std::vector<float> expected_residual(actual_residual.size(), 0.0f);

    for (int row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < 4; ++col) {
            const float combined = input_values[row * 4 + col] + residual_values[row * 4 + col];
            expected_residual[row * 4 + col] = combined;
            sum_sq += combined * combined;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / 4.0f + 1e-6f);
        for (int col = 0; col < 4; ++col) {
            expected_output[row * 4 + col] = expected_residual[row * 4 + col] * inv_rms * weight_values[col];
        }
    }

    for (size_t index = 0; index < actual_output.size(); ++index) {
        EXPECT_NEAR(actual_output[index], expected_output[index], 1e-5f);
        EXPECT_NEAR(actual_residual[index], expected_residual[index], 1e-6f);
    }
}

TEST(TensorTest, RotaryEmbeddingMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<int32_t> positions = {1, 3};
    const std::vector<float> cache_values = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.25f, 0.75f, -0.5f,
        0.0f, -1.0f, 1.0f, 0.0f,
        0.8f, -0.3f, 0.6f, 0.4f,
    };
    const std::vector<float> query_values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.0f, -2.5f,
        0.25f, -0.75f, 1.5f, 2.0f,
        3.0f, -2.0f, 1.0f, 0.5f,
    };
    const std::vector<float> key_values = {
        2.0f, -1.0f, 1.0f, 0.0f,
        0.5f, 1.5f, -0.5f, 2.5f,
    };

    Tensor positions_tensor = make_device_int_tensor({2}, positions, allocator);
    Tensor cache = make_device_float_tensor({4, 1, 4}, cache_values, allocator);
    Tensor query = make_device_float_tensor({2, 2, 4}, query_values, allocator);
    Tensor key = make_device_float_tensor({2, 1, 4}, key_values, allocator);

    rotary_embedding(positions_tensor, cache, query, key);

    const std::vector<float> actual_query = copy_device_floats(query, allocator);
    const std::vector<float> actual_key = copy_device_floats(key, allocator);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;

    auto rotate_rows = [&](std::vector<float>& values, int rows_per_token) {
        for (int token = 0; token < 2; ++token) {
            const int position = positions[token];
            const float cos0 = cache_values[position * 4 + 0];
            const float cos1 = cache_values[position * 4 + 1];
            const float sin0 = cache_values[position * 4 + 2];
            const float sin1 = cache_values[position * 4 + 3];
            for (int head = 0; head < rows_per_token; ++head) {
                const int row = token * rows_per_token + head;
                const float x0 = values[row * 4 + 0];
                const float x1 = values[row * 4 + 1];
                const float x2 = values[row * 4 + 2];
                const float x3 = values[row * 4 + 3];
                values[row * 4 + 0] = x0 * cos0 - x2 * sin0;
                values[row * 4 + 2] = x2 * cos0 + x0 * sin0;
                values[row * 4 + 1] = x1 * cos1 - x3 * sin1;
                values[row * 4 + 3] = x3 * cos1 + x1 * sin1;
            }
        }
    };

    rotate_rows(expected_query, 2);
    rotate_rows(expected_key, 1);

    for (size_t index = 0; index < actual_query.size(); ++index) {
        EXPECT_NEAR(actual_query[index], expected_query[index], 1e-5f);
    }
    for (size_t index = 0; index < actual_key.size(); ++index) {
        EXPECT_NEAR(actual_key[index], expected_key[index], 1e-5f);
    }
}

TEST(TensorTest, RotaryEmbeddingFallbackMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<int32_t> positions = {0, 2};
    const std::vector<float> cache_values = {
        1.0f, 0.5f, -1.0f, 0.0f, 0.25f, -0.5f,
        0.5f, 0.25f, 0.75f, -0.25f, 0.1f, 0.2f,
        -0.5f, 1.0f, 0.25f, 0.75f, -0.5f, 0.5f,
    };
    const std::vector<float> query_values = {
        1.0f, -1.0f, 0.5f, 2.0f, -2.0f, 1.5f,
        0.0f, 1.0f, -0.5f, 2.5f, -1.5f, 0.25f,
        -0.75f, 0.5f, 1.0f, -2.0f, 3.0f, -1.0f,
        0.25f, -0.5f, 0.75f, 1.5f, -2.5f, 0.5f,
    };

    Tensor positions_tensor = make_device_int_tensor({2}, positions, allocator);
    Tensor cache = make_device_float_tensor({3, 6}, cache_values, allocator);
    Tensor query = make_device_float_tensor({2, 2, 6}, query_values, allocator);
    Tensor key = make_device_float_tensor({2, 2, 6}, query_values, allocator);

    rotary_embedding(positions_tensor, cache, query, key);

    const std::vector<float> actual_query = copy_device_floats(query, allocator);
    std::vector<float> expected_query = query_values;
    for (int token = 0; token < 2; ++token) {
        const int position = positions[token];
        for (int head = 0; head < 2; ++head) {
            const int row = token * 2 + head;
            for (int idx = 0; idx < 3; ++idx) {
                const float cos_v = cache_values[position * 6 + idx];
                const float sin_v = cache_values[position * 6 + idx + 3];
                const float x1 = expected_query[row * 6 + idx];
                const float x2 = expected_query[row * 6 + idx + 3];
                expected_query[row * 6 + idx] = x1 * cos_v - x2 * sin_v;
                expected_query[row * 6 + idx + 3] = x2 * cos_v + x1 * sin_v;
            }
        }
    }

    for (size_t index = 0; index < actual_query.size(); ++index) {
        EXPECT_NEAR(actual_query[index], expected_query[index], 1e-5f);
    }
}

TEST(TensorTest, SiluAndMulMatchesReference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<float> gate_up_values = {
        -1.0f, 0.5f, 2.0f, 1.5f, -2.0f, 0.25f,
        0.0f, -0.25f, 1.25f, 3.0f, 2.0f, -1.0f,
    };

    Tensor gate_up = make_device_float_tensor({2, 6}, gate_up_values, allocator);
    Tensor output = Tensor::zeros({2, 3}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);

    silu_and_mul(gate_up, output);

    const std::vector<float> actual = copy_device_floats(output, allocator);
    std::vector<float> expected(actual.size(), 0.0f);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            const float gate = gate_up_values[row * 6 + col];
            const float up = gate_up_values[row * 6 + col + 3];
            expected[row * 3 + col] = silu_ref(gate) * up;
        }
    }

    for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-6f);
    }
}

TEST(TensorTest, StoreKvCacheWritesSlotsAndSkipsPadding) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    const std::vector<float> key_values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    const std::vector<float> value_values = {
        -1.0f, -2.0f, -3.0f, -4.0f,
        -5.0f, -6.0f, -7.0f, -8.0f,
        -9.0f, -10.0f, -11.0f, -12.0f,
    };
    const std::vector<int32_t> slots = {2, -1, 5};

    Tensor key = make_device_float_tensor({3, 1, 4}, key_values, allocator);
    Tensor value = make_device_float_tensor({3, 1, 4}, value_values, allocator);
    Tensor k_cache = Tensor::zeros({2, 3, 1, 4}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    Tensor v_cache = Tensor::zeros({2, 3, 1, 4}, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    Tensor slot_mapping = make_device_int_tensor({3}, slots, allocator);

    store_kvcache(key, value, k_cache, v_cache, slot_mapping);

    const std::vector<float> actual_k = copy_device_floats(k_cache, allocator);
    const std::vector<float> actual_v = copy_device_floats(v_cache, allocator);
    std::vector<float> expected_k(24, 0.0f);
    std::vector<float> expected_v(24, 0.0f);
    std::copy(key_values.begin(), key_values.begin() + 4, expected_k.begin() + 8);
    std::copy(value_values.begin(), value_values.begin() + 4, expected_v.begin() + 8);
    std::copy(key_values.begin() + 8, key_values.begin() + 12, expected_k.begin() + 20);
    std::copy(value_values.begin() + 8, value_values.begin() + 12, expected_v.begin() + 20);

    for (size_t index = 0; index < actual_k.size(); ++index) {
        EXPECT_NEAR(actual_k[index], expected_k[index], 1e-6f);
        EXPECT_NEAR(actual_v[index], expected_v[index], 1e-6f);
    }
}

} // namespace
} // namespace nano_vllm