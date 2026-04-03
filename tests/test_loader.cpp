#include "layers/embed_head.h"
#include "layers/linear.h"
#include "utils/cuda_allocator.h"
#include "utils/loader.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <vector>

namespace nano_vllm {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

bool has_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

struct TensorSpec {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<std::byte> bytes;
};

std::vector<std::byte> make_bytes(const std::vector<float>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<std::byte> make_bytes(const std::vector<int64_t>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

uint16_t float_to_bfloat16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

std::vector<std::byte> make_bfloat16_bytes(const std::vector<float>& values) {
    std::vector<uint16_t> bf16(values.size(), 0);
    for (size_t index = 0; index < values.size(); ++index) {
        bf16[index] = float_to_bfloat16_bits(values[index]);
    }
    std::vector<std::byte> bytes(bf16.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), bf16.data(), bytes.size());
    return bytes;
}

fs::path write_safetensors_file(const std::vector<TensorSpec>& specs) {
    static std::atomic<int> counter{0};
    const fs::path path = fs::temp_directory_path() /
                          ("nano_vllm_loader_" + std::to_string(::getpid()) + "_" +
                           std::to_string(counter.fetch_add(1)) + ".safetensors");

    json header = json::object();
    size_t offset = 0;
    for (const TensorSpec& spec : specs) {
        header[spec.name] = {
            {"dtype", spec.dtype},
            {"shape", spec.shape},
            {"data_offsets", {offset, offset + spec.bytes.size()}},
        };
        offset += spec.bytes.size();
    }

    const std::string header_text = header.dump();
    const uint64_t header_size = static_cast<uint64_t>(header_text.size());
    std::ofstream out(path, std::ios::binary);
    for (int shift = 0; shift < 8; ++shift) {
        const char byte = static_cast<char>((header_size >> (shift * 8)) & 0xffu);
        out.write(&byte, 1);
    }
    out.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const TensorSpec& spec : specs) {
        out.write(reinterpret_cast<const char*>(spec.bytes.data()), static_cast<std::streamsize>(spec.bytes.size()));
    }
    out.close();
    return path;
}

Tensor make_borrowed_cpu_float_tensor(const std::vector<int64_t>& sizes, std::vector<float>& values) {
    auto memory = Memory::make_borrowed(values.data(), values.size() * sizeof(float), Device{DeviceType::kCPU, 0});
    return Tensor::from_memory(memory, sizes, ScalarType::kFloat32);
}

std::vector<float> copy_device_floats(const Tensor& tensor, CudaAllocator& allocator) {
    std::vector<float> values(tensor.numel(), 0.0f);
    allocator.copy_to_host_async(values.data(), tensor.data(), tensor.nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);
    return values;
}

TEST(LoaderTest, SafeTensorsParserExposesCpuTensorViews) {
    const fs::path path = write_safetensors_file({
        TensorSpec{"linear.weight", "F32", {2, 2}, make_bytes(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f})},
        TensorSpec{"linear.bias", "I64", {2}, make_bytes(std::vector<int64_t>{7, 9})},
    });

    const SafeTensorsFile file = SafeTensorsFile::open(path.string());
    ASSERT_TRUE(file.contains("linear.weight"));
    ASSERT_TRUE(file.contains("linear.bias"));
    EXPECT_EQ(file.keys().size(), 2u);

    const Tensor& weight = file.tensor("linear.weight");
    const Tensor& bias = file.tensor("linear.bias");
    EXPECT_EQ(weight.device().type, DeviceType::kCPU);
    EXPECT_EQ(weight.dtype(), ScalarType::kFloat32);
    EXPECT_EQ(weight.sizes(), (std::vector<int64_t>{2, 2}));
    EXPECT_FLOAT_EQ(weight.data_as<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(weight.data_as<float>()[3], 4.0f);
    EXPECT_EQ(bias.dtype(), ScalarType::kInt64);
    EXPECT_EQ(bias.data_as<int64_t>()[0], 7);
    EXPECT_EQ(bias.data_as<int64_t>()[1], 9);

    fs::remove(path);
}

TEST(LoaderTest, BaseLinearWeightLoadersCopyFromCpuTensors) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    ColumnParallelLinear column(2, 3, true, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    std::vector<float> column_weight_host = {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 3.0f};
    std::vector<float> column_bias_host = {0.5f, -1.0f, 2.0f};
    column.weight_loader(make_borrowed_cpu_float_tensor({3, 2}, column_weight_host), allocator);
    column.bias_loader(make_borrowed_cpu_float_tensor({3}, column_bias_host), allocator);

    RowParallelLinear row(2, 3, true, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    std::vector<float> row_weight_host = {1.0f, -1.0f, 0.0f, 2.0f, 3.0f, 1.0f};
    std::vector<float> row_bias_host = {0.0f, 1.0f, -0.5f};
    row.weight_loader(make_borrowed_cpu_float_tensor({3, 2}, row_weight_host), allocator);
    row.bias_loader(make_borrowed_cpu_float_tensor({3}, row_bias_host), allocator);

    const std::vector<float> column_weight = copy_device_floats(column.weight(), allocator);
    const std::vector<float> column_bias = copy_device_floats(column.bias(), allocator);
    const std::vector<float> row_weight = copy_device_floats(row.weight(), allocator);
    const std::vector<float> row_bias = copy_device_floats(row.bias(), allocator);

    EXPECT_EQ(column_weight, (std::vector<float>{1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 3.0f}));
    EXPECT_EQ(column_bias, (std::vector<float>{0.5f, -1.0f, 2.0f}));
    EXPECT_EQ(row_weight, (std::vector<float>{1.0f, -1.0f, 0.0f, 2.0f, 3.0f, 1.0f}));
    EXPECT_EQ(row_bias, (std::vector<float>{0.0f, 1.0f, -0.5f}));
}

TEST(LoaderTest, TensorParallelWeightLoadersSelectExpectedShards) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    CudaAllocator& allocator = CudaAllocator::instance();

    RowParallelLinear row(4, 3, true, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 1);
    std::vector<float> row_weight_host = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    std::vector<float> row_bias_host = {0.5f, -1.0f, 2.5f};
    row.weight_loader(make_borrowed_cpu_float_tensor({3, 4}, row_weight_host), allocator);
    row.bias_loader(make_borrowed_cpu_float_tensor({3}, row_bias_host), allocator);

    VocabParallelEmbedding embedding(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 1);
    ParallelLMHead lm_head(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator, 2, 1);
    std::vector<float> vocab_weight_host = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        2.0f, 2.0f,
        -1.0f, 3.0f,
    };
    embedding.weight_loader(make_borrowed_cpu_float_tensor({4, 2}, vocab_weight_host), allocator);
    lm_head.weight_loader(make_borrowed_cpu_float_tensor({4, 2}, vocab_weight_host), allocator);

    EXPECT_EQ(copy_device_floats(row.weight(), allocator), (std::vector<float>{3.0f, 4.0f, 7.0f, 8.0f, 11.0f, 12.0f}));
    EXPECT_EQ(copy_device_floats(row.bias(), allocator), row_bias_host);
    EXPECT_EQ(copy_device_floats(embedding.weight(), allocator), (std::vector<float>{2.0f, 2.0f, -1.0f, 3.0f}));
    EXPECT_EQ(copy_device_floats(lm_head.weight(), allocator), (std::vector<float>{2.0f, 2.0f, -1.0f, 3.0f}));
}

TEST(LoaderTest, ParameterRegistryLoadsPackedBfloat16Weights) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device available";
    }

    const fs::path path = write_safetensors_file({
        TensorSpec{"model.layers.0.self_attn.q_proj.weight", "BF16", {2, 2}, make_bfloat16_bytes({1.0f, 2.0f, 3.0f, 4.0f})},
        TensorSpec{"model.layers.0.self_attn.k_proj.weight", "BF16", {1, 2}, make_bfloat16_bytes({5.0f, 6.0f})},
        TensorSpec{"model.layers.0.self_attn.v_proj.weight", "BF16", {1, 2}, make_bfloat16_bytes({7.0f, 8.0f})},
        TensorSpec{"model.layers.0.mlp.gate_proj.weight", "BF16", {1, 2}, make_bfloat16_bytes({9.0f, 10.0f})},
        TensorSpec{"model.layers.0.mlp.up_proj.weight", "BF16", {2, 2}, make_bfloat16_bytes({11.0f, 12.0f, 13.0f, 14.0f})},
        TensorSpec{"model.embed_tokens.weight", "F32", {4, 2}, make_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, -1.0f, 3.0f})},
    });

    CudaAllocator& allocator = CudaAllocator::instance();
    QKVParallelLinear qkv(2, 1, 2, 1, false, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    MergedColumnParallelLinear merged(2, {1, 2}, false, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    VocabParallelEmbedding embedding(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);
    ParallelLMHead lm_head(4, 2, ScalarType::kFloat32, Device{DeviceType::kCUDA, 0}, allocator);

    ParameterRegistry registry;
    registry.register_parameter("model.layers.0.self_attn.qkv_proj.weight",
                                [&](const Tensor& source,
                                    std::optional<int> shard_id,
                                    DeviceAllocator& reg_allocator,
                                    cudaStream_t stream) {
                                    ASSERT_TRUE(shard_id.has_value());
                                    qkv.weight_loader(source, *shard_id, reg_allocator, stream);
                                });
    registry.register_parameter("model.layers.0.mlp.gate_up_proj.weight",
                                [&](const Tensor& source,
                                    std::optional<int> shard_id,
                                    DeviceAllocator& reg_allocator,
                                    cudaStream_t stream) {
                                    ASSERT_TRUE(shard_id.has_value());
                                    merged.weight_loader(source, *shard_id, reg_allocator, stream);
                                });
    registry.register_parameter("model.embed_tokens.weight",
                                [&](const Tensor& source,
                                    std::optional<int>,
                                    DeviceAllocator& reg_allocator,
                                    cudaStream_t stream) {
                                    embedding.weight_loader(source, reg_allocator, stream);
                                    lm_head.weight_loader(source, reg_allocator, stream);
                                });

    registry.load(path.string(),
                  {
                      PackedModuleMapping{"q_proj", "qkv_proj", 0},
                      PackedModuleMapping{"k_proj", "qkv_proj", 1},
                      PackedModuleMapping{"v_proj", "qkv_proj", 2},
                      PackedModuleMapping{"gate_proj", "gate_up_proj", 0},
                      PackedModuleMapping{"up_proj", "gate_up_proj", 1},
                  },
                  allocator);

    EXPECT_EQ(copy_device_floats(qkv.weight(), allocator),
              (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}));
    EXPECT_EQ(copy_device_floats(merged.weight(), allocator),
              (std::vector<float>{9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f}));
    EXPECT_EQ(copy_device_floats(embedding.weight(), allocator),
              (std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, -1.0f, 3.0f}));
    EXPECT_EQ(copy_device_floats(lm_head.weight(), allocator),
              (std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, -1.0f, 3.0f}));

    fs::remove(path);
}

} // namespace
} // namespace nano_vllm