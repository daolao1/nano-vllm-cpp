#include "layers/embed_head.h"
#include "layers/sampler.h"
#include "utils/context.h"
#include "utils/cuda_allocator.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
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

std::vector<float> make_host_floats(size_t count, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>((index % 97) - 48) * scale;
    }
    return values;
}

std::vector<int32_t> make_host_ids(size_t count, int32_t vocab_size) {
    std::vector<int32_t> values(count, 0);
    for (size_t index = 0; index < count; ++index) {
        values[index] = static_cast<int32_t>(index % static_cast<size_t>(vocab_size));
    }
    return values;
}

std::vector<int32_t> make_cu_seqlens(int batch_size, int seq_len) {
    std::vector<int32_t> cu(static_cast<size_t>(batch_size + 1), 0);
    for (int batch = 0; batch <= batch_size; ++batch) {
        cu[static_cast<size_t>(batch)] = batch * seq_len;
    }
    return cu;
}

template <typename Fn>
double benchmark_ms(const std::string& name,
                    int warmup,
                    int iterations,
                    CudaAllocator& allocator,
                    Fn&& fn) {
    for (int iter = 0; iter < warmup; ++iter) {
        fn();
        allocator.synchronize_stream(nullptr);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fn();
        allocator.synchronize_stream(nullptr);
    }
    const auto end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double avg_ms = total_ms / static_cast<double>(iterations);
    std::cout << std::left << std::setw(24) << name
              << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms << '\n';
    return avg_ms;
}

} // namespace

} // namespace nano_vllm

int main() {
    using namespace nano_vllm;

    if (!has_cuda_device()) {
        std::cout << "No CUDA device available, benchmark skipped.\n";
        return 0;
    }

    CudaAllocator& allocator = CudaAllocator::instance();
    constexpr int kVocabSize = 8192;
    constexpr int kHiddenDim = 1024;
    constexpr int kBatchSize = 32;
    constexpr int kSeqLen = 64;
    constexpr int kWarmup = 20;
    constexpr int kIterations = 100;

    std::cout << "Benchmark config: vocab=" << kVocabSize
              << ", hidden=" << kHiddenDim
              << ", batch=" << kBatchSize
              << ", seq=" << kSeqLen
              << ", iterations=" << kIterations << "\n";

    VocabParallelEmbedding embedding(kVocabSize,
                                     kHiddenDim,
                                     ScalarType::kFloat32,
                                     Device{DeviceType::kCUDA, 0},
                                     allocator);
    ParallelLMHead lm_head(kVocabSize,
                           kHiddenDim,
                           ScalarType::kFloat32,
                           Device{DeviceType::kCUDA, 0},
                           allocator);
    lm_head.set_weight(embedding.weight());
    Sampler sampler(123);

    const std::vector<float> weight_values = make_host_floats(static_cast<size_t>(kVocabSize) * kHiddenDim, 0.001f);
    allocator.copy_to_device_async(embedding.weight().data(), weight_values.data(), embedding.weight().nbytes(), nullptr);
    allocator.synchronize_stream(nullptr);

    const Tensor input_ids = make_device_int_tensor({kBatchSize, kSeqLen}, make_host_ids(kBatchSize * kSeqLen, kVocabSize), allocator);
    const Tensor hidden_prefill = make_device_float_tensor({kBatchSize * kSeqLen, kHiddenDim},
                                                           make_host_floats(static_cast<size_t>(kBatchSize * kSeqLen) * kHiddenDim, 0.01f),
                                                           allocator);
    const Tensor hidden_decode = make_device_float_tensor({kBatchSize, kHiddenDim},
                                                          make_host_floats(static_cast<size_t>(kBatchSize) * kHiddenDim, 0.02f),
                                                          allocator);
    const Tensor logits = make_device_float_tensor({kBatchSize, kVocabSize},
                                                   make_host_floats(static_cast<size_t>(kBatchSize) * kVocabSize, 0.001f),
                                                   allocator);
    const Tensor temperatures = make_device_float_tensor({kBatchSize}, std::vector<float>(kBatchSize, 1.0f), allocator);
    const Tensor top_ks = make_device_int_tensor({kBatchSize}, std::vector<int32_t>(kBatchSize, -1), allocator);
    const Tensor top_ps = make_device_float_tensor({kBatchSize}, std::vector<float>(kBatchSize, 1.0f), allocator);
    const Tensor cu_seqlens_q = make_device_int_tensor({kBatchSize + 1}, make_cu_seqlens(kBatchSize, kSeqLen), allocator);

    Tensor sink_tensor;
    std::vector<int32_t> sink_tokens;

    benchmark_ms("embedding_prefill", kWarmup, kIterations, allocator, [&] {
        sink_tensor = embedding.forward(input_ids, allocator);
    });

    benchmark_ms("lm_head_prefill", kWarmup, kIterations, allocator, [&] {
        set_context(true, cu_seqlens_q, Tensor(), kSeqLen, kSeqLen, Tensor(), Tensor(), Tensor());
        sink_tensor = lm_head.forward(hidden_prefill, allocator);
        reset_context();
    });

    benchmark_ms("lm_head_decode", kWarmup, kIterations, allocator, [&] {
        reset_context();
        sink_tensor = lm_head.forward(hidden_decode, allocator);
    });

    benchmark_ms("sampler_decode", kWarmup, kIterations, allocator, [&] {
        sink_tokens = sampler.forward(logits, temperatures, top_ks, top_ps,
                                      Tensor(), Tensor(), Tensor(), allocator);
    });

    std::cout << "sink_tensor_numel=" << sink_tensor.numel()
              << ", sink_tokens=" << sink_tokens.size() << "\n";
    return 0;
}