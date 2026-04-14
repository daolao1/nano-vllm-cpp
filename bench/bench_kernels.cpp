// bench_kernels.cpp
//
// Micro-benchmark for individual kernels using cudaEvent_t timing.
// Reports per-kernel latency + effective HBM bandwidth + % of theoretical peak.
// Used as poor-man's ncu when GPU performance counters are unavailable
// (typical on WSL2 where ncu requires Windows-side registry permission).
//
// Theoretical peak HBM bandwidth for RTX 4060 Laptop: 272 GB/s.

#include "kernels/rms_norm.h"
#include "kernels/rotary.h"
#include "kernels/split_qkv.h"
#include "kernels/kvcache.h"
#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr double kPeakBandwidthGBs = 272.0;  // RTX 4060 Laptop HBM peak
constexpr int kWarmup = 50;
constexpr int kIters = 200;

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Result {
    std::string name;
    int64_t bytes;
    double median_us;
    double bw_gbs;
    double pct_peak;
};

void print_header() {
    std::printf("\n%-46s %10s %10s %10s %8s\n",
                "kernel", "bytes", "med (us)", "BW GB/s", "%peak");
    std::printf("%s\n", std::string(88, '-').c_str());
}

void print_row(const Result& r) {
    std::printf("%-46s %10lld %10.2f %10.1f %7.1f%%\n",
                r.name.c_str(),
                static_cast<long long>(r.bytes),
                r.median_us,
                r.bw_gbs,
                r.pct_peak);
}

template <typename Fn>
Result time_kernel(const std::string& name,
                   int64_t bytes_per_call,
                   Fn&& fn,
                   cudaStream_t stream) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < kWarmup; ++i) fn();
    cudaStreamSynchronize(stream);

    // Measure
    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back(static_cast<double>(ms) * 1000.0);  // us
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    Result r;
    r.name = name;
    r.bytes = bytes_per_call;
    r.median_us = median(samples);
    double bw = static_cast<double>(bytes_per_call) / (r.median_us * 1e-6) / 1e9;
    r.bw_gbs = bw;
    r.pct_peak = bw / kPeakBandwidthGBs * 100.0;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace nano_vllm;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("Device: %s, SMs=%d, peak HBM=%.0f GB/s (assumed)\n",
                prop.name, prop.multiProcessorCount, kPeakBandwidthGBs);

    CudaAllocator& alloc = CudaAllocator::instance();
    Device dev{DeviceType::kCUDA, 0};
    cudaStream_t stream = nullptr;

    constexpr int hidden = 896;
    constexpr int num_q_heads = 14;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int q_size = num_q_heads * head_dim;
    constexpr int kv_size = num_kv_heads * head_dim;
    constexpr int qkv_size = q_size + 2 * kv_size;
    constexpr int num_slots = 4096;

    // Sweep rows to test launch-bound vs memory-bound.
    std::vector<int> batch_sweep = {1, 4, 16, 64, 256, 1024};
    if (argc > 1) {
        batch_sweep.clear();
        batch_sweep.push_back(std::atoi(argv[1]));
    }

    for (int rows : batch_sweep) {
        std::printf("\n=== rows=%d ===\n", rows);
        std::printf("Workload: hidden=%d, num_q=%d, num_kv=%d, head_dim=%d, qkv_cols=%d\n",
                    hidden, num_q_heads, num_kv_heads, head_dim, qkv_size);
        print_header();

    // ---- 1. RMSNorm on hidden_states [rows, hidden] bf16 ----
    {
        Tensor input  = Tensor::empty({rows, hidden}, ScalarType::kBFloat16, dev, alloc);
        Tensor weight = Tensor::empty({hidden},       ScalarType::kBFloat16, dev, alloc);
        Tensor output = Tensor::empty({rows, hidden}, ScalarType::kBFloat16, dev, alloc);
        cudaMemsetAsync(input.data(),  0, input.nbytes(),  stream);
        cudaMemsetAsync(weight.data(), 0, weight.nbytes(), stream);

        // bytes: read input + read weight + write output
        int64_t bytes = input.nbytes() + weight.nbytes() + output.nbytes();

        char label[64]; std::snprintf(label, sizeof(label), "rms_norm bf16 [%dx%d]", rows, hidden);
        auto r = time_kernel(label, bytes,
            [&]{ rms_norm(input, weight, 1e-6f, output, stream); },
            stream);
        print_row(r);
    }

    // ---- 2. add_rms_norm (fused residual+norm) ----
    {
        Tensor input    = Tensor::empty({rows, hidden}, ScalarType::kBFloat16, dev, alloc);
        Tensor residual = Tensor::empty({rows, hidden}, ScalarType::kBFloat16, dev, alloc);
        Tensor weight   = Tensor::empty({hidden},       ScalarType::kBFloat16, dev, alloc);
        Tensor output   = Tensor::empty({rows, hidden}, ScalarType::kBFloat16, dev, alloc);
        Tensor res_out  = Tensor::empty({rows, hidden}, ScalarType::kBFloat16, dev, alloc);
        cudaMemsetAsync(input.data(),    0, input.nbytes(),    stream);
        cudaMemsetAsync(residual.data(), 0, residual.nbytes(), stream);
        cudaMemsetAsync(weight.data(),   0, weight.nbytes(),   stream);

        int64_t bytes = input.nbytes() + residual.nbytes() + weight.nbytes()
                      + output.nbytes() + res_out.nbytes();

        char label[64]; std::snprintf(label, sizeof(label), "add_rms_norm bf16 [%dx%d]", rows, hidden);
        auto r = time_kernel(label, bytes,
            [&]{ add_rms_norm(input, residual, weight, 1e-6f, output, res_out, stream); },
            stream);
        print_row(r);
    }

    // ---- 3. rotary_embedding on Q and K ----
    {
        Tensor query     = Tensor::empty({rows, num_q_heads, head_dim},  ScalarType::kBFloat16, dev, alloc);
        Tensor key       = Tensor::empty({rows, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor positions = Tensor::empty({rows}, ScalarType::kInt32, dev, alloc);
        Tensor cos_sin   = Tensor::empty({4096, head_dim}, ScalarType::kFloat32, dev, alloc);
        cudaMemsetAsync(query.data(),     0, query.nbytes(),     stream);
        cudaMemsetAsync(key.data(),       0, key.nbytes(),       stream);
        cudaMemsetAsync(positions.data(), 0, positions.nbytes(), stream);
        cudaMemsetAsync(cos_sin.data(),   0, cos_sin.nbytes(),   stream);

        // Read+write Q and K + read small cos_sin slice
        int64_t bytes = 2 * query.nbytes() + 2 * key.nbytes() + rows * head_dim * 4;

        char label[64]; std::snprintf(label, sizeof(label), "rotary_embedding bf16 [%dx%dx%d]", rows, num_q_heads+num_kv_heads, head_dim);
        auto r = time_kernel(label, bytes,
            [&]{ rotary_embedding(positions, cos_sin, query, key, stream); },
            stream);
        print_row(r);
    }

    // ---- 4. split_qkv ----
    {
        Tensor qkv = Tensor::empty({rows, qkv_size}, ScalarType::kBFloat16, dev, alloc);
        Tensor q   = Tensor::empty({rows, q_size},   ScalarType::kBFloat16, dev, alloc);
        Tensor k   = Tensor::empty({rows, kv_size},  ScalarType::kBFloat16, dev, alloc);
        Tensor v   = Tensor::empty({rows, kv_size},  ScalarType::kBFloat16, dev, alloc);
        cudaMemsetAsync(qkv.data(), 0, qkv.nbytes(), stream);

        int64_t bytes = qkv.nbytes() + q.nbytes() + k.nbytes() + v.nbytes();

        char label[64]; std::snprintf(label, sizeof(label), "split_qkv bf16 [%dx%d]", rows, qkv_size);
        auto r = time_kernel(label, bytes,
            [&]{ split_qkv(qkv, q_size, kv_size, q, k, v, stream); },
            stream);
        print_row(r);
    }

    // ---- 5. store_kvcache (vectorized path: row_bytes = 128*2 = 256 = 16 int4) ----
    {
        // Shape: [rows, num_kv_heads, head_dim] bf16
        Tensor key   = Tensor::empty({rows, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor value = Tensor::empty({rows, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor k_cache = Tensor::empty({num_slots, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor v_cache = Tensor::empty({num_slots, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        std::vector<int32_t> slots_h(rows);
        for (int i = 0; i < rows; ++i) slots_h[i] = i;
        Tensor slot_mapping = Tensor::empty({rows}, ScalarType::kInt32, dev, alloc);
        cudaMemcpyAsync(slot_mapping.data(), slots_h.data(),
                        slots_h.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(key.data(),   0, key.nbytes(),   stream);
        cudaMemsetAsync(value.data(), 0, value.nbytes(), stream);

        int64_t bytes = key.nbytes() + value.nbytes()
                      + rows * num_kv_heads * head_dim * 2 * 2;  // K+V cache writes

        char label[64]; std::snprintf(label, sizeof(label), "store_kvcache bf16 (vec) [%d]", rows);
        auto r = time_kernel(label, bytes,
            [&]{ store_kvcache(key, value, k_cache, v_cache, slot_mapping, stream); },
            stream);
        print_row(r);
    }

    // ---- 6. fused_split_norm_rope_store (the 5-kernel fusion) ----
    {
        Tensor qkv = Tensor::empty({rows, qkv_size}, ScalarType::kBFloat16, dev, alloc);
        Tensor positions = Tensor::empty({rows}, ScalarType::kInt32, dev, alloc);
        Tensor q_norm_w = Tensor::empty({head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor k_norm_w = Tensor::empty({head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor cos_sin = Tensor::empty({4096, head_dim}, ScalarType::kFloat32, dev, alloc);
        std::vector<int32_t> slots_h(rows);
        for (int i = 0; i < rows; ++i) slots_h[i] = i;
        Tensor slot_mapping = Tensor::empty({rows}, ScalarType::kInt32, dev, alloc);
        cudaMemcpyAsync(slot_mapping.data(), slots_h.data(),
                        slots_h.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        Tensor k_cache = Tensor::empty({num_slots, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor v_cache = Tensor::empty({num_slots, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor q_out = Tensor::empty({rows, num_q_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        cudaMemsetAsync(qkv.data(), 0, qkv.nbytes(), stream);
        cudaMemsetAsync(positions.data(), 0, positions.nbytes(), stream);
        cudaMemsetAsync(q_norm_w.data(), 0, q_norm_w.nbytes(), stream);
        cudaMemsetAsync(k_norm_w.data(), 0, k_norm_w.nbytes(), stream);
        cudaMemsetAsync(cos_sin.data(), 0, cos_sin.nbytes(), stream);

        // bytes: read qkv + tiny constants + write q + write k_cache slot + write v_cache slot
        int64_t bytes = qkv.nbytes()
                      + q_out.nbytes()
                      + rows * (num_kv_heads * head_dim) * 2 * 2  // k+v cache writes
                      + rows * head_dim * 4;                       // cos_sin lookup
        char label[64]; std::snprintf(label, sizeof(label), "fused_split_norm_rope_store bf16 [%d]", rows);
        auto r = time_kernel(label, bytes,
            [&]{ fused_split_norm_rope_store(
                    qkv, positions, q_norm_w, k_norm_w, 1e-6f,
                    cos_sin, slot_mapping, k_cache, v_cache,
                    num_q_heads, num_kv_heads, head_dim, q_out, stream); },
            stream);
        print_row(r);
    }

    // ---- 7. separate kernels (split + q_norm + k_norm + rotary + store_kvcache) ----
    //   This is the prefill code path; comparing this row vs the fused row above
    //   answers "would prefill be faster if we used the fused kernel?"
    {
        Tensor qkv = Tensor::empty({rows, qkv_size}, ScalarType::kBFloat16, dev, alloc);
        Tensor positions = Tensor::empty({rows}, ScalarType::kInt32, dev, alloc);
        Tensor q_norm_w = Tensor::empty({head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor k_norm_w = Tensor::empty({head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor cos_sin = Tensor::empty({4096, head_dim}, ScalarType::kFloat32, dev, alloc);
        std::vector<int32_t> slots_h(rows);
        for (int i = 0; i < rows; ++i) slots_h[i] = i;
        Tensor slot_mapping = Tensor::empty({rows}, ScalarType::kInt32, dev, alloc);
        cudaMemcpyAsync(slot_mapping.data(), slots_h.data(),
                        slots_h.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        Tensor k_cache = Tensor::empty({num_slots, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor v_cache = Tensor::empty({num_slots, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        // Intermediate tensors that the separate path materializes (decode skips these via fusion).
        Tensor q   = Tensor::empty({rows, num_q_heads, head_dim},  ScalarType::kBFloat16, dev, alloc);
        Tensor k   = Tensor::empty({rows, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor v   = Tensor::empty({rows, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        Tensor q_n = Tensor::empty({rows, num_q_heads, head_dim},  ScalarType::kBFloat16, dev, alloc);
        Tensor k_n = Tensor::empty({rows, num_kv_heads, head_dim}, ScalarType::kBFloat16, dev, alloc);
        cudaMemsetAsync(qkv.data(), 0, qkv.nbytes(), stream);
        cudaMemsetAsync(positions.data(), 0, positions.nbytes(), stream);
        cudaMemsetAsync(q_norm_w.data(), 0, q_norm_w.nbytes(), stream);
        cudaMemsetAsync(k_norm_w.data(), 0, k_norm_w.nbytes(), stream);
        cudaMemsetAsync(cos_sin.data(), 0, cos_sin.nbytes(), stream);

        // Reshape Q/K to [rows*num_heads, head_dim] for per-head RMSNorm
        Tensor q_flat   = q.reshape({rows * num_q_heads,  head_dim});
        Tensor k_flat   = k.reshape({rows * num_kv_heads, head_dim});
        Tensor qn_flat  = q_n.reshape({rows * num_q_heads,  head_dim});
        Tensor kn_flat  = k_n.reshape({rows * num_kv_heads, head_dim});
        Tensor v_flat   = v.reshape({rows, kv_size});

        // Same logical bytes as fused (qkv read + q write + 2 cache writes), kept for fair %peak compare
        int64_t bytes = qkv.nbytes()
                      + q.nbytes()
                      + rows * (num_kv_heads * head_dim) * 2 * 2
                      + rows * head_dim * 4;
        char label[64]; std::snprintf(label, sizeof(label), "SEPARATE 5-kernel chain bf16 [%d]", rows);
        auto r = time_kernel(label, bytes,
            [&]{
                split_qkv(qkv, q_size, kv_size, q_flat, k_flat, v_flat, stream);
                rms_norm(q_flat, q_norm_w, 1e-6f, qn_flat, stream);
                rms_norm(k_flat, k_norm_w, 1e-6f, kn_flat, stream);
                rotary_embedding(positions, cos_sin, q_n, k_n, stream);
                store_kvcache(k_n, v, k_cache, v_cache, slot_mapping, stream);
            },
            stream);
        print_row(r);
    }
    } // end rows loop

    std::printf("\nNotes:\n");
    std::printf("  - 'bytes' = read+write HBM bytes (lower bound; actual may include re-reads)\n");
    std::printf("  - %%peak  = effective_BW / %.0f GB/s (RTX 4060 Laptop HBM peak)\n", kPeakBandwidthGBs);
    std::printf("  - >70%% peak = memory-bound and well-tuned\n");
    std::printf("  - <30%% peak = either compute-bound or launch-bound\n");
    return 0;
}
