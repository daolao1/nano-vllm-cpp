#pragma once

#include "core/config.h"
#include "engine/sequence.h"
#include "layers/sampler.h"
#include "models/transformer.h"
#include "utils/cuda_allocator.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace nano_vllm {

/// Captures and replays CUDA graphs for decode steps at fixed batch sizes.
class CudaGraphRunner {
public:
    CudaGraphRunner() = default;
    ~CudaGraphRunner();

    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    /// Capture graphs for a set of batch sizes. Must be called after model warmup.
    void capture(TransformerForCausalLM& model,
                 DeviceAllocator& allocator,
                 const Config& config,
                 ScalarType dtype,
                 Device device,
                 cudaStream_t stream = nullptr);

    /// Check if a graph is available for the given batch size.
    bool available(int batch_size) const;

    /// Select the smallest captured batch size >= bs. Returns 0 if none fits.
    int select_bs(int bs) const;

    /// Run a decode step using the captured graph.
    /// Copies inputs into the pre-allocated buffers, replays the graph, returns logits[:bs].
    Tensor run(int actual_bs,
               const Tensor& input_ids,
               const Tensor& positions,
               const Tensor& slot_mapping,
               const Tensor& context_lens,
               const Tensor& block_tables,
               DeviceAllocator& allocator,
               cudaStream_t stream = nullptr);

private:
    struct CapturedGraph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        int bs = 0;
        // Pre-allocated device buffers (fixed pointers).
        Tensor input_ids;
        Tensor positions;
        Tensor slot_mapping;
        Tensor context_lens;
        Tensor block_tables;
        Tensor output;  // hidden states from model forward
    };

    std::vector<int> captured_bs_;          // sorted batch sizes
    std::map<int, CapturedGraph> graphs_;
    int max_num_blocks_ = 0;
    int hidden_size_ = 0;
    void* scratch_buffer_ = nullptr;
    size_t scratch_size_ = 0;
};

class ModelRunner {
public:
    struct ModelInputs {
        Tensor input_ids;
        Tensor positions;
    };

    struct SampleInputs {
        Tensor temperatures;
        Tensor top_ks;
        Tensor top_ps;
        Tensor penalty_token_ids;
        Tensor penalty_token_counts;
        Tensor penalties;
    };

    ModelRunner(const Config& config,
                DeviceAllocator& allocator,
                int rank = 0,
                ScalarType dtype = ScalarType::kFloat32,
                Device device = Device{DeviceType::kCUDA, 0});

    void warmup_model(cudaStream_t stream = nullptr);
    void allocate_kv_cache();
    void capture_cudagraph(cudaStream_t stream = nullptr);

    ModelInputs prepare_prefill(const std::vector<Sequence*>& seqs,
                                cudaStream_t stream = nullptr);
    ModelInputs prepare_decode(const std::vector<Sequence*>& seqs,
                               cudaStream_t stream = nullptr);
    SampleInputs prepare_sample(const std::vector<Sequence*>& seqs,
                               cudaStream_t stream = nullptr);

    Tensor run_model(const Tensor& input_ids,
                     const Tensor& positions,
                     bool is_prefill,
                     cudaStream_t stream = nullptr);
    std::vector<int32_t> run(const std::vector<Sequence*>& seqs,
                             bool is_prefill,
                             cudaStream_t stream = nullptr);

    int rank() const { return rank_; }
    ScalarType dtype() const { return dtype_; }
    Device device() const { return device_; }
    int num_kvcache_blocks() const { return num_kvcache_blocks_; }

    const Config& config() const { return config_; }
    const Tensor& kv_cache() const { return kv_cache_; }
    TransformerForCausalLM& model() { return model_; }
    const TransformerForCausalLM& model() const { return model_; }
    Sampler& sampler() { return sampler_; }
    const Sampler& sampler() const { return sampler_; }

private:
    Tensor build_block_tables(const std::vector<Sequence*>& seqs,
                              cudaStream_t stream = nullptr);
    Tensor make_device_int_tensor(const std::vector<int64_t>& sizes,
                                  const std::vector<int32_t>& values,
                                  cudaStream_t stream = nullptr);
    Tensor make_device_float_tensor(const std::vector<int64_t>& sizes,
                                    const std::vector<float>& values,
                                    cudaStream_t stream = nullptr);
    // Bump-allocate `bytes` from the pinned staging buffer (auto-grows).
    // Valid until reset_pinned_staging() is called (at the start of run()).
    void* pinned_alloc(size_t bytes);
    void reset_pinned_staging();

    Config config_;
    int rank_ = 0;
    Device device_{};
    ScalarType dtype_ = ScalarType::kFloat32;
    DeviceAllocator& allocator_;
    TransformerForCausalLM model_;
    Sampler sampler_;
    Tensor kv_cache_;
    Tensor kv_scale_;  // fp16 scale tensor for int8 KV cache
    int num_kvcache_blocks_ = 0;
    CudaGraphRunner cuda_graph_runner_;

    // Pinned host staging buffer: per-step bump allocator backing truly async H2D.
    std::unique_ptr<PinnedHostBuffer> pinned_staging_;
    size_t pinned_offset_ = 0;
};

} // namespace nano_vllm