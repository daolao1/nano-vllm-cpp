#include "engine/model_runner.h"

#include "utils/context.h"
#include "utils/cuda_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nano_vllm {
namespace {

class ScopedContextReset {
public:
    ~ScopedContextReset() {
        reset_context();
    }
};

void set_current_device(Device device) {
    if (device.type != DeviceType::kCUDA) {
        throw std::invalid_argument("ModelRunner requires CUDA execution");
    }
    throw_if_cuda_error(cudaSetDevice(device.index), "cudaSetDevice");
}

int resolve_head_dim(const HFConfig& config) {
    if (config.head_dim > 0) {
        return config.head_dim;
    }
    if (config.num_attention_heads <= 0 || config.hidden_size % config.num_attention_heads != 0) {
        throw std::invalid_argument("hf_config must define a valid head_dim or hidden_size / num_attention_heads ratio");
    }
    return config.hidden_size / config.num_attention_heads;
}

int64_t ceil_div(int64_t numerator, int64_t denominator) {
    if (denominator <= 0) {
        throw std::invalid_argument("denominator must be positive");
    }
    return (numerator + denominator - 1) / denominator;
}

} // namespace

// ---------------------------------------------------------------------------
// ScratchAllocator — bump allocator for CUDA graph capture.
// All allocations come from a pre-allocated device buffer.
// No cudaMalloc/cudaFree during capture → safe under stream capture mode.
// ---------------------------------------------------------------------------

class ScratchAllocator final : public DeviceAllocator {
public:
    ScratchAllocator(void* base, size_t capacity, Device device, DeviceAllocator& delegate)
        : base_(static_cast<char*>(base)), capacity_(capacity), device_(device), delegate_(delegate) {}

    void reset() { offset_ = 0; }
    size_t peak() const { return peak_; }

    Memory::Ptr allocate(size_t bytes, Device device) override {
        if (bytes == 0) { return Memory::make_borrowed(nullptr, 0, device); }
        // 256-byte alignment for coalesced access.
        bytes = (bytes + 255) & ~size_t(255);
        if (offset_ + bytes > capacity_) {
            throw std::runtime_error("ScratchAllocator: out of memory (need " +
                                     std::to_string(offset_ + bytes) + ", have " +
                                     std::to_string(capacity_) + ")");
        }
        void* ptr = base_ + offset_;
        offset_ += bytes;
        if (offset_ > peak_) peak_ = offset_;
        return Memory::make_borrowed(ptr, bytes, device_);
    }

    void memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) const override {
        delegate_.memset_async(ptr, value, bytes, stream);
    }
    void copy_to_device_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) const override {
        delegate_.copy_to_device_async(dst, src, bytes, stream);
    }
    void copy_to_host_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) const override {
        delegate_.copy_to_host_async(dst, src, bytes, stream);
    }
    void synchronize_stream(cudaStream_t stream) const override {
        delegate_.synchronize_stream(stream);
    }

private:
    char* base_;
    size_t capacity_;
    size_t offset_ = 0;
    size_t peak_ = 0;
    Device device_;
    DeviceAllocator& delegate_;
};

// ---------------------------------------------------------------------------
// CudaGraphRunner
// ---------------------------------------------------------------------------

CudaGraphRunner::~CudaGraphRunner() {
    for (auto& [bs, cg] : graphs_) {
        if (cg.exec) cudaGraphExecDestroy(cg.exec);
        if (cg.graph) cudaGraphDestroy(cg.graph);
    }
    if (scratch_buffer_) cudaFree(scratch_buffer_);
}

void CudaGraphRunner::capture(TransformerForCausalLM& model,
                               DeviceAllocator& allocator,
                               const Config& config,
                               ScalarType dtype,
                               Device device,
                               cudaStream_t /*unused_stream*/) {
    const int max_bs = std::min(config.max_num_seqs, 512);
    max_num_blocks_ = (config.max_model_len + config.kvcache_block_size - 1) / config.kvcache_block_size;
    hidden_size_ = config.hf_config.hidden_size;

    // Graph capture requires a non-default stream.
    cudaStream_t stream = nullptr;
    throw_if_cuda_error(cudaStreamCreate(&stream), "cudaStreamCreate for graph capture");

    // Batch sizes to capture: 1, 2, 4, 8, then every 4 up to max_bs.
    captured_bs_.clear();
    for (int bs : {1, 2, 4, 8}) {
        if (bs <= max_bs) captured_bs_.push_back(bs);
    }
    for (int bs = 12; bs <= max_bs; bs += 4) {
        captured_bs_.push_back(bs);
    }
    if (captured_bs_.empty() || captured_bs_.back() < max_bs) {
        captured_bs_.push_back(max_bs);
    }
    std::sort(captured_bs_.begin(), captured_bs_.end());
    captured_bs_.erase(std::unique(captured_bs_.begin(), captured_bs_.end()), captured_bs_.end());

    // Pre-allocate input buffers at max batch size (all graphs share same backing memory).
    Tensor g_input_ids = Tensor::zeros({max_bs}, ScalarType::kInt32, device, allocator, stream);
    Tensor g_positions = Tensor::zeros({max_bs}, ScalarType::kInt32, device, allocator, stream);
    Tensor g_slot_mapping = Tensor::zeros({max_bs}, ScalarType::kInt32, device, allocator, stream);
    Tensor g_context_lens = Tensor::zeros({max_bs}, ScalarType::kInt32, device, allocator, stream);
    Tensor g_block_tables = Tensor::zeros({static_cast<int64_t>(max_bs), static_cast<int64_t>(max_num_blocks_)},
                                          ScalarType::kInt32, device, allocator, stream);
    allocator.synchronize_stream(stream);

    // Step 1: Probe scratch size with the largest batch size.
    // Use a very large temporary scratch to find peak usage.
    constexpr size_t kProbeScratchBytes = 512ULL * 1024 * 1024; // 512 MB
    void* probe_buffer = nullptr;
    throw_if_cuda_error(cudaMalloc(&probe_buffer, kProbeScratchBytes), "cudaMalloc probe scratch");

    {
        ScratchAllocator probe_alloc(probe_buffer, kProbeScratchBytes, device, allocator);
        auto g_ids_view = Tensor::from_memory(g_input_ids.memory(), {max_bs}, ScalarType::kInt32);
        auto g_pos_view = Tensor::from_memory(g_positions.memory(), {max_bs}, ScalarType::kInt32);
        auto g_slot_view = Tensor::from_memory(g_slot_mapping.memory(), {max_bs}, ScalarType::kInt32);
        auto g_ctx_view = Tensor::from_memory(g_context_lens.memory(), {max_bs}, ScalarType::kInt32);
        auto g_bt_view = Tensor::from_memory(g_block_tables.memory(),
                                             {static_cast<int64_t>(max_bs), static_cast<int64_t>(max_num_blocks_)},
                                             ScalarType::kInt32);
        set_context(false, Tensor(), Tensor(), 0, 0, g_slot_view, g_ctx_view, g_bt_view);
        Tensor probe_out = model.forward(g_ids_view, g_pos_view, probe_alloc, stream);
        allocator.synchronize_stream(stream);
        reset_context();

        // Allocate the real scratch buffer sized to peak usage + 10% headroom.
        scratch_size_ = probe_alloc.peak();
        scratch_size_ = (scratch_size_ * 11 / 10 + 4095) & ~size_t(4095); // 10% headroom, page-align
    }

    cudaFree(probe_buffer);

    throw_if_cuda_error(cudaMalloc(&scratch_buffer_, scratch_size_), "cudaMalloc scratch");
    std::fprintf(stderr, "[cuda_graph] Scratch buffer: %.1f MB\n",
                 static_cast<double>(scratch_size_) / (1024.0 * 1024.0));

    // Step 2: Capture graphs (largest first for CUDA memory pool reuse).
    for (auto it = captured_bs_.rbegin(); it != captured_bs_.rend(); ++it) {
        const int bs = *it;

        auto g_ids_view = Tensor::from_memory(g_input_ids.memory(), {static_cast<int64_t>(bs)}, ScalarType::kInt32);
        auto g_pos_view = Tensor::from_memory(g_positions.memory(), {static_cast<int64_t>(bs)}, ScalarType::kInt32);
        auto g_slot_view = Tensor::from_memory(g_slot_mapping.memory(), {static_cast<int64_t>(bs)}, ScalarType::kInt32);
        auto g_ctx_view = Tensor::from_memory(g_context_lens.memory(), {static_cast<int64_t>(bs)}, ScalarType::kInt32);
        auto g_bt_view = Tensor::from_memory(g_block_tables.memory(),
                                             {static_cast<int64_t>(bs), static_cast<int64_t>(max_num_blocks_)},
                                             ScalarType::kInt32);

        set_context(false, Tensor(), Tensor(), 0, 0, g_slot_view, g_ctx_view, g_bt_view);

        // Warmup run to stabilize cuBLAS plan selection etc.
        ScratchAllocator warmup_alloc(scratch_buffer_, scratch_size_, device, allocator);
        Tensor warmup_out = model.forward(g_ids_view, g_pos_view, warmup_alloc, stream);
        allocator.synchronize_stream(stream);

        // Capture run: reuse the SAME scratch buffer at the SAME offsets.
        ScratchAllocator capture_alloc(scratch_buffer_, scratch_size_, device, allocator);

        cudaGraph_t graph = nullptr;
        throw_if_cuda_error(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture");
        Tensor captured_out = model.forward(g_ids_view, g_pos_view, capture_alloc, stream);
        throw_if_cuda_error(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");

        cudaGraphExec_t exec = nullptr;
        throw_if_cuda_error(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");

        CapturedGraph cg;
        cg.graph = graph;
        cg.exec = exec;
        cg.bs = bs;
        cg.input_ids = g_input_ids;
        cg.positions = g_positions;
        cg.slot_mapping = g_slot_mapping;
        cg.context_lens = g_context_lens;
        cg.block_tables = g_block_tables;
        cg.output = captured_out;  // Points into scratch buffer — stable address.
        graphs_[bs] = std::move(cg);

        reset_context();
    }

    cudaStreamDestroy(stream);

    std::fprintf(stderr, "[cuda_graph] Captured %zu graphs: ", captured_bs_.size());
    for (size_t i = 0; i < captured_bs_.size(); ++i) {
        if (i > 0) std::fprintf(stderr, ", ");
        std::fprintf(stderr, "%d", captured_bs_[i]);
    }
    std::fprintf(stderr, "\n");
}

bool CudaGraphRunner::available(int batch_size) const {
    return select_bs(batch_size) > 0;
}

int CudaGraphRunner::select_bs(int bs) const {
    auto it = std::lower_bound(captured_bs_.begin(), captured_bs_.end(), bs);
    return (it != captured_bs_.end()) ? *it : 0;
}

Tensor CudaGraphRunner::run(int actual_bs,
                             const Tensor& input_ids,
                             const Tensor& positions,
                             const Tensor& slot_mapping,
                             const Tensor& context_lens,
                             const Tensor& block_tables,
                             DeviceAllocator& allocator,
                             cudaStream_t stream) {
    const int graph_bs = select_bs(actual_bs);
    if (graph_bs <= 0) {
        throw std::runtime_error("No captured CUDA graph for batch size " + std::to_string(actual_bs));
    }

    auto& cg = graphs_.at(graph_bs);

    // Copy inputs into pre-allocated graph buffers (only the used portion).
    const size_t int_bytes = static_cast<size_t>(actual_bs) * sizeof(int32_t);

    // Fill slot_mapping with -1 (so padded slots are skip) then overwrite.
    allocator.memset_async(cg.slot_mapping.data(), 0xFF, static_cast<size_t>(graph_bs) * sizeof(int32_t), stream); // -1 in int32
    throw_if_cuda_error(cudaMemcpyAsync(cg.input_ids.data(), input_ids.data(), int_bytes, cudaMemcpyDeviceToDevice, stream), "graph copy input_ids");
    throw_if_cuda_error(cudaMemcpyAsync(cg.positions.data(), positions.data(), int_bytes, cudaMemcpyDeviceToDevice, stream), "graph copy positions");
    throw_if_cuda_error(cudaMemcpyAsync(cg.slot_mapping.data(), slot_mapping.data(), int_bytes, cudaMemcpyDeviceToDevice, stream), "graph copy slot_mapping");

    // Zero context_lens for padded slots (attention kernel uses context_lens to determine KV range).
    allocator.memset_async(cg.context_lens.data(), 0, static_cast<size_t>(graph_bs) * sizeof(int32_t), stream);
    throw_if_cuda_error(cudaMemcpyAsync(cg.context_lens.data(), context_lens.data(), int_bytes, cudaMemcpyDeviceToDevice, stream), "graph copy context_lens");

    // Copy block tables: only the used rows and columns.
    if (block_tables.defined() && block_tables.numel() > 0) {
        const int64_t src_cols = block_tables.sizes()[1];
        const int64_t dst_cols = max_num_blocks_;
        if (src_cols == dst_cols) {
            const size_t row_bytes = static_cast<size_t>(actual_bs) * static_cast<size_t>(dst_cols) * sizeof(int32_t);
            throw_if_cuda_error(cudaMemcpyAsync(cg.block_tables.data(), block_tables.data(), row_bytes, cudaMemcpyDeviceToDevice, stream), "graph copy block_tables");
        } else {
            throw_if_cuda_error(cudaMemcpy2DAsync(
                cg.block_tables.data(), static_cast<size_t>(dst_cols) * sizeof(int32_t),
                block_tables.data(), static_cast<size_t>(src_cols) * sizeof(int32_t),
                static_cast<size_t>(src_cols) * sizeof(int32_t),
                static_cast<size_t>(actual_bs),
                cudaMemcpyDeviceToDevice, stream), "graph copy block_tables 2D");
        }
    }

    // Replay the captured graph.
    throw_if_cuda_error(cudaGraphLaunch(cg.exec, stream), "cudaGraphLaunch");

    // Return a view of the output covering only the actual batch.
    return Tensor::from_memory(cg.output.memory(),
                               {static_cast<int64_t>(actual_bs), static_cast<int64_t>(hidden_size_)},
                               cg.output.dtype());
}

// ---------------------------------------------------------------------------
// ModelRunner
// ---------------------------------------------------------------------------

ModelRunner::ModelRunner(const Config& config,
                         DeviceAllocator& allocator,
                         int rank,
                         ScalarType dtype,
                         Device device)
    : config_(config),
      rank_(rank),
      device_(device),
      dtype_(dtype),
      allocator_(allocator),
      model_(config_, allocator_, dtype_, device_, rank_) {
    if (config_.kvcache_block_size != Sequence::BLOCK_SIZE) {
        throw std::invalid_argument("ModelRunner currently requires kvcache_block_size == Sequence::BLOCK_SIZE");
    }
    if (rank_ < 0 || rank_ >= config_.tensor_parallel_size) {
        throw std::invalid_argument("rank must satisfy 0 <= rank < tensor_parallel_size");
    }
    set_current_device(device_);
    if (!config_.model.empty()) {
        model_.load_weights(config_.model, allocator_);
    }
    warmup_model();
    allocate_kv_cache();
    if (!config_.enforce_eager) {
        capture_cudagraph();
    }
}

void ModelRunner::warmup_model(cudaStream_t stream) {
    set_current_device(device_);

    const int warmup_tokens = std::max(1, std::min(config_.max_model_len, config_.kvcache_block_size));
    std::vector<int32_t> input_ids(static_cast<size_t>(warmup_tokens), 0);
    std::vector<int32_t> positions(static_cast<size_t>(warmup_tokens), 0);
    for (int index = 0; index < warmup_tokens; ++index) {
        positions[static_cast<size_t>(index)] = index;
    }
    std::vector<int32_t> cu_seqlens = {0, warmup_tokens};

    Tensor input_tensor = make_device_int_tensor({warmup_tokens}, input_ids, stream);
    Tensor position_tensor = make_device_int_tensor({warmup_tokens}, positions, stream);
    Tensor cu_seqlens_tensor = make_device_int_tensor({2}, cu_seqlens, stream);

    ScopedContextReset context_reset;
    set_context(true, cu_seqlens_tensor, cu_seqlens_tensor, warmup_tokens, warmup_tokens);
    Tensor hidden_states = model_.forward(input_tensor, position_tensor, allocator_, stream);
    Tensor logits = model_.compute_logits(hidden_states, allocator_, stream);
    (void)logits;
    allocator_.synchronize_stream(stream);
}

void ModelRunner::allocate_kv_cache() {
    set_current_device(device_);

    const int num_layers = config_.hf_config.num_hidden_layers;
    const int block_size = config_.kvcache_block_size;
    const int num_kv_heads = config_.hf_config.num_key_value_heads;
    if (num_layers <= 0 || block_size <= 0 || num_kv_heads <= 0) {
        throw std::invalid_argument("ModelRunner requires positive layer, block, and KV-head counts");
    }
    if (num_kv_heads % config_.tensor_parallel_size != 0) {
        throw std::invalid_argument("num_key_value_heads must be divisible by tensor_parallel_size");
    }

    const int local_num_kv_heads = num_kv_heads / config_.tensor_parallel_size;
    const int head_dim = resolve_head_dim(config_.hf_config);

    // Compute bytes per block for memory budgeting.
    // Int8: 1 byte per element + 2 bytes scale per head per token.
    // Non-int8: scalar_type_size(dtype_) per element.
    const bool use_int8 = config_.kv_cache_int8;
    uint64_t bytes_per_block;
    if (use_int8) {
        const uint64_t kv_bytes = static_cast<uint64_t>(2) *
                                  static_cast<uint64_t>(num_layers) *
                                  static_cast<uint64_t>(block_size) *
                                  static_cast<uint64_t>(local_num_kv_heads) *
                                  static_cast<uint64_t>(head_dim) * 1;  // int8 = 1 byte
        const uint64_t scale_bytes = static_cast<uint64_t>(2) *
                                     static_cast<uint64_t>(num_layers) *
                                     static_cast<uint64_t>(block_size) *
                                     static_cast<uint64_t>(local_num_kv_heads) * 2;  // fp16 = 2 bytes
        bytes_per_block = kv_bytes + scale_bytes;
    } else {
        bytes_per_block = static_cast<uint64_t>(2) *
                          static_cast<uint64_t>(num_layers) *
                          static_cast<uint64_t>(block_size) *
                          static_cast<uint64_t>(local_num_kv_heads) *
                          static_cast<uint64_t>(head_dim) *
                          static_cast<uint64_t>(scalar_type_size(dtype_));
    }
    if (bytes_per_block == 0) {
        throw std::invalid_argument("KV cache block size must be non-zero");
    }

    if (config_.num_kvcache_blocks > 0) {
        num_kvcache_blocks_ = config_.num_kvcache_blocks;
    } else {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        throw_if_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

        const float utilization = std::clamp(config_.gpu_memory_utilization, 0.0f, 1.0f);
        const int64_t memory_limited_blocks = static_cast<int64_t>(static_cast<long double>(free_bytes) * utilization /
                                                                    static_cast<long double>(bytes_per_block));
        const int64_t per_seq_blocks = ceil_div(config_.max_model_len, block_size);
        const int64_t desired_blocks = std::max<int64_t>(1, per_seq_blocks * config_.max_num_seqs);
        const int64_t chosen_blocks = std::max<int64_t>(1, std::min(desired_blocks, memory_limited_blocks));
        num_kvcache_blocks_ = static_cast<int>(chosen_blocks);
    }

    config_.num_kvcache_blocks = num_kvcache_blocks_;
    kv_cache_ = Tensor::zeros({2,
                               num_layers,
                               num_kvcache_blocks_,
                               block_size,
                               local_num_kv_heads,
                               head_dim},
                              use_int8 ? ScalarType::kInt8 : dtype_,
                              device_,
                              allocator_);

    if (use_int8) {
        kv_scale_ = Tensor::zeros({2,
                                   num_layers,
                                   num_kvcache_blocks_,
                                   block_size,
                                   local_num_kv_heads},
                                  ScalarType::kFloat16,
                                  device_,
                                  allocator_);
    }

    const size_t layer_elements = static_cast<size_t>(num_kvcache_blocks_) *
                                  static_cast<size_t>(block_size) *
                                  static_cast<size_t>(local_num_kv_heads) *
                                  static_cast<size_t>(head_dim);
    const size_t kind_elements = static_cast<size_t>(num_layers) * layer_elements;

    const size_t scale_layer_elements = static_cast<size_t>(num_kvcache_blocks_) *
                                        static_cast<size_t>(block_size) *
                                        static_cast<size_t>(local_num_kv_heads);
    const size_t scale_kind_elements = static_cast<size_t>(num_layers) * scale_layer_elements;

    const ScalarType cache_dtype = use_int8 ? ScalarType::kInt8 : dtype_;
    for (int layer_index = 0; layer_index < num_layers; ++layer_index) {
        const size_t layer_offset = static_cast<size_t>(layer_index) * layer_elements;
        Tensor k_cache = Tensor::from_memory(kv_cache_.memory(),
                                             {num_kvcache_blocks_, block_size, local_num_kv_heads, head_dim},
                                             cache_dtype,
                                             layer_offset);
        Tensor v_cache = Tensor::from_memory(kv_cache_.memory(),
                                             {num_kvcache_blocks_, block_size, local_num_kv_heads, head_dim},
                                             cache_dtype,
                                             kind_elements + layer_offset);
        if (use_int8) {
            const size_t s_offset = static_cast<size_t>(layer_index) * scale_layer_elements;
            Tensor k_scale = Tensor::from_memory(kv_scale_.memory(),
                                                  {num_kvcache_blocks_, block_size, local_num_kv_heads},
                                                  ScalarType::kFloat16,
                                                  s_offset);
            Tensor v_scale = Tensor::from_memory(kv_scale_.memory(),
                                                  {num_kvcache_blocks_, block_size, local_num_kv_heads},
                                                  ScalarType::kFloat16,
                                                  scale_kind_elements + s_offset);
            model_.set_layer_kv_cache_int8(static_cast<size_t>(layer_index),
                                            k_cache, v_cache, k_scale, v_scale);
        } else {
            model_.set_layer_kv_cache(static_cast<size_t>(layer_index), k_cache, v_cache);
        }
    }
}

void ModelRunner::capture_cudagraph(cudaStream_t stream) {
    set_current_device(device_);
    cuda_graph_runner_.capture(model_, allocator_, config_, dtype_, device_, stream);
}

ModelRunner::ModelInputs ModelRunner::prepare_prefill(const std::vector<Sequence*>& seqs,
                                                      cudaStream_t stream) {
    set_current_device(device_);
    if (seqs.empty()) {
        throw std::invalid_argument("prepare_prefill requires at least one sequence");
    }

    std::vector<int32_t> input_ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> cu_seqlens_q = {0};
    std::vector<int32_t> cu_seqlens_k = {0};
    std::vector<int32_t> slot_mapping;
    int max_seqlen_q = 0;
    int max_seqlen_k = 0;

    for (const Sequence* seq : seqs) {
        if (seq == nullptr) {
            throw std::invalid_argument("prepare_prefill received a null sequence pointer");
        }
        if (seq->len() <= 0) {
            throw std::invalid_argument("prepare_prefill does not support empty sequences");
        }
        if (seq->block_table.size() != static_cast<size_t>(seq->num_blocks())) {
            throw std::invalid_argument("prepare_prefill requires allocated block_table entries for each logical block");
        }

        int query_begin = seq->num_computed_tokens;
        int query_end;
        if (seq->num_tokens_to_process > 0) {
            // Scheduler has set an explicit chunk size.  Trust it.
            query_end = query_begin + seq->num_tokens_to_process;
        } else {
            // Legacy fallback: use the old num_cached_tokens semantics.
            query_begin = seq->num_cached_tokens;
            if (query_begin >= seq->len()) {
                query_begin = seq->len() - 1;
            }
            query_end = seq->len();
        }
        if (query_end > seq->len()) {
            query_end = seq->len();
        }

        for (int token_index = query_begin; token_index < query_end; ++token_index) {
            input_ids.push_back(seq->token_ids[static_cast<size_t>(token_index)]);
            positions.push_back(token_index);

            const int block_index = token_index / config_.kvcache_block_size;
            const int block_offset = token_index % config_.kvcache_block_size;
            const int block_id = seq->block_table[static_cast<size_t>(block_index)];
            slot_mapping.push_back(block_id * config_.kvcache_block_size + block_offset);
        }

        const int q_len = query_end - query_begin;
        const int k_len = query_end;  // total K attended so far (includes prior chunks).
        cu_seqlens_q.push_back(cu_seqlens_q.back() + q_len);
        cu_seqlens_k.push_back(cu_seqlens_k.back() + k_len);
        max_seqlen_q = std::max(max_seqlen_q, q_len);
        max_seqlen_k = std::max(max_seqlen_k, k_len);
    }

    Tensor input_tensor = make_device_int_tensor({static_cast<int64_t>(input_ids.size())}, input_ids, stream);
    Tensor position_tensor = make_device_int_tensor({static_cast<int64_t>(positions.size())}, positions, stream);
    Tensor cu_q_tensor = make_device_int_tensor({static_cast<int64_t>(cu_seqlens_q.size())}, cu_seqlens_q, stream);
    Tensor cu_k_tensor = make_device_int_tensor({static_cast<int64_t>(cu_seqlens_k.size())}, cu_seqlens_k, stream);
    Tensor slot_mapping_tensor = make_device_int_tensor({static_cast<int64_t>(slot_mapping.size())}, slot_mapping, stream);
    Tensor block_tables_tensor;
    if (cu_seqlens_k.back() > cu_seqlens_q.back()) {
        block_tables_tensor = build_block_tables(seqs, stream);
    }

    set_context(true,
                cu_q_tensor,
                cu_k_tensor,
                max_seqlen_q,
                max_seqlen_k,
                slot_mapping_tensor,
                Tensor(),
                block_tables_tensor);
    return {input_tensor, position_tensor};
}

ModelRunner::ModelInputs ModelRunner::prepare_decode(const std::vector<Sequence*>& seqs,
                                                     cudaStream_t stream) {
    set_current_device(device_);
    if (seqs.empty()) {
        throw std::invalid_argument("prepare_decode requires at least one sequence");
    }

    std::vector<int32_t> input_ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> slot_mapping;
    std::vector<int32_t> context_lens;
    input_ids.reserve(seqs.size());
    positions.reserve(seqs.size());
    slot_mapping.reserve(seqs.size());
    context_lens.reserve(seqs.size());

    for (const Sequence* seq : seqs) {
        if (seq == nullptr) {
            throw std::invalid_argument("prepare_decode received a null sequence pointer");
        }
        if (seq->len() <= 0) {
            throw std::invalid_argument("prepare_decode does not support empty sequences");
        }
        if (seq->block_table.empty()) {
            throw std::invalid_argument("prepare_decode requires a non-empty block_table");
        }

        input_ids.push_back(seq->last_token);
        positions.push_back(seq->len() - 1);
        context_lens.push_back(seq->len());

        const int block_id = seq->block_table.back();
        const int block_offset = seq->last_block_num_tokens() - 1;
        slot_mapping.push_back(block_id * config_.kvcache_block_size + block_offset);
    }

    // Batch all small int arrays into one allocation + one H2D copy.
    const size_t n = seqs.size();
    std::vector<int32_t> packed;
    packed.reserve(n * 4);
    packed.insert(packed.end(), input_ids.begin(), input_ids.end());
    packed.insert(packed.end(), positions.begin(), positions.end());
    packed.insert(packed.end(), slot_mapping.begin(), slot_mapping.end());
    packed.insert(packed.end(), context_lens.begin(), context_lens.end());
    Tensor packed_tensor = make_device_int_tensor({static_cast<int64_t>(packed.size())}, packed, stream);

    // from_memory offset is in elements, not bytes.
    auto view = [&](size_t elem_offset, int64_t count) {
        return Tensor::from_memory(packed_tensor.memory(),
                                   {count},
                                   ScalarType::kInt32,
                                   elem_offset);
    };
    Tensor input_tensor = view(0, static_cast<int64_t>(n));
    Tensor position_tensor = view(n, static_cast<int64_t>(n));
    Tensor slot_mapping_tensor = view(n * 2, static_cast<int64_t>(n));
    Tensor context_lens_tensor = view(n * 3, static_cast<int64_t>(n));
    Tensor block_tables_tensor = build_block_tables(seqs, stream);

    set_context(false,
                Tensor(),
                Tensor(),
                0,
                0,
                slot_mapping_tensor,
                context_lens_tensor,
                block_tables_tensor);
    return {input_tensor, position_tensor};
}

ModelRunner::SampleInputs ModelRunner::prepare_sample(const std::vector<Sequence*>& seqs,
                                                      cudaStream_t stream) {
    set_current_device(device_);
    const auto n = static_cast<int64_t>(seqs.size());

    std::vector<float> temperatures;
    std::vector<int32_t> top_ks;
    std::vector<float> top_ps;
    std::vector<float> penalties;
    temperatures.reserve(seqs.size());
    top_ks.reserve(seqs.size());
    top_ps.reserve(seqs.size());
    penalties.reserve(seqs.size());

    // Find max token count for penalty_token_ids padding.
    int64_t max_tokens = 0;
    bool any_penalty = false;
    for (const Sequence* seq : seqs) {
        if (seq == nullptr) {
            throw std::invalid_argument("prepare_sample received a null sequence pointer");
        }
        temperatures.push_back(seq->temperature);
        top_ks.push_back(seq->top_k);
        top_ps.push_back(seq->top_p);
        penalties.push_back(seq->repetition_penalty);
        if (seq->repetition_penalty != 1.0f) {
            any_penalty = true;
            const auto len = static_cast<int64_t>(seq->token_ids.size());
            if (len > max_tokens) max_tokens = len;
        }
    }

    SampleInputs inputs;
    inputs.temperatures = make_device_float_tensor({n}, temperatures, stream);
    inputs.top_ks = make_device_int_tensor({n}, top_ks, stream);
    inputs.top_ps = make_device_float_tensor({n}, top_ps, stream);

    if (any_penalty && max_tokens > 0) {
        // Build padded token id matrix for penalty application.
        // Use unique token ids per sequence to avoid double-penalizing.
        std::vector<int32_t> flat_ids(static_cast<size_t>(n * max_tokens), -1);
        std::vector<int32_t> counts;
        counts.reserve(seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) {
            // Deduplicate token ids for this sequence.
            const auto& tids = seqs[i]->token_ids;
            std::vector<int32_t> unique_tids;
            unique_tids.reserve(tids.size());
            // Simple dedup: we need to keep unique tokens.  Use a local set
            // approach but stay within reasonable memory.
            // For simplicity, just copy (duplicates will just re-apply penalty
            // to the same logit index, which is idempotent for the kernel).
            int32_t count = static_cast<int32_t>(tids.size());
            for (int32_t j = 0; j < count && j < static_cast<int32_t>(max_tokens); ++j) {
                flat_ids[i * static_cast<size_t>(max_tokens) + static_cast<size_t>(j)] = tids[static_cast<size_t>(j)];
            }
            counts.push_back(std::min(count, static_cast<int32_t>(max_tokens)));
        }
        inputs.penalty_token_ids = make_device_int_tensor({n, max_tokens}, flat_ids, stream);
        inputs.penalty_token_counts = make_device_int_tensor({n}, counts, stream);
        inputs.penalties = make_device_float_tensor({n}, penalties, stream);
    }

    return inputs;
}

Tensor ModelRunner::run_model(const Tensor& input_ids,
                              const Tensor& positions,
                              bool is_prefill,
                              cudaStream_t stream) {
    set_current_device(device_);
    (void)is_prefill;
    Tensor hidden_states = model_.forward(input_ids, positions, allocator_, stream);
    return model_.compute_logits(hidden_states, allocator_, stream);
}

std::vector<int32_t> ModelRunner::run(const std::vector<Sequence*>& seqs,
                                      bool is_prefill,
                                      cudaStream_t stream) {
    if (seqs.empty()) {
        return {};
    }

    ScopedContextReset context_reset;
    reset_pinned_staging();
    const ModelInputs inputs = is_prefill ? prepare_prefill(seqs, stream) : prepare_decode(seqs, stream);
    SampleInputs sample = prepare_sample(seqs, stream);

    Tensor logits;
    const int bs = static_cast<int>(seqs.size());
    if (!is_prefill && !config_.enforce_eager && cuda_graph_runner_.available(bs)) {
        // CUDA Graph path: replay the graph for model.forward(), then compute_logits outside.
        const auto& ctx = get_context();
        Tensor hidden = cuda_graph_runner_.run(
            bs,
            inputs.input_ids, inputs.positions,
            ctx.slot_mapping, ctx.context_lens, ctx.block_tables,
            allocator_, stream);
        logits = model_.compute_logits(hidden, allocator_, stream);
    } else {
        logits = run_model(inputs.input_ids, inputs.positions, is_prefill, stream);
    }

    if (config_.tensor_parallel_size > 1 && rank_ != 0) {
        return {};
    }
    if (!logits.defined()) {
        return {};
    }
    return sampler_.forward(logits, sample.temperatures, sample.top_ks, sample.top_ps,
                            sample.penalty_token_ids, sample.penalty_token_counts,
                            sample.penalties, allocator_);
}

Tensor ModelRunner::build_block_tables(const std::vector<Sequence*>& seqs,
                                       cudaStream_t stream) {
    int64_t max_blocks = 0;
    for (const Sequence* seq : seqs) {
        if (seq == nullptr) {
            throw std::invalid_argument("build_block_tables received a null sequence pointer");
        }
        max_blocks = std::max<int64_t>(max_blocks, static_cast<int64_t>(seq->block_table.size()));
    }
    if (max_blocks == 0) {
        return Tensor();
    }

    std::vector<int32_t> host_block_tables(static_cast<size_t>(seqs.size() * max_blocks), 0);
    for (size_t row = 0; row < seqs.size(); ++row) {
        const std::vector<int32_t>& block_table = seqs[row]->block_table;
        std::copy(block_table.begin(),
                  block_table.end(),
                  host_block_tables.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(max_blocks)));
    }
    return make_device_int_tensor({static_cast<int64_t>(seqs.size()), max_blocks}, host_block_tables, stream);
}

Tensor ModelRunner::make_device_int_tensor(const std::vector<int64_t>& sizes,
                                           const std::vector<int32_t>& values,
                                           cudaStream_t stream) {
    if (values.size() != compute_numel(sizes)) {
        throw std::invalid_argument("int tensor host value count does not match the requested shape");
    }
    Tensor tensor = Tensor::empty(sizes, ScalarType::kInt32, device_, allocator_);
    const size_t bytes = tensor.nbytes();
    if (bytes == 0) {
        return tensor;
    }
    void* staging = pinned_alloc(bytes);
    std::memcpy(staging, values.data(), bytes);
    allocator_.copy_to_device_async(tensor.data(), staging, bytes, stream);
    return tensor;
}

Tensor ModelRunner::make_device_float_tensor(const std::vector<int64_t>& sizes,
                                             const std::vector<float>& values,
                                             cudaStream_t stream) {
    if (values.size() != compute_numel(sizes)) {
        throw std::invalid_argument("float tensor host value count does not match the requested shape");
    }
    Tensor tensor = Tensor::empty(sizes, ScalarType::kFloat32, device_, allocator_);
    const size_t bytes = tensor.nbytes();
    if (bytes == 0) {
        return tensor;
    }
    void* staging = pinned_alloc(bytes);
    std::memcpy(staging, values.data(), bytes);
    allocator_.copy_to_device_async(tensor.data(), staging, bytes, stream);
    return tensor;
}

void* ModelRunner::pinned_alloc(size_t bytes) {
    // 16-byte align for vectorized memcpy/H2D.
    const size_t aligned = (bytes + 15) & ~size_t(15);
    if (!pinned_staging_ || pinned_offset_ + aligned > pinned_staging_->bytes()) {
        // Grow: start at 256 KB, double until it fits, keep at least current size.
        size_t new_bytes = pinned_staging_ ? pinned_staging_->bytes() * 2 : size_t(256 * 1024);
        const size_t needed = pinned_offset_ + aligned;
        while (new_bytes < needed) {
            new_bytes *= 2;
        }
        // Must not drop an in-flight buffer: ensure any pending async H2D has
        // drained before we free the old pinned memory.
        if (pinned_staging_) {
            throw_if_cuda_error(cudaDeviceSynchronize(),
                                "cudaDeviceSynchronize (pinned staging grow)");
        }
        pinned_staging_.reset(new PinnedHostBuffer(new_bytes));
        pinned_offset_ = 0;  // old offsets are invalid after grow
        // After grow, current `bytes` must still fit from offset 0.
        if (aligned > pinned_staging_->bytes()) {
            throw std::runtime_error("pinned_alloc: request exceeds buffer capacity after grow");
        }
    }
    void* ptr = static_cast<char*>(pinned_staging_->data()) + pinned_offset_;
    pinned_offset_ += aligned;
    return ptr;
}

void ModelRunner::reset_pinned_staging() {
    pinned_offset_ = 0;
}

} // namespace nano_vllm