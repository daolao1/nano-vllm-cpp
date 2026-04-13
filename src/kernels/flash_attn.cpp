#include "kernels/flash_attn.h"

#include "utils/cuda_common.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace nano_vllm {

namespace {

// Replicate the PhiloxCudaState struct from ATen to match ABI.
struct PhiloxCudaState {
    union Payload {
        uint64_t val;
        int64_t* ptr;
    };
    Payload seed_{};
    Payload offset_{};
    uint64_t offset_intragraph_ = 0;
    bool captured_ = false;
};

// Replicate Qkv_params + Flash_fwd_params from flash_attn/src/flash.h
// Must match the exact layout of the flash_attn 2.8.x structs.
struct Qkv_params {
    using index_t = int64_t;
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    int h, h_k;
    int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
    void* __restrict__ o_ptr;
    void* __restrict__ oaccum_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void* __restrict__ p_ptr;

    void* __restrict__ softmax_lse_ptr;
    void* __restrict__ softmax_lseaccum_ptr;

    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    float scale_softmax;
    float scale_softmax_log2;

    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;
    int* __restrict__ leftpad_k;

    int* __restrict__ seqused_k;

    int* __restrict__ blockmask;

    void* __restrict__ knew_ptr;
    void* __restrict__ vnew_ptr;

    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    void* __restrict__ rotary_cos_ptr;
    void* __restrict__ rotary_sin_ptr;

    int* __restrict__ cache_batch_idx;

    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    float p_dropout;
    uint8_t p_dropout_in_uint8_t;

    float rp_dropout;
    float scale_softmax_rp_dropout;

    int window_size_left, window_size_right;
    float softcap;

    PhiloxCudaState philox_args;

    uint64_t* rng_state;

    bool is_bf16;
    bool is_causal;

    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;

    void* __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;
};

// Function pointer type: run_mha_fwd_<bf16, 128, true>(Flash_fwd_params&, cudaStream_t)
using RunMhaFwdFn = void (*)(Flash_fwd_params&, cudaStream_t);

struct FlashAttnLib {
    void* handle = nullptr;
    RunMhaFwdFn run_mha_fwd = nullptr;          // prefill: run_mha_fwd_<bf16, 128, causal=true>
    RunMhaFwdFn run_splitkv_dispatch = nullptr;  // decode:  run_mha_fwd_splitkv_dispatch<bf16, 128, causal=false>
    bool initialized = false;
    bool available = false;
};

FlashAttnLib& get_flash_lib() {
    static FlashAttnLib lib;
    static std::once_flag flag;
    std::call_once(flag, []() {
        // Pre-load Python (for PyByteArray_Type etc).
        if (!dlopen("libpython3.12.so", RTLD_NOW | RTLD_GLOBAL)) {
            std::fprintf(stderr, "[flash_attn] Failed to load libpython3.12: %s\n", dlerror());
            return;
        }

        // Pre-load CUDA runtime from PyTorch's bundled CUDA (provides newer symbols).
        dlopen("/home/zzy/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
               RTLD_NOW | RTLD_GLOBAL);

        // Pre-load PyTorch dependencies required by flash_attn_2_cuda.
        const char* torch_lib = "/home/zzy/venv/lib/python3.12/site-packages/torch/lib/";
        const char* deps[] = {"libc10.so", "libc10_cuda.so", "libtorch_cuda.so"};
        for (const char* dep : deps) {
            std::string path = std::string(torch_lib) + dep;
            void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!h) {
                std::fprintf(stderr, "[flash_attn] Failed to load %s: %s\n", dep, dlerror());
                return;
            }
        }

        const char* so_path = "/home/zzy/venv/lib/python3.12/site-packages/"
                              "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so";
        lib.handle = dlopen(so_path, RTLD_LAZY | RTLD_GLOBAL);
        if (!lib.handle) {
            std::fprintf(stderr, "[flash_attn] Failed to load flash_attn_2_cuda: %s\n", dlerror());
            return;
        }
        // Use the specific template instantiation for bf16, head_dim=128, causal=true.
        // This avoids needing the dispatcher which may pull in more Python deps.
        const char* sym = "_ZN5flash12run_mha_fwd_IN7cutlass10bfloat16_tELi128ELb1EEEvRNS_16Flash_fwd_paramsEP11CUstream_st";
        lib.run_mha_fwd = reinterpret_cast<RunMhaFwdFn>(dlsym(lib.handle, sym));
        if (lib.run_mha_fwd) {
            lib.available = true;
        } else {
            std::fprintf(stderr, "[flash_attn] Symbol not found: %s\n", dlerror());
        }

        // Split-K dispatch for decode with paged KV cache (bf16, head_dim=128, causal=false).
        const char* splitkv_sym =
            "_ZN5flash28run_mha_fwd_splitkv_dispatchIN7cutlass10bfloat16_tELi128ELb0EEEvRNS_16Flash_fwd_paramsEP11CUstream_st";
        lib.run_splitkv_dispatch = reinterpret_cast<RunMhaFwdFn>(dlsym(lib.handle, splitkv_sym));
        if (!lib.run_splitkv_dispatch) {
            std::fprintf(stderr, "[flash_attn] splitkv_dispatch symbol not found: %s\n", dlerror());
        }
    });
    return lib;
}

int round_up_to_power_of_2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

} // namespace

bool flash_attn_available() {
    return get_flash_lib().available;
}

void flash_attn_varlen_fwd(const Tensor& query,
                           const Tensor& key,
                           const Tensor& value,
                           const Tensor& cu_seqlens_q,
                           const Tensor& cu_seqlens_k,
                           int max_seqlen_q,
                           int max_seqlen_k,
                           float scale,
                           bool is_causal,
                           Tensor& output,
                           DeviceAllocator& allocator,
                           cudaStream_t stream) {
    auto& lib = get_flash_lib();
    if (!lib.available) {
        throw std::runtime_error("flash_attn_2_cuda library not available");
    }

    // query: [total_q, num_heads, head_dim]
    // key/value: [total_k, num_kv_heads, head_dim]
    const int64_t total_q = query.sizes()[0];
    const int num_heads = static_cast<int>(query.sizes()[1]);
    const int head_dim = static_cast<int>(query.sizes()[2]);
    const int64_t total_k = key.sizes()[0];
    const int num_kv_heads = static_cast<int>(key.sizes()[1]);
    const int batch_size = static_cast<int>(cu_seqlens_q.numel() - 1);

    // Allocate softmax_lse: [num_heads, total_q]
    Tensor softmax_lse = Tensor::empty(
        {num_heads, total_q}, ScalarType::kFloat32, query.device(), allocator);

    Flash_fwd_params params;
    std::memset(&params, 0, sizeof(params));

    params.q_ptr = const_cast<void*>(query.data());
    params.k_ptr = const_cast<void*>(key.data());
    params.v_ptr = const_cast<void*>(value.data());
    params.o_ptr = output.data();
    params.oaccum_ptr = nullptr;

    // Strides: [total_tokens, heads, head_dim] layout.
    // Row stride = heads * head_dim, head stride = head_dim.
    params.q_batch_stride = 0;  // varlen: no batch stride
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;
    params.o_batch_stride = 0;
    params.q_row_stride = num_heads * head_dim;
    params.k_row_stride = num_kv_heads * head_dim;
    params.v_row_stride = num_kv_heads * head_dim;
    params.o_row_stride = num_heads * head_dim;
    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.h_h_k_ratio = num_heads / num_kv_heads;

    params.b = batch_size;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_knew = 0;
    params.d = head_dim;
    params.seqlen_q_rounded = round_up_to_power_of_2(max_seqlen_q);
    params.seqlen_k_rounded = round_up_to_power_of_2(max_seqlen_k);
    params.d_rounded = round_up_to_power_of_2(head_dim);
    params.rotary_dim = 0;
    params.total_q = static_cast<int>(total_q);

    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * static_cast<float>(M_LOG2E);

    params.cu_seqlens_q = const_cast<int*>(
        static_cast<const int*>(cu_seqlens_q.data()));
    params.cu_seqlens_k = const_cast<int*>(
        static_cast<const int*>(cu_seqlens_k.data()));

    params.softmax_lse_ptr = softmax_lse.data();

    params.p_dropout = 1.0f;  // Keep all (no dropout)
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;

    params.is_bf16 = (query.dtype() == ScalarType::kBFloat16);
    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = true;

    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;
    params.softcap = 0.0f;

    params.num_splits = 1;
    params.unpadded_lse = true;  // varlen mode: LSE in [num_heads, total_q]

    lib.run_mha_fwd(params, stream);
    throw_if_cuda_error(cudaPeekAtLastError(), "flash_attn run_mha_fwd");
}

void flash_attn_decode_kvcache(const Tensor& query,
                               const Tensor& k_cache,
                               const Tensor& v_cache,
                               const Tensor& context_lens,
                               const Tensor& block_tables,
                               int num_kv_heads,
                               int head_dim,
                               int block_size,
                               float scale,
                               Tensor& output,
                               DeviceAllocator& allocator,
                               cudaStream_t stream) {
    auto& lib = get_flash_lib();
    if (!lib.run_splitkv_dispatch) {
        throw std::runtime_error("flash_attn splitkv_dispatch not available");
    }

    // query: [batch_size, num_heads, head_dim]
    // k_cache/v_cache: [num_blocks, page_block_size, num_kv_heads, head_dim]
    // context_lens: [batch_size] int32
    // block_tables: [batch_size, max_num_blocks_per_seq] int32
    const int batch_size = static_cast<int>(query.sizes()[0]);
    const int num_heads = static_cast<int>(query.sizes()[1]);
    const int page_block_size = static_cast<int>(k_cache.sizes()[1]);
    const int max_num_blocks_per_seq = static_cast<int>(block_tables.sizes()[1]);
    const int seqlen_k = max_num_blocks_per_seq * page_block_size;

    // GQA optimization: swap seqlen_q with ngroups to halve the grid
    // Mirrors mha_fwd_kvcache's seqlenq_ngroups_swapped logic
    const bool do_gqa_swap = (num_heads > num_kv_heads) && (head_dim % 8 == 0);
    const int ngroups = num_heads / num_kv_heads;
    const int eff_seqlen_q = do_gqa_swap ? ngroups : 1;
    const int eff_num_heads = do_gqa_swap ? num_kv_heads : num_heads;

    // Allocate softmax_lse: [batch_size, eff_num_heads, eff_seqlen_q]
    Tensor softmax_lse = Tensor::empty(
        {batch_size, eff_num_heads, eff_seqlen_q}, ScalarType::kFloat32, query.device(), allocator);

    Flash_fwd_params params;
    std::memset(&params, 0, sizeof(params));

    params.q_ptr = const_cast<void*>(query.data());
    params.k_ptr = const_cast<void*>(k_cache.data());
    params.v_ptr = const_cast<void*>(v_cache.data());
    params.o_ptr = output.data();
    params.oaccum_ptr = nullptr;

    if (do_gqa_swap) {
        // Q viewed as [batch, 1, num_heads, head_dim] = [batch, 1, nkv*ngroups, head_dim]
        // Reshape to [batch, nkv, ngroups, head_dim], transpose to [batch, ngroups, nkv, head_dim]
        // In memory, Q is contiguous [batch, num_heads, head_dim]
        // Transposed strides: batch_stride=num_heads*head_dim, row_stride=head_dim, head_stride=ngroups*head_dim
        params.q_batch_stride = num_heads * head_dim;
        params.q_row_stride = head_dim;                   // stride along ngroups (seqlen_q) dim
        params.q_head_stride = ngroups * head_dim;         // stride along nkv (heads) dim
        // Output: write directly in [batch, num_heads, head_dim] layout
        // For (group g, kv_head kv): target head = kv * ngroups + g, offset = head * head_dim
        // So: o_row_stride = head_dim (between groups), o_head_stride = ngroups * head_dim (between kv_heads)
        params.o_batch_stride = num_heads * head_dim;
        params.o_row_stride = head_dim;
        params.o_head_stride = ngroups * head_dim;
    } else {
        // Standard layout: q/o as [batch, 1, num_heads, head_dim]
        params.q_batch_stride = num_heads * head_dim;
        params.q_row_stride = num_heads * head_dim;
        params.q_head_stride = head_dim;
        params.o_batch_stride = num_heads * head_dim;
        params.o_row_stride = num_heads * head_dim;
        params.o_head_stride = head_dim;
    }

    // k/v strides: [num_blocks, page_block_size, num_kv_heads, head_dim]
    // block_stride (batch_stride for paged): page_block_size * num_kv_heads * head_dim
    params.k_batch_stride = static_cast<int64_t>(page_block_size) * num_kv_heads * head_dim;
    params.v_batch_stride = params.k_batch_stride;
    params.k_row_stride = num_kv_heads * head_dim;
    params.v_row_stride = num_kv_heads * head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;

    params.h = eff_num_heads;
    params.h_k = num_kv_heads;
    params.h_h_k_ratio = eff_num_heads / num_kv_heads;

    params.b = batch_size;
    params.seqlen_q = eff_seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_knew = 0;
    params.d = head_dim;
    params.seqlen_q_rounded = round_up_to_power_of_2(eff_seqlen_q);
    params.seqlen_k_rounded = round_up_to_power_of_2(seqlen_k);
    params.d_rounded = round_up_to_power_of_2(head_dim);
    params.rotary_dim = 0;
    params.total_q = batch_size * eff_seqlen_q;

    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * static_cast<float>(M_LOG2E);

    // Per-sequence context lengths (not cumulative)
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = const_cast<int*>(static_cast<const int*>(context_lens.data()));
    params.is_seqlens_k_cumulative = false;

    params.softmax_lse_ptr = softmax_lse.data();
    params.softmax_lseaccum_ptr = nullptr;

    // Paged KV cache
    params.block_table = const_cast<int*>(static_cast<const int*>(block_tables.data()));
    params.block_table_batch_stride = max_num_blocks_per_seq;
    params.page_block_size = page_block_size;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;

    params.is_bf16 = (query.dtype() == ScalarType::kBFloat16);
    params.is_causal = false;  // seqlen_q=1 → no causal needed

    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;

    params.num_splits = 1;  // No split-K for our batch sizes
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    lib.run_splitkv_dispatch(params, stream);
    throw_if_cuda_error(cudaPeekAtLastError(), "flash_attn run_splitkv_dispatch");
}

} // namespace nano_vllm
