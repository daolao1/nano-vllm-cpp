#include "layers/attention.h"

#include "kernels/attention.h"
#include "kernels/flash_attn.h"
#include "layers/kernel_ops.h"
#include "utils/context.h"

#include <cmath>
#include <stdexcept>

namespace nano_vllm {

namespace {

void ensure_cuda_contiguous(const char* name, const Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument(std::string(name) + " must live on CUDA");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    }
}

void ensure_compute_dtype(const char* name, const Tensor& tensor) {
    if (tensor.dtype() != ScalarType::kFloat32 &&
        tensor.dtype() != ScalarType::kFloat16 &&
        tensor.dtype() != ScalarType::kBFloat16) {
        throw std::invalid_argument(std::string(name) + " must be float32, float16, or bfloat16");
    }
}

void ensure_index_tensor(const char* name, const Tensor& tensor) {
    ensure_cuda_contiguous(name, tensor);
    if (tensor.dtype() != ScalarType::kInt32 && tensor.dtype() != ScalarType::kInt64) {
        throw std::invalid_argument(std::string(name) + " must be int32 or int64");
    }
}

} // namespace

Attention::Attention(int num_heads, int num_kv_heads, int head_dim, int block_size)
    : num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      block_size_(block_size),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {
    if (num_heads_ <= 0 || num_kv_heads_ <= 0 || head_dim_ <= 0 || block_size_ <= 0) {
        throw std::invalid_argument("attention dimensions must be positive");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument("num_heads must be divisible by num_kv_heads");
    }
}

void Attention::set_kv_cache(const Tensor& k_cache, const Tensor& v_cache) {
    ensure_cuda_contiguous("k_cache", k_cache);
    ensure_cuda_contiguous("v_cache", v_cache);
    ensure_compute_dtype("k_cache", k_cache);
    ensure_compute_dtype("v_cache", v_cache);
    if (k_cache.sizes() != v_cache.sizes()) {
        throw std::invalid_argument("k_cache and v_cache must have the same shape");
    }
    k_cache_ = k_cache;
    v_cache_ = v_cache;
    is_int8_ = false;
}

void Attention::set_kv_cache_int8(const Tensor& k_cache, const Tensor& v_cache,
                                   const Tensor& k_scale, const Tensor& v_scale) {
    ensure_cuda_contiguous("k_cache", k_cache);
    ensure_cuda_contiguous("v_cache", v_cache);
    ensure_cuda_contiguous("k_scale", k_scale);
    ensure_cuda_contiguous("v_scale", v_scale);
    if (k_cache.dtype() != ScalarType::kInt8 || v_cache.dtype() != ScalarType::kInt8) {
        throw std::invalid_argument("int8 KV cache tensors must be int8");
    }
    k_cache_ = k_cache;
    v_cache_ = v_cache;
    k_scale_ = k_scale;
    v_scale_ = v_scale;
    is_int8_ = true;
}

Tensor Attention::forward(const Tensor& query,
                          const Tensor& key,
                          const Tensor& value,
                          DeviceAllocator& allocator,
                          cudaStream_t stream) {
    ensure_cuda_contiguous("query", query);
    ensure_cuda_contiguous("key", key);
    ensure_cuda_contiguous("value", value);
    ensure_compute_dtype("query", query);
    ensure_compute_dtype("key", key);
    ensure_compute_dtype("value", value);
    if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3) {
        throw std::invalid_argument("attention expects [tokens, heads, head_dim] tensors");
    }
    if (query.sizes()[1] != num_heads_ || key.sizes()[1] != num_kv_heads_ || value.sizes()[1] != num_kv_heads_) {
        throw std::invalid_argument("attention tensor heads do not match the configured head counts");
    }
    if (query.sizes()[2] != head_dim_ || key.sizes()[2] != head_dim_ || value.sizes()[2] != head_dim_) {
        throw std::invalid_argument("attention tensor head_dim does not match the configured head_dim");
    }
    if (key.sizes() != value.sizes()) {
        throw std::invalid_argument("key and value must have the same shape");
    }

    Context& context = get_context();
    if (context.cu_seqlens_q.defined()) {
        ensure_index_tensor("cu_seqlens_q", context.cu_seqlens_q);
    }
    if (context.cu_seqlens_k.defined()) {
        ensure_index_tensor("cu_seqlens_k", context.cu_seqlens_k);
    }
    if (context.slot_mapping.defined()) {
        ensure_index_tensor("slot_mapping", context.slot_mapping);
    }
    if (context.context_lens.defined()) {
        ensure_index_tensor("context_lens", context.context_lens);
    }
    if (context.block_tables.defined()) {
        ensure_index_tensor("block_tables", context.block_tables);
    }

    if (k_cache_.defined() && context.slot_mapping.defined()) {
        if (is_int8_) {
            store_kvcache_int8(key, value, k_cache_, v_cache_,
                               k_scale_, v_scale_, context.slot_mapping, stream);
        } else {
            store_kvcache(key, value, k_cache_, v_cache_, context.slot_mapping, stream);
        }
    }

    Tensor output = Tensor::empty(query.sizes(), query.dtype(), query.device(), allocator);
    const bool use_cache = k_cache_.defined() && v_cache_.defined() && context.block_tables.defined();
    if (context.is_prefill) {
        const Tensor* cu_q = context.cu_seqlens_q.defined() ? &context.cu_seqlens_q : nullptr;
        const Tensor* cu_k = context.cu_seqlens_k.defined() ? &context.cu_seqlens_k : nullptr;
        if (use_cache && !is_int8_) {
            if (cu_k == nullptr) {
                throw std::invalid_argument("prefill attention with block_tables requires cu_seqlens_k");
            }
            launch_prefill_attention_paged(query,
                                           cu_q,
                                           k_cache_,
                                           v_cache_,
                                           cu_k,
                                           context.block_tables,
                                           num_kv_heads_,
                                           head_dim_,
                                           block_size_,
                                           scale_,
                                           output,
                                           stream);
        } else {
            // For int8 cache or no cache, use dense K/V for prefill attention.
            if (flash_attn_available() && cu_q && cu_k &&
                (query.dtype() == ScalarType::kBFloat16 || query.dtype() == ScalarType::kFloat16)) {
                flash_attn_varlen_fwd(query, key, value,
                                      *cu_q, *cu_k,
                                      context.max_seqlen_q,
                                      context.max_seqlen_k,
                                      scale_, /*is_causal=*/true,
                                      output, allocator, stream);
            } else {
                launch_prefill_attention_dense(query,
                                               cu_q,
                                               key,
                                               value,
                                               cu_k,
                                               num_kv_heads_,
                                               head_dim_,
                                               scale_,
                                               output,
                                               stream);
            }
        }
    } else {
        if (!use_cache || !context.context_lens.defined()) {
            throw std::invalid_argument("decode attention requires KV cache, context_lens, and block_tables");
        }
        if (is_int8_) {
            launch_decode_attention_paged_int8(query,
                                               k_cache_,
                                               v_cache_,
                                               k_scale_,
                                               v_scale_,
                                               context.context_lens,
                                               context.block_tables,
                                               num_kv_heads_,
                                               head_dim_,
                                               block_size_,
                                               scale_,
                                               output,
                                               stream);
        } else if (flash_attn_available() &&
            (query.dtype() == ScalarType::kBFloat16 || query.dtype() == ScalarType::kFloat16)) {
            flash_attn_decode_kvcache(query,
                                      k_cache_,
                                      v_cache_,
                                      context.context_lens,
                                      context.block_tables,
                                      num_kv_heads_,
                                      head_dim_,
                                      block_size_,
                                      scale_,
                                      output,
                                      allocator,
                                      stream);
        } else {
            launch_decode_attention_paged(query,
                                          k_cache_,
                                          v_cache_,
                                          context.context_lens,
                                          context.block_tables,
                                          num_kv_heads_,
                                          head_dim_,
                                          block_size_,
                                          scale_,
                                          output,
                                          stream);
        }
    }
    return output;
}

Tensor Attention::forward_decode_flash_only(const Tensor& query,
                                            DeviceAllocator& allocator,
                                            cudaStream_t stream) {
    Context& context = get_context();
    if (!k_cache_.defined() || !v_cache_.defined() ||
        !context.block_tables.defined() || !context.context_lens.defined()) {
        throw std::invalid_argument("forward_decode_flash_only requires KV cache, context_lens, and block_tables");
    }

    Tensor output = Tensor::empty(query.sizes(), query.dtype(), query.device(), allocator);

    if (is_int8_) {
        launch_decode_attention_paged_int8(query,
                                           k_cache_,
                                           v_cache_,
                                           k_scale_,
                                           v_scale_,
                                           context.context_lens,
                                           context.block_tables,
                                           num_kv_heads_,
                                           head_dim_,
                                           block_size_,
                                           scale_,
                                           output,
                                           stream);
    } else if (flash_attn_available() &&
        (query.dtype() == ScalarType::kBFloat16 || query.dtype() == ScalarType::kFloat16)) {
        flash_attn_decode_kvcache(query,
                                  k_cache_,
                                  v_cache_,
                                  context.context_lens,
                                  context.block_tables,
                                  num_kv_heads_,
                                  head_dim_,
                                  block_size_,
                                  scale_,
                                  output,
                                  allocator,
                                  stream);
    } else {
        launch_decode_attention_paged(query,
                                      k_cache_,
                                      v_cache_,
                                      context.context_lens,
                                      context.block_tables,
                                      num_kv_heads_,
                                      head_dim_,
                                      block_size_,
                                      scale_,
                                      output,
                                      stream);
    }
    return output;
}

} // namespace nano_vllm