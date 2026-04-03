#include "layers/embed_head.h"

#include "layers/kernel_ops.h"
#include "layers/linear.h"
#include "utils/context.h"
#include "utils/loader.h"
#include "utils/tensor_parallel.h"

#include <stdexcept>

namespace nano_vllm {
namespace {

void validate_tp_rank(const char* name, int tp_size, int rank) {
    if (tp_size <= 0) {
        throw std::invalid_argument(std::string(name) + " tp_size must be positive");
    }
    if (rank < 0 || rank >= tp_size) {
        throw std::invalid_argument(std::string(name) + " rank must satisfy 0 <= rank < tp_size");
    }
}

void ensure_cuda_index_tensor(const char* name, const Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument(std::string(name) + " must live on CUDA");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    }
    if (tensor.dtype() != ScalarType::kInt32 && tensor.dtype() != ScalarType::kInt64) {
        throw std::invalid_argument(std::string(name) + " must be int32 or int64");
    }
}

std::shared_ptr<TensorParallelCommunicator> require_tensor_parallel_communicator(const char* name) {
    const std::shared_ptr<TensorParallelCommunicator> communicator = get_tensor_parallel_communicator();
    if (!communicator) {
        throw std::invalid_argument(std::string(name) +
                                    " with tp_size > 1 requires set_tensor_parallel_communicator(...)");
    }
    return communicator;
}

} // namespace

VocabParallelEmbedding::VocabParallelEmbedding(int vocab_size,
                                               int hidden_dim,
                                               ScalarType dtype,
                                               Device device,
                                               DeviceAllocator& allocator,
                                               int tp_size,
                                                                                             int rank,
                                                                                             std::string tp_group_key)
    : vocab_size_(vocab_size),
      hidden_dim_(hidden_dim),
      tp_size_(tp_size),
      rank_(rank),
      local_vocab_size_(vocab_size / tp_size),
      vocab_start_(local_vocab_size_ * rank),
      vocab_end_(vocab_start_ + local_vocab_size_),
            tp_group_key_(std::move(tp_group_key)),
      weight_(Tensor::zeros({local_vocab_size_, hidden_dim_}, dtype, device, allocator)) {
    validate_tp_rank("vocab parallel embedding", tp_size_, rank_);
    if (vocab_size_ % tp_size_ != 0) {
        throw std::invalid_argument("vocab_size must be divisible by tp_size");
    }
}

Tensor VocabParallelEmbedding::forward(const Tensor& input_ids,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) const {
    ensure_cuda_index_tensor("input_ids", input_ids);

    std::vector<int64_t> output_sizes = input_ids.sizes();
    output_sizes.push_back(hidden_dim_);
    Tensor output = Tensor::empty(output_sizes, weight_.dtype(), weight_.device(), allocator);
    embedding_lookup(input_ids, weight_, vocab_size_, vocab_start_, output, stream);
    if (tp_size_ > 1) {
        require_tensor_parallel_communicator("vocab parallel embedding")
            ->all_reduce_sum(tp_group_key_, tp_size_, rank_, output, allocator, stream);
    }
    return output;
}

void VocabParallelEmbedding::weight_loader(const Tensor& source,
                                          DeviceAllocator& allocator,
                                          cudaStream_t stream) {
    if (source.dim() != 2 || source.sizes()[1] != hidden_dim_) {
        throw std::invalid_argument("embedding weight must have shape [vocab, hidden_dim]");
    }
    Tensor shard = source;
    if (source.sizes()[0] == vocab_size_) {
        shard = select_tensor_shard(source, 0, vocab_start_, local_vocab_size_);
    } else if (source.sizes()[0] != local_vocab_size_) {
        throw std::invalid_argument("embedding weight has incompatible vocab dimension");
    }
    copy_tensor_to_parameter(shard, weight_, allocator, stream);
}

ParallelLMHead::ParallelLMHead(int vocab_size,
                               int hidden_dim,
                               ScalarType dtype,
                               Device device,
                               DeviceAllocator& allocator,
                               int tp_size,
                                                             int rank,
                                                             std::string tp_group_key)
    : vocab_size_(vocab_size),
      hidden_dim_(hidden_dim),
      tp_size_(tp_size),
      rank_(rank),
      local_vocab_size_(vocab_size / tp_size),
            tp_group_key_(std::move(tp_group_key)),
      weight_(Tensor::zeros({local_vocab_size_, hidden_dim_}, dtype, device, allocator)) {
    validate_tp_rank("parallel lm head", tp_size_, rank_);
    if (vocab_size_ % tp_size_ != 0) {
        throw std::invalid_argument("vocab_size must be divisible by tp_size");
    }
}

Tensor ParallelLMHead::forward(const Tensor& hidden_states,
                               DeviceAllocator& allocator,
                               cudaStream_t stream) const {
    if (hidden_states.dim() != 2 || hidden_states.sizes()[1] != hidden_dim_) {
        throw std::invalid_argument("hidden_states must have shape [N, hidden_dim]");
    }

    Tensor gathered = hidden_states;
    if (get_context().is_prefill && get_context().cu_seqlens_q.defined()) {
        const int64_t rows = static_cast<int64_t>(get_context().cu_seqlens_q.numel()) - 1;
        gathered = Tensor::zeros({rows, hidden_dim_}, hidden_states.dtype(), hidden_states.device(), allocator, stream);
        gather_last_tokens(hidden_states, get_context().cu_seqlens_q, gathered, stream);
    }

    Tensor logits = linear_forward(gathered, weight_, Tensor(), allocator, stream);
    if (tp_size_ > 1) {
        return require_tensor_parallel_communicator("parallel lm head")
            ->gather_last_dim_to_rank0(tp_group_key_,
                                       tp_size_,
                                       rank_,
                                       logits,
                                       vocab_size_,
                                       allocator,
                                       stream);
    }
    return logits;
}

void ParallelLMHead::weight_loader(const Tensor& source,
                                   DeviceAllocator& allocator,
                                   cudaStream_t stream) {
    if (source.dim() != 2 || source.sizes()[1] != hidden_dim_) {
        throw std::invalid_argument("lm_head weight must have shape [vocab, hidden_dim]");
    }
    Tensor shard = source;
    if (source.sizes()[0] == vocab_size_) {
        shard = select_tensor_shard(source, 0, static_cast<int64_t>(rank_) * local_vocab_size_, local_vocab_size_);
    } else if (source.sizes()[0] != local_vocab_size_) {
        throw std::invalid_argument("lm_head weight has incompatible vocab dimension");
    }
    copy_tensor_to_parameter(shard, weight_, allocator, stream);
}

} // namespace nano_vllm