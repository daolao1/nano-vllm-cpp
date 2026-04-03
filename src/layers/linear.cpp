#include "layers/linear.h"

#include "layers/kernel_ops.h"
#include "utils/cublas_wrapper.h"

#include "utils/loader.h"
#include "utils/tensor_parallel.h"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

std::string shape_string(const std::vector<int64_t>& sizes) {
    std::string result = "[";
    for (size_t index = 0; index < sizes.size(); ++index) {
        if (index != 0) {
            result += ", ";
        }
        result += std::to_string(sizes[index]);
    }
    result += "]";
    return result;
}

void ensure_cuda_compute_tensor(const char* name, const Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (tensor.device().type != DeviceType::kCUDA) {
        throw std::invalid_argument(std::string(name) + " must live on CUDA");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    }
    if (tensor.dtype() != ScalarType::kFloat32 &&
        tensor.dtype() != ScalarType::kFloat16 &&
        tensor.dtype() != ScalarType::kBFloat16) {
        throw std::invalid_argument(std::string(name) + " must be float32, float16, or bfloat16");
    }
}

int to_gemm_dim(const char* name, int64_t value) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " exceeds cuBLAS supported range");
    }
    return static_cast<int>(value);
}

CublasHandle& current_thread_cublas_handle() {
    thread_local CublasHandle handle;
    return handle;
}

std::shared_ptr<TensorParallelCommunicator> require_tensor_parallel_communicator(const char* name) {
    const std::shared_ptr<TensorParallelCommunicator> communicator = get_tensor_parallel_communicator();
    if (!communicator) {
        throw std::invalid_argument(std::string(name) +
                                    " with tp_size > 1 requires set_tensor_parallel_communicator(...)");
    }
    return communicator;
}

Tensor run_linear(const Tensor& input,
                  const Tensor& weight,
                  const Tensor& bias,
                  int input_size,
                  int output_size,
                  bool use_bias,
                  DeviceAllocator& allocator,
                  cudaStream_t stream) {
    ensure_cuda_compute_tensor("input", input);
    ensure_cuda_compute_tensor("weight", weight);
    if (use_bias) {
        ensure_cuda_compute_tensor("bias", bias);
    }
    if (input.sizes().empty() || input.sizes().back() != input_size) {
        throw std::invalid_argument("input last dimension does not match the linear input size");
    }
    if (weight.dim() != 2 || weight.sizes()[0] != output_size || weight.sizes()[1] != input_size) {
        throw std::invalid_argument("weight must have shape [output_size, input_size]");
    }

    std::vector<int64_t> output_sizes = input.sizes();
    output_sizes.back() = output_size;
    const ScalarType compute_dtype = input.dtype();
    Tensor output = Tensor::empty(output_sizes, compute_dtype, input.device(), allocator);

    const int64_t rows = static_cast<int64_t>(input.numel() / static_cast<size_t>(input_size));
    if (rows == 0 || output_size == 0) {
        return output;
    }

    CublasLtContext::instance().gemm_row_major(
        static_cast<int>(rows), output_size, input_size,
        compute_dtype, input.data(), weight.data(), output.data(), stream);
    if (use_bias) {
        add_bias(output, bias, stream);
    }
    return output;
}

Tensor select_output_partition(const Tensor& source,
                               int64_t full_size,
                               int64_t local_size,
                               int rank,
                               const char* name) {
    if (source.dim() == 0) {
        throw std::invalid_argument(std::string(name) + " must have rank >= 1");
    }
    if (source.sizes()[0] == local_size) {
        return source;
    }
    if (source.sizes()[0] == full_size) {
        return select_tensor_shard(source,
                                   0,
                                   static_cast<int64_t>(rank) * local_size,
                                   local_size);
    }
    throw std::invalid_argument(std::string(name) + " has incompatible leading dimension: got " +
                                shape_string(source.sizes()));
}

Tensor select_input_partition(const Tensor& source,
                              int64_t full_size,
                              int64_t local_size,
                              int rank,
                              const char* name) {
    if (source.dim() != 2) {
        throw std::invalid_argument(std::string(name) + " must have rank 2");
    }
    if (source.sizes()[1] == local_size) {
        return source;
    }
    if (source.sizes()[1] == full_size) {
        return select_tensor_shard(source,
                                   1,
                                   static_cast<int64_t>(rank) * local_size,
                                   local_size);
    }
    throw std::invalid_argument(std::string(name) + " has incompatible trailing dimension: got " +
                                shape_string(source.sizes()));
}

Tensor select_qkv_destination_shard(const Tensor& destination,
                                    int shard_id,
                                    int local_q_size,
                                    int local_kv_size) {
    int64_t start = 0;
    int64_t size = 0;
    switch (shard_id) {
    case 0:
        start = 0;
        size = local_q_size;
        break;
    case 1:
        start = local_q_size;
        size = local_kv_size;
        break;
    case 2:
        start = local_q_size + local_kv_size;
        size = local_kv_size;
        break;
    default:
        throw std::invalid_argument("QKV shard_id must be 0 (q), 1 (k), or 2 (v)");
    }
    return select_tensor_shard(destination, 0, start, size);
}

Tensor select_merged_destination_shard(const Tensor& destination,
                                       const std::vector<int>& shard_offsets,
                                       const std::vector<int>& shard_sizes,
                                       int shard_id) {
    if (shard_id < 0 || shard_id >= static_cast<int>(shard_sizes.size())) {
        throw std::invalid_argument("merged linear shard_id is out of range");
    }
    return select_tensor_shard(destination,
                               0,
                               shard_offsets[static_cast<size_t>(shard_id)],
                               shard_sizes[static_cast<size_t>(shard_id)]);
}

} // namespace

Tensor linear_forward(const Tensor& input,
                      const Tensor& weight,
                      const Tensor& bias,
                      DeviceAllocator& allocator,
                      cudaStream_t stream) {
    if (weight.dim() != 2) {
        throw std::invalid_argument("weight must have shape [output_size, input_size]");
    }
    if (weight.sizes()[0] < 0 || weight.sizes()[1] < 0) {
        throw std::invalid_argument("weight dimensions must be non-negative");
    }
    const bool use_bias = bias.defined();
    return run_linear(input,
                      weight,
                      bias,
                      static_cast<int>(weight.sizes()[1]),
                      static_cast<int>(weight.sizes()[0]),
                      use_bias,
                      allocator,
                      stream);
}

ColumnParallelLinear::ColumnParallelLinear(int input_size,
                                           int output_size,
                                           bool use_bias,
                                           ScalarType dtype,
                                           Device device,
                                           DeviceAllocator& allocator,
                                           int tp_size,
                                           int rank)
    : input_size_(input_size),
      output_size_(output_size),
      local_output_size_(output_size / tp_size),
      tp_size_(tp_size),
      rank_(rank),
      use_bias_(use_bias),
      weight_(Tensor::zeros({local_output_size_, input_size_}, dtype, device, allocator)),
      bias_(use_bias ? Tensor::zeros({local_output_size_}, dtype, device, allocator) : Tensor()) {
        validate_tp_rank("column linear", tp_size_, rank_);
        if (output_size_ % tp_size_ != 0) {
        throw std::invalid_argument("output_size must be divisible by tp_size");
    }
}

Tensor ColumnParallelLinear::forward(const Tensor& input,
                                     DeviceAllocator& allocator,
                                     cudaStream_t stream) const {
    return linear_forward(input,
                          weight_,
                          use_bias_ ? bias_ : Tensor(),
                          allocator,
                          stream);
}

void ColumnParallelLinear::weight_loader(const Tensor& source,
                                         DeviceAllocator& allocator,
                                         cudaStream_t stream) {
    if (source.dim() != 2 || source.sizes()[1] != input_size_) {
        throw std::invalid_argument("column linear weight must have shape [rows, input_size]");
    }
    const Tensor shard = select_output_partition(source,
                                                 output_size_,
                                                 local_output_size_,
                                                 rank_,
                                                 "column linear weight");
    copy_tensor_to_parameter(shard, weight_, allocator, stream);
}

void ColumnParallelLinear::bias_loader(const Tensor& source,
                                       DeviceAllocator& allocator,
                                       cudaStream_t stream) {
    if (!use_bias_) {
        throw std::invalid_argument("column linear bias_loader called on a bias-free layer");
    }
    if (source.dim() != 1) {
        throw std::invalid_argument("column linear bias must have rank 1");
    }
    const Tensor shard = select_output_partition(source,
                                                 output_size_,
                                                 local_output_size_,
                                                 rank_,
                                                 "column linear bias");
    copy_tensor_to_parameter(shard, bias_, allocator, stream);
}

RowParallelLinear::RowParallelLinear(int input_size,
                                     int output_size,
                                     bool use_bias,
                                     ScalarType dtype,
                                     Device device,
                                     DeviceAllocator& allocator,
                                     int tp_size,
                                                                         int rank,
                                                                         std::string tp_group_key)
    : local_input_size_(input_size / tp_size),
      output_size_(output_size),
      tp_size_(tp_size),
      rank_(rank),
      use_bias_(use_bias),
            tp_group_key_(std::move(tp_group_key)),
      weight_(Tensor::zeros({output_size_, local_input_size_}, dtype, device, allocator)),
      bias_(use_bias ? Tensor::zeros({output_size_}, dtype, device, allocator) : Tensor()) {
    validate_tp_rank("row linear", tp_size_, rank_);
    if (input_size % tp_size_ != 0) {
        throw std::invalid_argument("input_size must be divisible by tp_size");
    }
}

Tensor RowParallelLinear::forward(const Tensor& input,
                                  DeviceAllocator& allocator,
                                  cudaStream_t stream) const {
    const bool apply_bias = use_bias_ && (tp_size_ == 1 || rank_ == 0);
    Tensor output = linear_forward(input,
                                   weight_,
                                   apply_bias ? bias_ : Tensor(),
                                   allocator,
                                   stream);
    if (tp_size_ > 1) {
        require_tensor_parallel_communicator("row parallel linear")
            ->all_reduce_sum(tp_group_key_, tp_size_, rank_, output, allocator, stream);
    }
    return output;
}

void RowParallelLinear::weight_loader(const Tensor& source,
                                      DeviceAllocator& allocator,
                                      cudaStream_t stream) {
    if (source.dim() != 2 || source.sizes()[0] != output_size_) {
        throw std::invalid_argument("row linear weight must have shape [output_size, cols]");
    }
    const Tensor shard = select_input_partition(source,
                                                static_cast<int64_t>(local_input_size_) * tp_size_,
                                                local_input_size_,
                                                rank_,
                                                "row linear weight");
    copy_tensor_to_parameter(shard, weight_, allocator, stream);
}

void RowParallelLinear::bias_loader(const Tensor& source,
                                    DeviceAllocator& allocator,
                                    cudaStream_t stream) {
    if (!use_bias_) {
        throw std::invalid_argument("row linear bias_loader called on a bias-free layer");
    }
    if (source.sizes() != bias_.sizes()) {
        throw std::invalid_argument("row linear bias must have shape " + shape_string(bias_.sizes()));
    }
    copy_tensor_to_parameter(source, bias_, allocator, stream);
}

QKVParallelLinear::QKVParallelLinear(int hidden_size,
                                     int head_size,
                                     int num_heads,
                                     int num_kv_heads,
                                     bool use_bias,
                                     ScalarType dtype,
                                     Device device,
                                     DeviceAllocator& allocator,
                                     int tp_size,
                                     int rank)
    : full_q_size_(num_heads * head_size),
      full_kv_size_(num_kv_heads * head_size),
      local_q_size_((num_heads / tp_size) * head_size),
      local_kv_size_((num_kv_heads / tp_size) * head_size),
      tp_size_(tp_size),
      rank_(rank),
      linear_(hidden_size,
              (num_heads / tp_size) * head_size + 2 * (num_kv_heads / tp_size) * head_size,
              use_bias,
              dtype,
              device,
              allocator,
              /*tp_size=*/1,
              /*rank=*/0) {
    validate_tp_rank("qkv parallel linear", tp_size_, rank_);
    if (num_heads % tp_size_ != 0 || num_kv_heads % tp_size_ != 0) {
        throw std::invalid_argument("num_heads and num_kv_heads must be divisible by tp_size");
    }
}

Tensor QKVParallelLinear::forward(const Tensor& input,
                                  DeviceAllocator& allocator,
                                  cudaStream_t stream) const {
    return linear_.forward(input, allocator, stream);
}

void QKVParallelLinear::weight_loader(const Tensor& source,
                                      DeviceAllocator& allocator,
                                      cudaStream_t stream) {
    linear_.weight_loader(source, allocator, stream);
}

void QKVParallelLinear::weight_loader(const Tensor& source,
                                      int shard_id,
                                      DeviceAllocator& allocator,
                                      cudaStream_t stream) {
    if (source.dim() != 2 || source.sizes()[1] != weight().sizes()[1]) {
        throw std::invalid_argument("qkv shard weight must have shape [rows, hidden_size]");
    }
    const int64_t local_size = shard_id == 0 ? local_q_size_ : local_kv_size_;
    int64_t shard_full_size = 0;
    switch (shard_id) {
    case 0:
        shard_full_size = full_q_size_;
        break;
    case 1:
    case 2:
        shard_full_size = full_kv_size_;
        break;
    default:
        throw std::invalid_argument("QKV shard_id must be 0 (q), 1 (k), or 2 (v)");
    }
    const Tensor source_shard = select_output_partition(source,
                                                        shard_full_size,
                                                        local_size,
                                                        rank_,
                                                        "qkv shard weight");
    const Tensor destination = select_qkv_destination_shard(weight(), shard_id, local_q_size_, local_kv_size_);
    copy_tensor_to_parameter(source_shard, destination, allocator, stream);
}

void QKVParallelLinear::bias_loader(const Tensor& source,
                                    DeviceAllocator& allocator,
                                    cudaStream_t stream) {
    linear_.bias_loader(source, allocator, stream);
}

void QKVParallelLinear::bias_loader(const Tensor& source,
                                    int shard_id,
                                    DeviceAllocator& allocator,
                                    cudaStream_t stream) {
    if (source.dim() != 1) {
        throw std::invalid_argument("qkv shard bias must have rank 1");
    }
    const int64_t local_size = shard_id == 0 ? local_q_size_ : local_kv_size_;
    int64_t shard_full_size = 0;
    switch (shard_id) {
    case 0:
        shard_full_size = full_q_size_;
        break;
    case 1:
    case 2:
        shard_full_size = full_kv_size_;
        break;
    default:
        throw std::invalid_argument("QKV shard_id must be 0 (q), 1 (k), or 2 (v)");
    }
    const Tensor source_shard = select_output_partition(source,
                                                        shard_full_size,
                                                        local_size,
                                                        rank_,
                                                        "qkv shard bias");
    const Tensor destination = select_qkv_destination_shard(bias(), shard_id, local_q_size_, local_kv_size_);
    copy_tensor_to_parameter(source_shard, destination, allocator, stream);
}

MergedColumnParallelLinear::MergedColumnParallelLinear(int input_size,
                                                       const std::vector<int>& output_sizes,
                                                       bool use_bias,
                                                       ScalarType dtype,
                                                       Device device,
                                                       DeviceAllocator& allocator,
                                                       int tp_size,
                                                       int rank)
        : output_sizes_(output_sizes),
            tp_size_(tp_size),
            rank_(rank),
            linear_(input_size,
              output_sizes.empty() ? 0 : std::accumulate(output_sizes.begin(), output_sizes.end(), 0),
              use_bias,
              dtype,
              device,
              allocator,
              tp_size,
              rank) {
    validate_tp_rank("merged column parallel linear", tp_size_, rank_);
    int offset = 0;
    local_output_sizes_.reserve(output_sizes_.size());
    shard_offsets_.reserve(output_sizes_.size());
    for (int output_size : output_sizes_) {
        if (output_size % tp_size_ != 0) {
            throw std::invalid_argument("each merged linear shard size must be divisible by tp_size");
        }
        const int local_size = output_size / tp_size_;
        local_output_sizes_.push_back(local_size);
        shard_offsets_.push_back(offset);
        offset += local_size;
    }
}

Tensor MergedColumnParallelLinear::forward(const Tensor& input,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream) const {
    return linear_.forward(input, allocator, stream);
}

void MergedColumnParallelLinear::weight_loader(const Tensor& source,
                                               DeviceAllocator& allocator,
                                               cudaStream_t stream) {
    linear_.weight_loader(source, allocator, stream);
}

void MergedColumnParallelLinear::weight_loader(const Tensor& source,
                                               int shard_id,
                                               DeviceAllocator& allocator,
                                               cudaStream_t stream) {
    if (source.dim() != 2 || source.sizes()[1] != weight().sizes()[1]) {
        throw std::invalid_argument("merged shard weight must have shape [rows, input_size]");
    }
    if (shard_id < 0 || shard_id >= static_cast<int>(output_sizes_.size())) {
        throw std::invalid_argument("merged linear shard_id is out of range");
    }
    const Tensor source_shard = select_output_partition(source,
                                                        output_sizes_[static_cast<size_t>(shard_id)],
                                                        local_output_sizes_[static_cast<size_t>(shard_id)],
                                                        rank_,
                                                        "merged shard weight");
    const Tensor destination = select_merged_destination_shard(weight(), shard_offsets_, local_output_sizes_, shard_id);
    copy_tensor_to_parameter(source_shard, destination, allocator, stream);
}

void MergedColumnParallelLinear::bias_loader(const Tensor& source,
                                             DeviceAllocator& allocator,
                                             cudaStream_t stream) {
    linear_.bias_loader(source, allocator, stream);
}

void MergedColumnParallelLinear::bias_loader(const Tensor& source,
                                             int shard_id,
                                             DeviceAllocator& allocator,
                                             cudaStream_t stream) {
    if (source.dim() != 1) {
        throw std::invalid_argument("merged shard bias must have rank 1");
    }
    if (shard_id < 0 || shard_id >= static_cast<int>(output_sizes_.size())) {
        throw std::invalid_argument("merged linear shard_id is out of range");
    }
    const Tensor source_shard = select_output_partition(source,
                                                        output_sizes_[static_cast<size_t>(shard_id)],
                                                        local_output_sizes_[static_cast<size_t>(shard_id)],
                                                        rank_,
                                                        "merged shard bias");
    const Tensor destination = select_merged_destination_shard(bias(), shard_offsets_, local_output_sizes_, shard_id);
    copy_tensor_to_parameter(source_shard, destination, allocator, stream);
}

} // namespace nano_vllm