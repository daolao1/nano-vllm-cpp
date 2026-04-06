#include "utils/tensor_parallel.h"

#include "layers/kernel_ops.h"
#include "utils/cuda_common.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if NANO_VLLM_USE_NCCL
#include <nccl.h>
#endif

namespace nano_vllm {
namespace {

class ScopedCudaDevice {
public:
    explicit ScopedCudaDevice(int device_index) {
        throw_if_cuda_error(cudaGetDevice(&previous_device_), "cudaGetDevice");
        if (previous_device_ != device_index) {
            throw_if_cuda_error(cudaSetDevice(device_index), "cudaSetDevice");
            restore_ = true;
        }
    }

    ~ScopedCudaDevice() {
        if (restore_) {
            cudaSetDevice(previous_device_);
        }
    }

    ScopedCudaDevice(const ScopedCudaDevice&) = delete;
    ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

private:
    int previous_device_ = 0;
    bool restore_ = false;
};

void validate_group_key(const std::string& group_key) {
    if (group_key.empty()) {
        throw std::invalid_argument("tensor parallel communicator requires a non-empty group_key");
    }
}

void validate_rank(int tp_size, int rank) {
    if (tp_size <= 0) {
        throw std::invalid_argument("tp_size must be positive");
    }
    if (rank < 0 || rank >= tp_size) {
        throw std::invalid_argument("rank must satisfy 0 <= rank < tp_size");
    }
}

void ensure_defined_cuda_contiguous(const char* name, const Tensor& tensor) {
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

void ensure_tensor_on_rank_device(const Tensor& tensor, int expected_device, const char* op_name) {
    if (tensor.device().index != expected_device) {
        throw std::invalid_argument(std::string(op_name) + " expects tensor.device().index to match the rank device");
    }
}

void ensure_same_metadata(const Tensor& lhs, const Tensor& rhs, const char* op_name) {
    if (lhs.sizes() != rhs.sizes()) {
        throw std::invalid_argument(std::string(op_name) + " requires matching tensor shapes");
    }
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument(std::string(op_name) + " requires matching tensor dtypes");
    }
    if (lhs.device() != rhs.device()) {
        throw std::invalid_argument(std::string(op_name) + " requires matching tensor devices");
    }
}

void device_copy(const Tensor& source, const Tensor& destination, cudaStream_t stream) {
    ensure_same_metadata(source, destination, "device copy");
    if (source.nbytes() == 0) {
        return;
    }
    throw_if_cuda_error(cudaMemcpyAsync(destination.data(),
                                        source.data(),
                                        source.nbytes(),
                                        cudaMemcpyDeviceToDevice,
                                        stream),
                        "cudaMemcpyAsync device-to-device");
}

void copy_last_dim_shard(const Tensor& source,
                        Tensor& destination,
                        int rank,
                        cudaStream_t stream) {
    ensure_defined_cuda_contiguous("source", source);
    ensure_defined_cuda_contiguous("destination", destination);
    if (source.dim() != destination.dim() || source.dim() == 0) {
        throw std::invalid_argument("gather_last_dim_to_rank0 requires tensors with the same rank >= 1");
    }
    if (source.dtype() != destination.dtype() || source.device() != destination.device()) {
        throw std::invalid_argument("gather_last_dim_to_rank0 requires matching dtype and device");
    }

    const std::vector<int64_t>& source_sizes = source.sizes();
    const std::vector<int64_t>& destination_sizes = destination.sizes();
    const int64_t source_last_dim = source_sizes.back();
    const int64_t destination_last_dim = destination_sizes.back();
    if (source_last_dim <= 0 || destination_last_dim <= 0) {
        return;
    }
    for (int64_t dim = 0; dim < source.dim() - 1; ++dim) {
        if (source_sizes[static_cast<size_t>(dim)] != destination_sizes[static_cast<size_t>(dim)]) {
            throw std::invalid_argument("gather_last_dim_to_rank0 requires matching leading dimensions");
        }
    }

    const int64_t rows = static_cast<int64_t>(source.numel()) / source_last_dim;
    const size_t element_size = source.element_size();
    const size_t row_bytes = static_cast<size_t>(source_last_dim) * element_size;
    const size_t destination_pitch = static_cast<size_t>(destination_last_dim) * element_size;
    const size_t column_offset = static_cast<size_t>(rank) * static_cast<size_t>(source_last_dim) * element_size;
    char* destination_base = static_cast<char*>(destination.data()) + column_offset;
    throw_if_cuda_error(cudaMemcpy2DAsync(destination_base,
                                          destination_pitch,
                                          source.data(),
                                          row_bytes,
                                          row_bytes,
                                          static_cast<size_t>(rows),
                                          cudaMemcpyDeviceToDevice,
                                          stream),
                        "cudaMemcpy2DAsync gather shard");
}

struct PendingAllReduce {
    PendingAllReduce() = default;
    explicit PendingAllReduce(int tp_size_in) : tensors(static_cast<size_t>(tp_size_in)) {}

    std::vector<Tensor> tensors;
    int arrived = 0;
};

struct PendingGather {
    PendingGather() = default;
    PendingGather(int tp_size_in, int64_t full_last_dim_in)
        : local_tensors(static_cast<size_t>(tp_size_in)),
          full_last_dim(full_last_dim_in) {}

    std::vector<Tensor> local_tensors;
    Tensor rank0_output;
    int64_t full_last_dim = 0;
    int arrived = 0;
};

class InProcessTensorParallelCommunicator final : public TensorParallelCommunicator {
public:
    void all_reduce_sum(const std::string& group_key,
                        int tp_size,
                        int rank,
                        Tensor& tensor,
                        DeviceAllocator& allocator,
                        cudaStream_t stream) override {
        (void)allocator;
        validate_group_key(group_key);
        validate_rank(tp_size, rank);
        ensure_defined_cuda_contiguous("tensor", tensor);

        std::lock_guard<std::mutex> lock(mutex_);
        PendingAllReduce& pending = pending_all_reduce_[group_key];
        if (pending.tensors.empty()) {
            pending = PendingAllReduce(tp_size);
        } else if (static_cast<int>(pending.tensors.size()) != tp_size) {
            throw std::invalid_argument("inconsistent tp_size for all_reduce group");
        }
        if (pending.tensors[static_cast<size_t>(rank)].defined()) {
            throw std::invalid_argument("duplicate rank in all_reduce group");
        }

        pending.tensors[static_cast<size_t>(rank)] = tensor;
        ++pending.arrived;
        if (pending.arrived != tp_size) {
            return;
        }

        Tensor& accumulator = pending.tensors.front();
        for (int current_rank = 1; current_rank < tp_size; ++current_rank) {
            const Tensor& shard = pending.tensors[static_cast<size_t>(current_rank)];
            ensure_same_metadata(accumulator, shard, "all_reduce_sum");
            add_inplace(accumulator, shard, stream);
        }
        for (int current_rank = 1; current_rank < tp_size; ++current_rank) {
            device_copy(accumulator, pending.tensors[static_cast<size_t>(current_rank)], stream);
        }
        pending_all_reduce_.erase(group_key);
    }

    Tensor gather_last_dim_to_rank0(const std::string& group_key,
                                    int tp_size,
                                    int rank,
                                    const Tensor& local_tensor,
                                    int64_t full_last_dim,
                                    DeviceAllocator& allocator,
                                    cudaStream_t stream) override {
        validate_group_key(group_key);
        validate_rank(tp_size, rank);
        ensure_defined_cuda_contiguous("local_tensor", local_tensor);
        if (local_tensor.dim() == 0) {
            throw std::invalid_argument("gather_last_dim_to_rank0 requires rank >= 1");
        }
        if (full_last_dim <= 0) {
            throw std::invalid_argument("full_last_dim must be positive");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        PendingGather& pending = pending_gather_[group_key];
        if (pending.local_tensors.empty()) {
            pending = PendingGather(tp_size, full_last_dim);
        } else if (static_cast<int>(pending.local_tensors.size()) != tp_size || pending.full_last_dim != full_last_dim) {
            throw std::invalid_argument("inconsistent gather metadata for group");
        }
        if (pending.local_tensors[static_cast<size_t>(rank)].defined()) {
            throw std::invalid_argument("duplicate rank in gather group");
        }

        pending.local_tensors[static_cast<size_t>(rank)] = local_tensor;
        ++pending.arrived;

        if (rank == 0) {
            std::vector<int64_t> gathered_sizes = local_tensor.sizes();
            gathered_sizes.back() = full_last_dim;
            pending.rank0_output = Tensor::zeros(gathered_sizes,
                                                 local_tensor.dtype(),
                                                 local_tensor.device(),
                                                 allocator,
                                                 stream);
        }

        if (pending.arrived == tp_size) {
            if (!pending.rank0_output.defined()) {
                std::vector<int64_t> gathered_sizes = pending.local_tensors.front().sizes();
                gathered_sizes.back() = full_last_dim;
                pending.rank0_output = Tensor::zeros(gathered_sizes,
                                                     pending.local_tensors.front().dtype(),
                                                     pending.local_tensors.front().device(),
                                                     allocator,
                                                     stream);
            }
            for (int current_rank = 0; current_rank < tp_size; ++current_rank) {
                copy_last_dim_shard(pending.local_tensors[static_cast<size_t>(current_rank)],
                                    pending.rank0_output,
                                    current_rank,
                                    stream);
            }
        }

        Tensor result = rank == 0 ? pending.rank0_output : Tensor();
        if (pending.arrived == tp_size) {
            pending_gather_.erase(group_key);
        }
        return result;
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, PendingAllReduce> pending_all_reduce_;
    std::unordered_map<std::string, PendingGather> pending_gather_;
};

#if NANO_VLLM_USE_NCCL

void throw_if_nccl_error(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}

ncclDataType_t to_nccl_dtype(ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat16:
        return ncclFloat16;
    case ScalarType::kBFloat16:
        return ncclBfloat16;
    case ScalarType::kFloat32:
        return ncclFloat32;
    case ScalarType::kInt32:
        return ncclInt32;
    case ScalarType::kInt64:
        return ncclInt64;
    default:
        throw std::invalid_argument("unsupported dtype for NCCL tensor parallel communicator");
    }
}

class NcclTensorParallelCommunicator final : public TensorParallelCommunicator {
public:
    explicit NcclTensorParallelCommunicator(std::vector<int> device_indices)
        : device_indices_(std::move(device_indices)), comms_(device_indices_.size(), nullptr) {
        if (device_indices_.empty()) {
            throw std::invalid_argument("NCCL tensor parallel communicator requires at least one device");
        }

        std::vector<int> sorted_devices = device_indices_;
        std::sort(sorted_devices.begin(), sorted_devices.end());
        if (std::adjacent_find(sorted_devices.begin(), sorted_devices.end()) != sorted_devices.end()) {
            throw std::invalid_argument("NCCL tensor parallel communicator requires unique device indices");
        }

        throw_if_nccl_error(ncclCommInitAll(comms_.data(), static_cast<int>(device_indices_.size()), device_indices_.data()),
                            "ncclCommInitAll");
    }

    ~NcclTensorParallelCommunicator() override {
        for (size_t index = 0; index < comms_.size(); ++index) {
            if (comms_[index] != nullptr) {
                ScopedCudaDevice guard(device_indices_[index]);
                ncclCommDestroy(comms_[index]);
            }
        }
    }

    void all_reduce_sum(const std::string& group_key,
                        int tp_size,
                        int rank,
                        Tensor& tensor,
                        DeviceAllocator& allocator,
                        cudaStream_t stream) override {
        (void)allocator;
        validate_group_key(group_key);
        validate_rank(tp_size, rank);
        ensure_defined_cuda_contiguous("tensor", tensor);
        validate_tp_size(tp_size);
        ensure_tensor_on_rank_device(tensor, device_indices_[static_cast<size_t>(rank)], "ncclAllReduce");

        ScopedCudaDevice guard(device_indices_[static_cast<size_t>(rank)]);
        throw_if_nccl_error(ncclAllReduce(tensor.data(),
                                          tensor.data(),
                                          tensor.numel(),
                                          to_nccl_dtype(tensor.dtype()),
                                          ncclSum,
                                          comms_[static_cast<size_t>(rank)],
                                          stream),
                            "ncclAllReduce");
    }

    Tensor gather_last_dim_to_rank0(const std::string& group_key,
                                    int tp_size,
                                    int rank,
                                    const Tensor& local_tensor,
                                    int64_t full_last_dim,
                                    DeviceAllocator& allocator,
                                    cudaStream_t stream) override {
        validate_group_key(group_key);
        validate_rank(tp_size, rank);
        ensure_defined_cuda_contiguous("local_tensor", local_tensor);
        validate_tp_size(tp_size);
        ensure_tensor_on_rank_device(local_tensor,
                                     device_indices_[static_cast<size_t>(rank)],
                                     "nccl gather_last_dim_to_rank0");
        if (local_tensor.dim() == 0) {
            throw std::invalid_argument("gather_last_dim_to_rank0 requires rank >= 1");
        }
        if (full_last_dim <= 0) {
            throw std::invalid_argument("full_last_dim must be positive");
        }
        if (full_last_dim % tp_size != 0) {
            throw std::invalid_argument("gather_last_dim_to_rank0 requires full_last_dim divisible by tp_size");
        }
        if (local_tensor.sizes().back() * tp_size != full_last_dim) {
            throw std::invalid_argument("gather_last_dim_to_rank0 local shard width does not match full_last_dim");
        }

        ScopedCudaDevice guard(device_indices_[static_cast<size_t>(rank)]);
        const size_t message_count = local_tensor.numel();
        const ncclDataType_t dtype = to_nccl_dtype(local_tensor.dtype());
        if (rank != 0) {
            throw_if_nccl_error(ncclSend(local_tensor.data(),
                                         message_count,
                                         dtype,
                                         0,
                                         comms_[static_cast<size_t>(rank)],
                                         stream),
                                "ncclSend");
            allocator.synchronize_stream(stream);
            return Tensor();
        }

        std::vector<Tensor> received_shards;
        received_shards.reserve(static_cast<size_t>(tp_size - 1));
        for (int source_rank = 1; source_rank < tp_size; ++source_rank) {
            received_shards.push_back(
                Tensor::zeros(local_tensor.sizes(), local_tensor.dtype(), local_tensor.device(), allocator, stream));
        }

        if (!received_shards.empty()) {
            throw_if_nccl_error(ncclGroupStart(), "ncclGroupStart");
            try {
                for (int source_rank = 1; source_rank < tp_size; ++source_rank) {
                    throw_if_nccl_error(ncclRecv(received_shards[static_cast<size_t>(source_rank - 1)].data(),
                                                 message_count,
                                                 dtype,
                                                 source_rank,
                                                 comms_.front(),
                                                 stream),
                                        "ncclRecv");
                }
            } catch (...) {
                ncclGroupEnd();
                throw;
            }
            throw_if_nccl_error(ncclGroupEnd(), "ncclGroupEnd");
        }

        std::vector<int64_t> gathered_sizes = local_tensor.sizes();
        gathered_sizes.back() = full_last_dim;
        Tensor gathered = Tensor::zeros(gathered_sizes,
                                        local_tensor.dtype(),
                                        local_tensor.device(),
                                        allocator,
                                        stream);
        copy_last_dim_shard(local_tensor, gathered, 0, stream);
        for (int source_rank = 1; source_rank < tp_size; ++source_rank) {
            copy_last_dim_shard(received_shards[static_cast<size_t>(source_rank - 1)], gathered, source_rank, stream);
        }
        return gathered;
    }

private:
    void validate_tp_size(int tp_size) const {
        if (tp_size != static_cast<int>(device_indices_.size())) {
            throw std::invalid_argument("NCCL tensor parallel communicator tp_size does not match device_indices size");
        }
    }

    std::vector<int> device_indices_;
    std::vector<ncclComm_t> comms_;
};

#endif

std::shared_ptr<TensorParallelCommunicator> g_tensor_parallel_communicator;

} // namespace

void set_tensor_parallel_communicator(std::shared_ptr<TensorParallelCommunicator> communicator) {
    g_tensor_parallel_communicator = std::move(communicator);
}

std::shared_ptr<TensorParallelCommunicator> get_tensor_parallel_communicator() {
    return g_tensor_parallel_communicator;
}

void reset_tensor_parallel_communicator() {
    g_tensor_parallel_communicator.reset();
}

std::shared_ptr<TensorParallelCommunicator> make_in_process_tensor_parallel_communicator() {
    return std::make_shared<InProcessTensorParallelCommunicator>();
}

bool nccl_tensor_parallel_available() {
#if NANO_VLLM_USE_NCCL
    return true;
#else
    return false;
#endif
}

std::shared_ptr<TensorParallelCommunicator> make_nccl_tensor_parallel_communicator(
    const std::vector<int>& device_indices) {
#if NANO_VLLM_USE_NCCL
    return std::make_shared<NcclTensorParallelCommunicator>(device_indices);
#else
    (void)device_indices;
    throw std::runtime_error("NCCL tensor parallel communicator is unavailable because nano-vllm-cpp was built without NCCL");
#endif
}

} // namespace nano_vllm