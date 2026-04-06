#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace nano_vllm {

class TensorParallelCommunicator {
public:
    virtual ~TensorParallelCommunicator() = default;

    virtual void all_reduce_sum(const std::string& group_key,
                                int tp_size,
                                int rank,
                                Tensor& tensor,
                                DeviceAllocator& allocator,
                                cudaStream_t stream = nullptr) = 0;

    virtual Tensor gather_last_dim_to_rank0(const std::string& group_key,
                                            int tp_size,
                                            int rank,
                                            const Tensor& local_tensor,
                                            int64_t full_last_dim,
                                            DeviceAllocator& allocator,
                                            cudaStream_t stream = nullptr) = 0;
};

void set_tensor_parallel_communicator(std::shared_ptr<TensorParallelCommunicator> communicator);
std::shared_ptr<TensorParallelCommunicator> get_tensor_parallel_communicator();
void reset_tensor_parallel_communicator();

std::shared_ptr<TensorParallelCommunicator> make_in_process_tensor_parallel_communicator();
bool nccl_tensor_parallel_available();
std::shared_ptr<TensorParallelCommunicator> make_nccl_tensor_parallel_communicator(
    const std::vector<int>& device_indices);

} // namespace nano_vllm