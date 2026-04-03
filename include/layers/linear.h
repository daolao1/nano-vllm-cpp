#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <string>
#include <vector>

namespace nano_vllm {

Tensor linear_forward(const Tensor& input,
                      const Tensor& weight,
                      const Tensor& bias,
                      DeviceAllocator& allocator,
                      cudaStream_t stream = nullptr);

class ColumnParallelLinear {
public:
    ColumnParallelLinear(int input_size,
                         int output_size,
                         bool use_bias,
                         ScalarType dtype,
                         Device device,
                         DeviceAllocator& allocator,
                         int tp_size = 1,
                         int rank = 0);

    Tensor forward(const Tensor& input,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);
    void bias_loader(const Tensor& source,
                     DeviceAllocator& allocator,
                     cudaStream_t stream = nullptr);

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }

private:
    int input_size_ = 0;
    int output_size_ = 0;
    int local_output_size_ = 0;
    int tp_size_ = 1;
    int rank_ = 0;
    bool use_bias_ = false;
    Tensor weight_;
    Tensor bias_;
};

class RowParallelLinear {
public:
    RowParallelLinear(int input_size,
                      int output_size,
                      bool use_bias,
                      ScalarType dtype,
                      Device device,
                      DeviceAllocator& allocator,
                      int tp_size = 1,
                      int rank = 0,
                      std::string tp_group_key = "");

    Tensor forward(const Tensor& input,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);
    void bias_loader(const Tensor& source,
                     DeviceAllocator& allocator,
                     cudaStream_t stream = nullptr);

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }

private:
    int local_input_size_ = 0;
    int output_size_ = 0;
    int tp_size_ = 1;
    int rank_ = 0;
    bool use_bias_ = false;
    std::string tp_group_key_;
    Tensor weight_;
    Tensor bias_;
};

class QKVParallelLinear {
public:
    QKVParallelLinear(int hidden_size,
                      int head_size,
                      int num_heads,
                      int num_kv_heads,
                      bool use_bias,
                      ScalarType dtype,
                      Device device,
                      DeviceAllocator& allocator,
                      int tp_size = 1,
                      int rank = 0);

    Tensor forward(const Tensor& input,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);
    void weight_loader(const Tensor& source,
                       int shard_id,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);
    void bias_loader(const Tensor& source,
                     DeviceAllocator& allocator,
                     cudaStream_t stream = nullptr);
    void bias_loader(const Tensor& source,
                     int shard_id,
                     DeviceAllocator& allocator,
                     cudaStream_t stream = nullptr);

    Tensor& weight() { return linear_.weight(); }
    const Tensor& weight() const { return linear_.weight(); }
    Tensor& bias() { return linear_.bias(); }
    const Tensor& bias() const { return linear_.bias(); }

private:
    int full_q_size_ = 0;
    int full_kv_size_ = 0;
    int local_q_size_ = 0;
    int local_kv_size_ = 0;
    int tp_size_ = 1;
    int rank_ = 0;
    ColumnParallelLinear linear_;
};

class MergedColumnParallelLinear {
public:
    MergedColumnParallelLinear(int input_size,
                               const std::vector<int>& output_sizes,
                               bool use_bias,
                               ScalarType dtype,
                               Device device,
                               DeviceAllocator& allocator,
                               int tp_size = 1,
                               int rank = 0);

    Tensor forward(const Tensor& input,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);
    void weight_loader(const Tensor& source,
                       int shard_id,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);
    void bias_loader(const Tensor& source,
                     DeviceAllocator& allocator,
                     cudaStream_t stream = nullptr);
    void bias_loader(const Tensor& source,
                     int shard_id,
                     DeviceAllocator& allocator,
                     cudaStream_t stream = nullptr);

    Tensor& weight() { return linear_.weight(); }
    const Tensor& weight() const { return linear_.weight(); }
    Tensor& bias() { return linear_.bias(); }
    const Tensor& bias() const { return linear_.bias(); }

private:
    std::vector<int> output_sizes_;
    std::vector<int> local_output_sizes_;
    std::vector<int> shard_offsets_;
    int tp_size_ = 1;
    int rank_ = 0;
    ColumnParallelLinear linear_;
};

} // namespace nano_vllm