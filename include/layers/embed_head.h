#pragma once

#include "utils/cuda_allocator.h"
#include "utils/tensor.h"

#include <string>

namespace nano_vllm {

class VocabParallelEmbedding {
public:
    VocabParallelEmbedding(int vocab_size,
                           int hidden_dim,
                           ScalarType dtype,
                           Device device,
                           DeviceAllocator& allocator,
                           int tp_size = 1,
                           int rank = 0,
                           std::string tp_group_key = "");

    Tensor forward(const Tensor& input_ids,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);

    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }

private:
    int vocab_size_ = 0;
    int hidden_dim_ = 0;
    int tp_size_ = 1;
    int rank_ = 0;
    int local_vocab_size_ = 0;
    int vocab_start_ = 0;
    int vocab_end_ = 0;
    std::string tp_group_key_;
    Tensor weight_;
};

class ParallelLMHead {
public:
    ParallelLMHead(int vocab_size,
                   int hidden_dim,
                   ScalarType dtype,
                   Device device,
                   DeviceAllocator& allocator,
                   int tp_size = 1,
                   int rank = 0,
                   std::string tp_group_key = "");

    Tensor forward(const Tensor& hidden_states,
                   DeviceAllocator& allocator,
                   cudaStream_t stream = nullptr) const;

    void weight_loader(const Tensor& source,
                       DeviceAllocator& allocator,
                       cudaStream_t stream = nullptr);

    void set_weight(const Tensor& weight) { weight_ = weight; }
    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }

private:
    int vocab_size_ = 0;
    int hidden_dim_ = 0;
    int tp_size_ = 1;
    int rank_ = 0;
    int local_vocab_size_ = 0;
    std::string tp_group_key_;
    Tensor weight_;
};

} // namespace nano_vllm