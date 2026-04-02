#pragma once

#include "utils/tensor.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nano_vllm {

class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;

    virtual Memory::Ptr allocate(size_t bytes, Device device) = 0;
    virtual void memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) const = 0;
    virtual void copy_to_device_async(void* dst,
                                      const void* src,
                                      size_t bytes,
                                      cudaStream_t stream) const = 0;
    virtual void copy_to_host_async(void* dst,
                                    const void* src,
                                    size_t bytes,
                                    cudaStream_t stream) const = 0;
    virtual void synchronize_stream(cudaStream_t stream) const = 0;
};

class CudaAllocator final : public DeviceAllocator {
public:
    static CudaAllocator& instance();

    Memory::Ptr allocate(size_t bytes, Device device) override;
    void memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) const override;
    void copy_to_device_async(void* dst,
                              const void* src,
                              size_t bytes,
                              cudaStream_t stream) const override;
    void copy_to_host_async(void* dst,
                            const void* src,
                            size_t bytes,
                            cudaStream_t stream) const override;
    void synchronize_stream(cudaStream_t stream) const override;

private:
    CudaAllocator() = default;

    // Round up to a bucket size to reduce fragmentation.
    static size_t round_size(size_t bytes);

    // Return a block to the cache instead of calling cudaFree.
    void return_block(Device device, void* ptr, size_t bucket_size);

    mutable std::mutex mutex_;
    // bucket_size -> list of free raw pointers.
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(size_t bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const { return data_; }
    size_t bytes() const { return bytes_; }

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace nano_vllm