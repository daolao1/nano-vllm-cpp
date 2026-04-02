#include "utils/cuda_allocator.h"

#include "utils/cuda_common.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace nano_vllm {

namespace {

void maybe_set_device(Device device) {
    if (device.type == DeviceType::kCUDA) {
        throw_if_cuda_error(cudaSetDevice(device.index), "cudaSetDevice");
    }
}

void release_cuda_ptr(Device device, void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    if (device.type == DeviceType::kCUDA) {
        cudaSetDevice(device.index);
        cudaFree(ptr);
    }
}

} // namespace

CudaAllocator& CudaAllocator::instance() {
    static CudaAllocator allocator;
    return allocator;
}

// Round up: <=1MB -> next power of 2; >1MB -> next multiple of 2MB.
size_t CudaAllocator::round_size(size_t bytes) {
    constexpr size_t kSmallThreshold = 1 << 20; // 1 MB
    constexpr size_t kLargeGranularity = 2 << 20; // 2 MB
    if (bytes <= 512) return 512;
    if (bytes <= kSmallThreshold) {
        size_t s = 512;
        while (s < bytes) s <<= 1;
        return s;
    }
    return ((bytes + kLargeGranularity - 1) / kLargeGranularity) * kLargeGranularity;
}

void CudaAllocator::return_block(Device device, void* ptr, size_t bucket_size) {
    (void)device;
    free_blocks_[bucket_size].push_back(ptr);
}

Memory::Ptr CudaAllocator::allocate(size_t bytes, Device device) {
    if (device.type != DeviceType::kCUDA) {
        throw std::invalid_argument("CudaAllocator only allocates CUDA tensors");
    }

    if (bytes == 0) {
        return Memory::make_borrowed(nullptr, 0, device);
    }

    maybe_set_device(device);
    const size_t bucket = round_size(bytes);

    void* ptr = nullptr;
    {
        auto it = free_blocks_.find(bucket);
        if (it != free_blocks_.end() && !it->second.empty()) {
            ptr = it->second.back();
            it->second.pop_back();
        }
    }

    if (ptr == nullptr) {
        throw_if_cuda_error(cudaMalloc(&ptr, bucket), "cudaMalloc");
    }

    return Memory::make_owned(ptr, bucket, device,
                              [this, device, bucket](void* raw_ptr) {
                                  this->return_block(device, raw_ptr, bucket);
                              });
}

void CudaAllocator::memset_async(void* ptr, int value, size_t bytes, cudaStream_t stream) const {
    if (bytes == 0 || ptr == nullptr) {
        return;
    }
    throw_if_cuda_error(cudaMemsetAsync(ptr, value, bytes, stream), "cudaMemsetAsync");
}

void CudaAllocator::copy_to_device_async(void* dst,
                                         const void* src,
                                         size_t bytes,
                                         cudaStream_t stream) const {
    if (bytes == 0) {
        return;
    }
    throw_if_cuda_error(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(host_to_device)");
}

void CudaAllocator::copy_to_host_async(void* dst,
                                       const void* src,
                                       size_t bytes,
                                       cudaStream_t stream) const {
    if (bytes == 0) {
        return;
    }
    throw_if_cuda_error(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(device_to_host)");
}

void CudaAllocator::synchronize_stream(cudaStream_t stream) const {
    throw_if_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

PinnedHostBuffer::PinnedHostBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) {
        return;
    }
    throw_if_cuda_error(cudaHostAlloc(&data_, bytes_, cudaHostAllocDefault), "cudaHostAlloc");
}

PinnedHostBuffer::~PinnedHostBuffer() {
    if (data_ != nullptr) {
        cudaFreeHost(data_);
    }
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (data_ != nullptr) {
        cudaFreeHost(data_);
    }

    data_ = other.data_;
    bytes_ = other.bytes_;
    other.data_ = nullptr;
    other.bytes_ = 0;
    return *this;
}

} // namespace nano_vllm