#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nano_vllm {

class DeviceAllocator;

enum class DeviceType {
    kCPU,
    kCUDA,
};

enum class ScalarType {
    kFloat16,
    kBFloat16,
    kFloat32,
    kInt32,
    kInt64,
    kInt8,
};

struct Device {
    DeviceType type = DeviceType::kCPU;
    int index = 0;
};

bool operator==(const Device& lhs, const Device& rhs);
bool operator!=(const Device& lhs, const Device& rhs);

std::string to_string(DeviceType device_type);
std::string to_string(ScalarType scalar_type);
size_t scalar_type_size(ScalarType scalar_type);

class Memory {
public:
    using Ptr = std::shared_ptr<Memory>;

    static Ptr make_owned(void* data,
                          size_t bytes,
                          Device device,
                          std::function<void(void*)> deleter);
    static Ptr make_borrowed(void* data, size_t bytes, Device device);

    ~Memory();

    void* data() const { return data_; }
    size_t bytes() const { return bytes_; }
    Device device() const { return device_; }

private:
        Memory(void* data,
            size_t bytes,
            Device device,
            std::function<void(void*)> deleter);

    void* data_ = nullptr;
    size_t bytes_ = 0;
    Device device_{};
    std::function<void(void*)> deleter_;
};

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& sizes);
size_t compute_numel(const std::vector<int64_t>& sizes);

class Tensor {
public:
    Tensor() = default;

    static Tensor empty(const std::vector<int64_t>& sizes,
                        ScalarType dtype,
                        Device device,
                        DeviceAllocator& allocator);
    static Tensor zeros(const std::vector<int64_t>& sizes,
                        ScalarType dtype,
                        Device device,
                        DeviceAllocator& allocator,
                        cudaStream_t stream = nullptr);
    static Tensor from_memory(const Memory::Ptr& memory,
                              const std::vector<int64_t>& sizes,
                              ScalarType dtype,
                              size_t memory_offset = 0);

    bool defined() const { return static_cast<bool>(memory_); }
    int64_t dim() const { return static_cast<int64_t>(sizes_.size()); }
    size_t numel() const;
    size_t element_size() const;
    size_t nbytes() const;
    bool is_contiguous() const;

    const std::vector<int64_t>& sizes() const { return sizes_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    ScalarType dtype() const { return dtype_; }
    Device device() const;
    size_t memory_offset() const { return memory_offset_; }
    const Memory::Ptr& memory() const { return memory_; }

    void* data() const;

    template <typename T>
    T* data_as() const {
        return static_cast<T*>(data());
    }

    Tensor reshape(const std::vector<int64_t>& sizes) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end) const;

private:
        Tensor(Memory::Ptr memory,
            std::vector<int64_t> sizes,
            std::vector<int64_t> strides,
            ScalarType dtype,
            size_t memory_offset);

        Memory::Ptr memory_;
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    ScalarType dtype_ = ScalarType::kFloat32;
        size_t memory_offset_ = 0;  // measured in elements
};

} // namespace nano_vllm