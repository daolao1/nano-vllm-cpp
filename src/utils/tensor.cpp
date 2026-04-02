#include "utils/tensor.h"

#include "utils/cuda_allocator.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace nano_vllm {

namespace {

void validate_sizes(const std::vector<int64_t>& sizes) {
    for (int64_t size : sizes) {
        if (size < 0) {
            throw std::invalid_argument("tensor sizes must be non-negative");
        }
    }
}

void validate_strides(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
    if (sizes.size() != strides.size()) {
        throw std::invalid_argument("tensor sizes and strides must have the same rank");
    }
}

} // namespace

bool operator==(const Device& lhs, const Device& rhs) {
    return lhs.type == rhs.type && lhs.index == rhs.index;
}

bool operator!=(const Device& lhs, const Device& rhs) {
    return !(lhs == rhs);
}

std::string to_string(DeviceType device_type) {
    switch (device_type) {
    case DeviceType::kCPU:
        return "cpu";
    case DeviceType::kCUDA:
        return "cuda";
    }
    throw std::invalid_argument("unsupported device type");
}

std::string to_string(ScalarType scalar_type) {
    switch (scalar_type) {
    case ScalarType::kFloat16:
        return "float16";
    case ScalarType::kBFloat16:
        return "bfloat16";
    case ScalarType::kFloat32:
        return "float32";
    case ScalarType::kInt32:
        return "int32";
    case ScalarType::kInt64:
        return "int64";
    case ScalarType::kInt8:
        return "int8";
    }
    throw std::invalid_argument("unsupported scalar type");
}

size_t scalar_type_size(ScalarType scalar_type) {
    switch (scalar_type) {
    case ScalarType::kFloat16:
    case ScalarType::kBFloat16:
        return 2;
    case ScalarType::kFloat32:
    case ScalarType::kInt32:
        return 4;
    case ScalarType::kInt64:
        return 8;
    case ScalarType::kInt8:
        return 1;
    }
    throw std::invalid_argument("unsupported scalar type");
}

Memory::Memory(void* data,
               size_t bytes,
               Device device,
               std::function<void(void*)> deleter)
    : data_(data), bytes_(bytes), device_(device), deleter_(std::move(deleter)) {}

Memory::Ptr Memory::make_owned(void* data,
                               size_t bytes,
                               Device device,
                               std::function<void(void*)> deleter) {
    return Memory::Ptr(new Memory(data, bytes, device, std::move(deleter)));
}

Memory::Ptr Memory::make_borrowed(void* data, size_t bytes, Device device) {
    return Memory::Ptr(new Memory(data, bytes, device, nullptr));
}

Memory::~Memory() {
    if (data_ != nullptr && deleter_) {
        deleter_(data_);
    }
}

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& sizes) {
    validate_sizes(sizes);

    std::vector<int64_t> strides(sizes.size(), 1);
    if (sizes.empty()) {
        return strides;
    }

    for (int64_t index = static_cast<int64_t>(sizes.size()) - 2; index >= 0; --index) {
        strides[static_cast<size_t>(index)] = strides[static_cast<size_t>(index + 1)] *
                                              sizes[static_cast<size_t>(index + 1)];
    }
    return strides;
}

size_t compute_numel(const std::vector<int64_t>& sizes) {
    validate_sizes(sizes);

    if (sizes.empty()) {
        return 1;
    }

    return std::accumulate(
        sizes.begin(), sizes.end(), static_cast<size_t>(1), [](size_t product, int64_t size) {
            return product * static_cast<size_t>(size);
        });
}

Tensor::Tensor(Memory::Ptr memory,
               std::vector<int64_t> sizes,
               std::vector<int64_t> strides,
               ScalarType dtype,
                             size_t memory_offset)
        : memory_(std::move(memory)),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      dtype_(dtype),
            memory_offset_(memory_offset) {
    validate_sizes(sizes_);
    validate_strides(sizes_, strides_);
}

Tensor Tensor::empty(const std::vector<int64_t>& sizes,
                     ScalarType dtype,
                     Device device,
                     DeviceAllocator& allocator) {
    const size_t bytes = compute_numel(sizes) * scalar_type_size(dtype);
    Memory::Ptr memory = allocator.allocate(bytes, device);
    return Tensor(memory, sizes, make_contiguous_strides(sizes), dtype, 0);
}

Tensor Tensor::zeros(const std::vector<int64_t>& sizes,
                     ScalarType dtype,
                     Device device,
                     DeviceAllocator& allocator,
                     cudaStream_t stream) {
    Tensor tensor = empty(sizes, dtype, device, allocator);
    if (tensor.nbytes() > 0) {
        allocator.memset_async(tensor.data(), 0, tensor.nbytes(), stream);
    }
    return tensor;
}

Tensor Tensor::from_memory(const Memory::Ptr& memory,
                           const std::vector<int64_t>& sizes,
                           ScalarType dtype,
                           size_t memory_offset) {
    return Tensor(memory, sizes, make_contiguous_strides(sizes), dtype, memory_offset);
}

size_t Tensor::numel() const {
    return compute_numel(sizes_);
}

size_t Tensor::element_size() const {
    return scalar_type_size(dtype_);
}

size_t Tensor::nbytes() const {
    return numel() * element_size();
}

bool Tensor::is_contiguous() const {
    return strides_ == make_contiguous_strides(sizes_);
}

Device Tensor::device() const {
    return memory_ ? memory_->device() : Device{};
}

void* Tensor::data() const {
    if (!memory_ || memory_->data() == nullptr) {
        return nullptr;
    }

    auto* base = static_cast<std::byte*>(memory_->data());
    return base + memory_offset_ * element_size();
}

Tensor Tensor::reshape(const std::vector<int64_t>& sizes) const {
    if (!is_contiguous()) {
        throw std::invalid_argument("reshape requires a contiguous tensor");
    }
    if (compute_numel(sizes) != numel()) {
        throw std::invalid_argument("reshape must preserve numel");
    }
    return Tensor(memory_, sizes, make_contiguous_strides(sizes), dtype_, memory_offset_);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
    if (dim < 0 || dim >= static_cast<int64_t>(sizes_.size())) {
        throw std::out_of_range("slice dimension is out of range");
    }

    const int64_t dim_size = sizes_[static_cast<size_t>(dim)];
    if (start < 0 || end < start || end > dim_size) {
        throw std::out_of_range("slice range is out of bounds");
    }

    std::vector<int64_t> sizes = sizes_;
    sizes[static_cast<size_t>(dim)] = end - start;
    const size_t memory_offset =
        memory_offset_ + static_cast<size_t>(start) * static_cast<size_t>(strides_[static_cast<size_t>(dim)]);

    return Tensor(memory_, sizes, strides_, dtype_, memory_offset);
}

} // namespace nano_vllm