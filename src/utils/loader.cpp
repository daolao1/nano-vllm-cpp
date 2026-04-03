#include "utils/loader.h"

#include "utils/cuda_allocator.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nano_vllm {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

uint64_t read_le_u64(const std::byte* bytes) {
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<uint64_t>(std::to_integer<unsigned char>(bytes[index])) << (index * 8);
    }
    return value;
}

ScalarType parse_safetensors_dtype(const std::string& dtype) {
    if (dtype == "F16") {
        return ScalarType::kFloat16;
    }
    if (dtype == "BF16") {
        return ScalarType::kBFloat16;
    }
    if (dtype == "F32") {
        return ScalarType::kFloat32;
    }
    if (dtype == "I32") {
        return ScalarType::kInt32;
    }
    if (dtype == "I64") {
        return ScalarType::kInt64;
    }
    throw std::invalid_argument("unsupported SafeTensors dtype: " + dtype);
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            oss << ", ";
        }
        oss << shape[index];
    }
    oss << ']';
    return oss.str();
}

bool matches_parameter_segment(const std::string& name,
                               size_t pos,
                               const std::string& segment) {
    const bool left_ok = pos == 0 || name[pos - 1] == '.';
    const size_t end = pos + segment.size();
    const bool right_ok = end == name.size() || name[end] == '.';
    return left_ok && right_ok;
}

std::vector<float> convert_bfloat16_to_float32(const uint16_t* values, size_t count) {
    std::vector<float> converted(count, 0.0f);
    for (size_t index = 0; index < count; ++index) {
        const uint32_t bits = static_cast<uint32_t>(values[index]) << 16;
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        converted[index] = out;
    }
    return converted;
}

size_t linear_index_to_storage_offset(const Tensor& tensor, size_t linear_index) {
    if (tensor.dim() == 0) {
        return 0;
    }

    size_t storage_offset = 0;
    size_t remaining = linear_index;
    for (int64_t dim = tensor.dim() - 1; dim >= 0; --dim) {
        const int64_t dim_size = tensor.sizes()[static_cast<size_t>(dim)];
        const size_t coord = static_cast<size_t>(remaining % static_cast<size_t>(dim_size));
        remaining /= static_cast<size_t>(dim_size);
        storage_offset += coord * static_cast<size_t>(tensor.strides()[static_cast<size_t>(dim)]);
    }
    return storage_offset;
}

std::vector<std::byte> make_contiguous_cpu_tensor_bytes(const Tensor& tensor) {
    if (tensor.device().type != DeviceType::kCPU) {
        throw std::invalid_argument("only CPU tensors can be materialized into a contiguous host buffer");
    }

    std::vector<std::byte> bytes(tensor.nbytes());
    if (tensor.numel() == 0) {
        return bytes;
    }

    const auto* source = static_cast<const std::byte*>(tensor.data());
    const size_t element_bytes = tensor.element_size();
    for (size_t linear_index = 0; linear_index < tensor.numel(); ++linear_index) {
        const size_t storage_offset = linear_index_to_storage_offset(tensor, linear_index);
        std::memcpy(bytes.data() + linear_index * element_bytes,
                    source + storage_offset * element_bytes,
                    element_bytes);
    }
    return bytes;
}

std::vector<fs::path> collect_safetensors_files(const std::string& path) {
    const fs::path input(path);
    std::vector<fs::path> files;
    if (fs::is_regular_file(input)) {
        if (input.extension() != ".safetensors") {
            throw std::invalid_argument("expected a .safetensors file: " + path);
        }
        files.push_back(input);
        return files;
    }
    if (!fs::is_directory(input)) {
        throw std::invalid_argument("path is neither a .safetensors file nor a directory: " + path);
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(input)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::invalid_argument("no .safetensors files found under: " + path);
    }
    return files;
}

void copy_same_dtype_tensor(const Tensor& source,
                            const Tensor& destination,
                            DeviceAllocator& allocator,
                            cudaStream_t stream) {
    const std::vector<std::byte> contiguous_source =
        source.is_contiguous() ? std::vector<std::byte>{} : make_contiguous_cpu_tensor_bytes(source);
    const void* source_ptr = source.is_contiguous() ? source.data() : contiguous_source.data();

    if (source.device().type == DeviceType::kCPU && destination.device().type == DeviceType::kCUDA) {
        allocator.copy_to_device_async(destination.data(), source_ptr, destination.nbytes(), stream);
        allocator.synchronize_stream(stream);
        return;
    }
    if (source.device().type == DeviceType::kCPU && destination.device().type == DeviceType::kCPU) {
        std::memcpy(destination.data(), source_ptr, destination.nbytes());
        return;
    }
    if (source.device().type == DeviceType::kCUDA && destination.device().type == DeviceType::kCUDA) {
        const cudaError_t err = cudaMemcpyAsync(destination.data(), source.data(),
                                                destination.nbytes(),
                                                cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("copy_tensor_to_parameter device-to-device: ") +
                                     cudaGetErrorString(err));
        }
        allocator.synchronize_stream(stream);
        return;
    }
    throw std::invalid_argument("copy_tensor_to_parameter only supports CPU or same-device CUDA source tensors");
}

} // namespace

SafeTensorsFile SafeTensorsFile::open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("failed to open SafeTensors file: " + path);
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("failed to stat SafeTensors file: " + path);
    }
    if (st.st_size < 8) {
        ::close(fd);
        throw std::invalid_argument("SafeTensors file is too small: " + path);
    }

    void* mapping = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error("mmap failed for SafeTensors file: " + path);
    }

    auto mapping_deleter = [length = static_cast<size_t>(st.st_size)](std::byte* ptr) {
        if (ptr != nullptr) {
            ::munmap(ptr, length);
        }
    };
    std::unique_ptr<std::byte, decltype(mapping_deleter)> mapping_guard(
        static_cast<std::byte*>(mapping), mapping_deleter);

    const uint64_t header_size = read_le_u64(mapping_guard.get());
    const size_t file_size = static_cast<size_t>(st.st_size);
    if (header_size > file_size - 8) {
        throw std::invalid_argument("SafeTensors header exceeds file size: " + path);
    }

    const size_t data_offset = 8 + static_cast<size_t>(header_size);
    const size_t data_size = file_size - data_offset;
    const std::string header(reinterpret_cast<const char*>(mapping_guard.get() + 8), static_cast<size_t>(header_size));
    const json parsed = json::parse(header);
    if (!parsed.is_object()) {
        throw std::invalid_argument("SafeTensors header must be a JSON object: " + path);
    }

    const auto data_memory = Memory::make_owned(
        mapping_guard.get() + data_offset,
        data_size,
        Device{DeviceType::kCPU, 0},
        [base = mapping_guard.get(), file_size](void*) {
            ::munmap(base, file_size);
        });
    mapping_guard.release();

    SafeTensorsFile file;
    file.path_ = path;
    file.data_memory_ = data_memory;
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }
        if (!it.value().is_object()) {
            throw std::invalid_argument("SafeTensors tensor entry must be an object: " + it.key());
        }
        const json& spec = it.value();
        if (!spec.contains("dtype") || !spec.contains("shape") || !spec.contains("data_offsets")) {
            throw std::invalid_argument("SafeTensors tensor entry is missing required fields: " + it.key());
        }

        const auto offsets = spec.at("data_offsets").get<std::vector<size_t>>();
        if (offsets.size() != 2 || offsets[1] < offsets[0]) {
            throw std::invalid_argument("SafeTensors data_offsets must contain [begin, end]: " + it.key());
        }

        SafeTensorsFile::Entry entry;
        entry.dtype = parse_safetensors_dtype(spec.at("dtype").get<std::string>());
        entry.shape = spec.at("shape").get<std::vector<int64_t>>();
        entry.data_offset = offsets[0];
        entry.data_size = offsets[1] - offsets[0];

        const size_t expected_bytes = compute_numel(entry.shape) * scalar_type_size(entry.dtype);
        if (entry.data_size != expected_bytes) {
            throw std::invalid_argument("SafeTensors byte size mismatch for " + it.key() +
                                        ": shape " + shape_string(entry.shape));
        }
        if (offsets[1] > data_size) {
            throw std::invalid_argument("SafeTensors data range exceeds file payload for: " + it.key());
        }
        const size_t element_size = scalar_type_size(entry.dtype);
        if (entry.data_offset % element_size != 0) {
            throw std::invalid_argument("SafeTensors data offset is not aligned for: " + it.key());
        }

        entry.tensor = Tensor::from_memory(data_memory,
                                           entry.shape,
                                           entry.dtype,
                                           entry.data_offset / element_size);
        file.keys_.push_back(it.key());
        file.entries_.emplace(it.key(), std::move(entry));
    }

    return file;
}

bool SafeTensorsFile::contains(const std::string& name) const {
    return entries_.find(name) != entries_.end();
}

const SafeTensorsFile::Entry& SafeTensorsFile::entry(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::out_of_range("SafeTensors entry not found: " + name);
    }
    return it->second;
}

const Tensor& SafeTensorsFile::tensor(const std::string& name) const {
    return entry(name).tensor;
}

void ParameterRegistry::register_parameter(const std::string& name, ParameterLoadFn loader) {
    const auto [it, inserted] = parameters_.emplace(name, std::move(loader));
    if (!inserted) {
        throw std::invalid_argument("duplicate parameter registration: " + name);
    }
    (void)it;
}

bool ParameterRegistry::contains(const std::string& name) const {
    return parameters_.find(name) != parameters_.end();
}

void ParameterRegistry::load(const std::string& path,
                             const std::vector<PackedModuleMapping>& packed_modules_mapping,
                             DeviceAllocator& allocator,
                             cudaStream_t stream) const {
    const std::vector<fs::path> files = collect_safetensors_files(path);
    for (const fs::path& file_path : files) {
        const SafeTensorsFile file = SafeTensorsFile::open(file_path.string());
        for (const std::string& source_name : file.keys()) {
            std::optional<int> shard_id;
            const std::string target_name =
                resolve_packed_parameter_name(source_name, packed_modules_mapping, shard_id);
            const auto it = parameters_.find(target_name);
            if (it == parameters_.end()) {
                throw std::invalid_argument("no parameter registered for SafeTensors entry " + source_name +
                                            " -> " + target_name);
            }
            it->second(file.tensor(source_name), shard_id, allocator, stream);
        }
    }
}

void copy_tensor_to_parameter(const Tensor& source,
                              const Tensor& destination,
                              DeviceAllocator& allocator,
                              cudaStream_t stream) {
    if (!source.defined() || !destination.defined()) {
        throw std::invalid_argument("copy_tensor_to_parameter requires defined source and destination tensors");
    }
    if (!destination.is_contiguous()) {
        throw std::invalid_argument("copy_tensor_to_parameter requires a contiguous destination tensor");
    }
    if (source.sizes() != destination.sizes()) {
        throw std::invalid_argument("copy_tensor_to_parameter shape mismatch: source=" +
                                    shape_string(source.sizes()) + " destination=" +
                                    shape_string(destination.sizes()));
    }

    if (source.dtype() == destination.dtype()) {
        copy_same_dtype_tensor(source, destination, allocator, stream);
        return;
    }

    if (source.dtype() == ScalarType::kBFloat16 && destination.dtype() == ScalarType::kFloat32) {
        if (source.device().type != DeviceType::kCPU) {
            throw std::invalid_argument("BF16 conversion currently expects a CPU source tensor");
        }
        const std::vector<std::byte> contiguous_source =
            source.is_contiguous() ? std::vector<std::byte>{} : make_contiguous_cpu_tensor_bytes(source);
        const auto* source_ptr = static_cast<const uint16_t*>(source.is_contiguous() ? source.data() : contiguous_source.data());
        const std::vector<float> converted = convert_bfloat16_to_float32(source_ptr, source.numel());
        if (destination.device().type == DeviceType::kCUDA) {
            allocator.copy_to_device_async(destination.data(), converted.data(), destination.nbytes(), stream);
            allocator.synchronize_stream(stream);
            return;
        }
        if (destination.device().type == DeviceType::kCPU) {
            std::memcpy(destination.data(), converted.data(), destination.nbytes());
            return;
        }
    }

    throw std::invalid_argument("unsupported tensor copy conversion from " +
                                to_string(source.dtype()) + " to " +
                                to_string(destination.dtype()));
}

Tensor select_tensor_shard(const Tensor& source,
                           int64_t dim,
                           int64_t offset,
                           int64_t size) {
    if (dim < 0 || dim >= source.dim()) {
        throw std::out_of_range("select_tensor_shard dimension is out of range");
    }
    if (offset < 0 || size < 0 || offset + size > source.sizes()[static_cast<size_t>(dim)]) {
        throw std::out_of_range("select_tensor_shard range is out of bounds");
    }
    if (offset == 0 && size == source.sizes()[static_cast<size_t>(dim)]) {
        return source;
    }
    return source.slice(dim, offset, offset + size);
}

std::string resolve_packed_parameter_name(const std::string& source_name,
                                          const std::vector<PackedModuleMapping>& packed_modules_mapping,
                                          std::optional<int>& shard_id) {
    shard_id.reset();
    for (const PackedModuleMapping& mapping : packed_modules_mapping) {
        const size_t pos = source_name.find(mapping.source_name);
        if (pos == std::string::npos || !matches_parameter_segment(source_name, pos, mapping.source_name)) {
            continue;
        }
        std::string resolved = source_name;
        resolved.replace(pos, mapping.source_name.size(), mapping.target_name);
        shard_id = mapping.shard_id;
        return resolved;
    }
    return source_name;
}

} // namespace nano_vllm