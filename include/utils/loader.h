#pragma once

#include "utils/tensor.h"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nano_vllm {

class DeviceAllocator;

struct PackedModuleMapping {
    std::string source_name;
    std::string target_name;
    int shard_id = 0;
};

class SafeTensorsFile {
public:
    struct Entry {
        ScalarType dtype = ScalarType::kFloat32;
        std::vector<int64_t> shape;
        size_t data_offset = 0;
        size_t data_size = 0;
        Tensor tensor;
    };

    static SafeTensorsFile open(const std::string& path);

    const std::string& path() const { return path_; }
    const std::vector<std::string>& keys() const { return keys_; }
    bool contains(const std::string& name) const;
    const Entry& entry(const std::string& name) const;
    const Tensor& tensor(const std::string& name) const;

private:
    std::string path_;
    Memory::Ptr data_memory_;
    std::vector<std::string> keys_;
    std::unordered_map<std::string, Entry> entries_;
};

using ParameterLoadFn = std::function<void(const Tensor& source,
                                           std::optional<int> shard_id,
                                           DeviceAllocator& allocator,
                                           cudaStream_t stream)>;

class ParameterRegistry {
public:
    void register_parameter(const std::string& name, ParameterLoadFn loader);
    bool contains(const std::string& name) const;
    void load(const std::string& path,
              const std::vector<PackedModuleMapping>& packed_modules_mapping,
              DeviceAllocator& allocator,
              cudaStream_t stream = nullptr) const;

private:
    std::unordered_map<std::string, ParameterLoadFn> parameters_;
};

void copy_tensor_to_parameter(const Tensor& source,
                              const Tensor& destination,
                              DeviceAllocator& allocator,
                              cudaStream_t stream = nullptr);

Tensor select_tensor_shard(const Tensor& source,
                           int64_t dim,
                           int64_t offset,
                           int64_t size);

std::string resolve_packed_parameter_name(const std::string& source_name,
                                          const std::vector<PackedModuleMapping>& packed_modules_mapping,
                                          std::optional<int>& shard_id);

} // namespace nano_vllm