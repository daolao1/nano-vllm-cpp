#pragma once

#include "engine/sequence.h"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nano_vllm {

static constexpr uint64_t kNoHash = UINT64_MAX;

struct Block {
    int block_id = 0;
    int ref_count = 0;
    uint64_t hash = kNoHash;
    std::vector<int32_t> token_ids;

    explicit Block(int id) : block_id(id) {}

    void update(uint64_t h, const std::vector<int32_t>& tids) {
        hash = h;
        token_ids = tids;
    }

    void reset() {
        ref_count = 1;
        hash = kNoHash;
        token_ids.clear();
    }
};

class BlockManager {
public:
    BlockManager(int num_blocks, int block_size);

    static uint64_t compute_hash(const std::vector<int32_t>& token_ids,
                                 uint64_t prefix = kNoHash);

    bool can_allocate(const Sequence& seq) const;
    void allocate(Sequence& seq);
    void deallocate(Sequence& seq);

    bool can_append(const Sequence& seq) const;
    void may_append(Sequence& seq);

    int num_free_blocks() const { return static_cast<int>(free_block_ids_.size()); }
    int num_used_blocks() const { return static_cast<int>(used_block_ids_.size()); }
    int num_total_blocks() const { return static_cast<int>(blocks_.size()); }

private:
    void allocate_block(int block_id);
    void deallocate_block(int block_id);

    int block_size_;
    std::vector<Block> blocks_;
    std::unordered_map<uint64_t, int> hash_to_block_id_;
    std::deque<int> free_block_ids_;
    std::unordered_set<int> used_block_ids_;
};

} // namespace nano_vllm
