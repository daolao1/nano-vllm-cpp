#include "engine/block_manager.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include <xxhash.h>

namespace nano_vllm {

// ---- 构造 ----

BlockManager::BlockManager(int num_blocks, int block_size)
    : block_size_(block_size) {
    blocks_.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(i);
        free_block_ids_.push_back(i);
    }
}

// ---- 哈希链 ----

uint64_t BlockManager::compute_hash(const std::vector<int32_t>& token_ids,
                                    uint64_t prefix) {
    XXH64_state_t* state = XXH64_createState();
    XXH64_reset(state, 0);
    if (prefix != kNoHash) {
        // Python: prefix.to_bytes(8, "little") — 小端 8 字节
        uint64_t le = prefix; // x86 已经是小端
        XXH64_update(state, &le, sizeof(le));
    }
    // Python: np.array(token_ids).tobytes() — int32 数组原始字节
    XXH64_update(state, token_ids.data(),
                 token_ids.size() * sizeof(int32_t));
    uint64_t digest = XXH64_digest(state);
    XXH64_freeState(state);
    return digest;
}

// ---- 内部 block 管理 ----

void BlockManager::allocate_block(int block_id) {
    Block& block = blocks_[block_id];
    assert(block.ref_count == 0);
    block.reset(); // ref_count = 1, hash = kNoHash, token_ids.clear()
    free_block_ids_.erase(
        std::find(free_block_ids_.begin(), free_block_ids_.end(), block_id));
    used_block_ids_.insert(block_id);
}

void BlockManager::deallocate_block(int block_id) {
    assert(blocks_[block_id].ref_count == 0);
    used_block_ids_.erase(block_id);
    free_block_ids_.push_back(block_id);
}

// ---- 公开接口 ----

bool BlockManager::can_allocate(const Sequence& seq) const {
    return static_cast<int>(free_block_ids_.size()) >= seq.num_blocks();
}

void BlockManager::allocate(Sequence& seq) {
    assert(seq.block_table.empty());
    uint64_t h = kNoHash;
    bool cache_miss = false;

    for (int i = 0; i < seq.num_blocks(); ++i) {
        auto token_ids = seq.block(i);

        // 完整 block 才计算 hash
        h = (static_cast<int>(token_ids.size()) == block_size_)
                ? compute_hash(token_ids, h)
                : kNoHash;

        // 查 prefix cache
        int block_id = -1;
        if (!cache_miss) {
            auto it = hash_to_block_id_.find(h);
            if (it != hash_to_block_id_.end()) {
                block_id = it->second;
            }
            if (block_id == -1 ||
                blocks_[block_id].token_ids != token_ids) {
                cache_miss = true;
            }
        }

        if (cache_miss) {
            // 分配新 block
            block_id = free_block_ids_.front();
            allocate_block(block_id);
        } else {
            // cache hit
            seq.num_cached_tokens += block_size_;
            if (used_block_ids_.count(block_id)) {
                blocks_[block_id].ref_count++;
            } else {
                allocate_block(block_id);
            }
        }

        if (h != kNoHash) {
            blocks_[block_id].update(h, token_ids);
            hash_to_block_id_[h] = block_id;
        }
        seq.block_table.push_back(block_id);
    }
}

void BlockManager::deallocate(Sequence& seq) {
    for (auto it = seq.block_table.rbegin(); it != seq.block_table.rend();
         ++it) {
        int block_id = *it;
        Block& block = blocks_[block_id];
        block.ref_count--;
        if (block.ref_count == 0) {
            deallocate_block(block_id);
        }
    }
    seq.num_cached_tokens = 0;
    seq.block_table.clear();
}

bool BlockManager::can_append(const Sequence& seq) const {
    int need = (seq.len() % block_size_ == 1) ? 1 : 0;
    return static_cast<int>(free_block_ids_.size()) >= need;
}

void BlockManager::may_append(Sequence& seq) {
    auto& block_table = seq.block_table;
    Block& last_block = blocks_[block_table.back()];

    int rem = seq.len() % block_size_;
    if (rem == 1) {
        // 上一个 block 已满（hash 已注册），需要分配新 block
        assert(last_block.hash != kNoHash);
        int block_id = free_block_ids_.front();
        allocate_block(block_id);
        block_table.push_back(block_id);
    } else if (rem == 0) {
        // 最后一个 block 刚好填满，注册 hash
        assert(last_block.hash == kNoHash);
        auto token_ids = seq.block(seq.num_blocks() - 1);
        uint64_t prefix = (block_table.size() > 1)
                              ? blocks_[block_table[block_table.size() - 2]].hash
                              : kNoHash;
        uint64_t h = compute_hash(token_ids, prefix);
        last_block.update(h, token_ids);
        hash_to_block_id_[h] = last_block.block_id;
    } else {
        // 未满，什么都不做
        assert(last_block.hash == kNoHash);
    }
}

} // namespace nano_vllm
