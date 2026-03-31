#include "engine/block_manager.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace nano_vllm;

// ============================================================
// 辅助：创建含 n 个 token 的序列
// ============================================================
static Sequence make_seq(int n_tokens, float temp = 1.0f) {
    std::vector<int32_t> ids(n_tokens);
    std::iota(ids.begin(), ids.end(), 0); // 0,1,2,...
    SamplingParams sp{};
    sp.temperature = temp;
    return Sequence(std::move(ids), sp);
}

static Sequence make_seq_from(const std::vector<int32_t>& ids) {
    SamplingParams sp{};
    return Sequence(ids, sp);
}

static constexpr int BS = Sequence::BLOCK_SIZE; // 256

// ============================================================
// compute_hash 基本测试
// ============================================================
TEST(BlockManagerTest, ComputeHashDeterministic) {
    std::vector<int32_t> ids(BS);
    std::iota(ids.begin(), ids.end(), 100);

    uint64_t h1 = BlockManager::compute_hash(ids);
    uint64_t h2 = BlockManager::compute_hash(ids);
    EXPECT_EQ(h1, h2);
    EXPECT_NE(h1, kNoHash);
}

TEST(BlockManagerTest, ComputeHashDifferentTokensDiffer) {
    std::vector<int32_t> a(BS, 0);
    std::vector<int32_t> b(BS, 1);
    EXPECT_NE(BlockManager::compute_hash(a), BlockManager::compute_hash(b));
}

TEST(BlockManagerTest, ComputeHashChainDiffers) {
    std::vector<int32_t> ids(BS, 42);
    uint64_t h_no_prefix = BlockManager::compute_hash(ids);
    uint64_t h_with_prefix = BlockManager::compute_hash(ids, 12345ULL);
    EXPECT_NE(h_no_prefix, h_with_prefix);
}

// ============================================================
// 分配 / 释放基本流程
// ============================================================
TEST(BlockManagerTest, AllocateThenDeallocate) {
    const int num_blocks = 16;
    BlockManager bm(num_blocks, BS);
    EXPECT_EQ(bm.num_free_blocks(), num_blocks);
    EXPECT_EQ(bm.num_used_blocks(), 0);

    // 1 个 block 的序列（256 tokens）
    auto seq = make_seq(BS);
    EXPECT_TRUE(bm.can_allocate(seq));
    bm.allocate(seq);

    EXPECT_EQ(seq.block_table.size(), 1u);
    EXPECT_EQ(bm.num_free_blocks(), num_blocks - 1);
    EXPECT_EQ(bm.num_used_blocks(), 1);

    bm.deallocate(seq);
    EXPECT_EQ(bm.num_free_blocks(), num_blocks);
    EXPECT_EQ(bm.num_used_blocks(), 0);
    EXPECT_TRUE(seq.block_table.empty());
    EXPECT_EQ(seq.num_cached_tokens, 0);
}

TEST(BlockManagerTest, AllocateMultipleBlocks) {
    const int num_blocks = 16;
    BlockManager bm(num_blocks, BS);

    // 3 个完整 block + 1 个不完整 block = 4 blocks
    auto seq = make_seq(3 * BS + 100);
    EXPECT_EQ(seq.num_blocks(), 4);
    EXPECT_TRUE(bm.can_allocate(seq));
    bm.allocate(seq);

    EXPECT_EQ(static_cast<int>(seq.block_table.size()), 4);
    EXPECT_EQ(bm.num_free_blocks(), num_blocks - 4);
    EXPECT_EQ(bm.num_used_blocks(), 4);

    bm.deallocate(seq);
    EXPECT_EQ(bm.num_free_blocks(), num_blocks);
}

TEST(BlockManagerTest, CannotAllocateWhenFull) {
    BlockManager bm(2, BS);
    auto seq = make_seq(3 * BS); // 需要 3 blocks
    EXPECT_FALSE(bm.can_allocate(seq));
}

// ============================================================
// N 个 block 分配再释放，free 数量正确
// ============================================================
TEST(BlockManagerTest, AllocateNBlocksThenFreeAll) {
    const int N = 10;
    BlockManager bm(N, BS);

    std::vector<Sequence> seqs;
    for (int i = 0; i < N; ++i) {
        // 每个序列用不同的 token 内容，避免 prefix cache hit
        std::vector<int32_t> ids(BS);
        std::iota(ids.begin(), ids.end(), i * BS);
        seqs.push_back(make_seq_from(ids));
    }

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(bm.num_free_blocks(), N - i);
        bm.allocate(seqs[i]);
    }
    EXPECT_EQ(bm.num_free_blocks(), 0);
    EXPECT_EQ(bm.num_used_blocks(), N);

    for (int i = 0; i < N; ++i) {
        bm.deallocate(seqs[i]);
    }
    EXPECT_EQ(bm.num_free_blocks(), N);
    EXPECT_EQ(bm.num_used_blocks(), 0);
}

// ============================================================
// Prefix Cache：两个相同 prefix 的序列
// ============================================================
TEST(BlockManagerTest, PrefixCacheHit) {
    const int num_blocks = 16;
    BlockManager bm(num_blocks, BS);

    // 两个序列共享相同的前 2 个 block (512 tokens)
    // 第一个序列：512 + 100 = 612 tokens (3 blocks)
    std::vector<int32_t> shared_prefix(2 * BS);
    std::iota(shared_prefix.begin(), shared_prefix.end(), 0);

    std::vector<int32_t> ids1 = shared_prefix;
    ids1.resize(2 * BS + 100, 999); // 后 100 个 token = 999
    auto seq1 = make_seq_from(ids1);

    // 第二个序列：512 + 50 = 562 tokens (3 blocks)
    std::vector<int32_t> ids2 = shared_prefix;
    ids2.resize(2 * BS + 50, 888); // 后 50 个 token = 888
    auto seq2 = make_seq_from(ids2);

    // seq1 首先分配——无 cache
    bm.allocate(seq1);
    EXPECT_EQ(seq1.num_cached_tokens, 0);
    EXPECT_EQ(static_cast<int>(seq1.block_table.size()), 3);

    // seq2 分配——前 2 个完整 block 应该 cache hit
    bm.allocate(seq2);
    EXPECT_EQ(seq2.num_cached_tokens, 2 * BS); // 命中 512 tokens
    // 命中的 2 blocks 复用 seq1 的，第 3 block 新分配
    EXPECT_EQ(static_cast<int>(seq2.block_table.size()), 3);

    // 前 2 个 block 应该与 seq1 共享（相同 block_id）
    EXPECT_EQ(seq2.block_table[0], seq1.block_table[0]);
    EXPECT_EQ(seq2.block_table[1], seq1.block_table[1]);
    // 第 3 个 block 不同（内容不同）
    EXPECT_NE(seq2.block_table[2], seq1.block_table[2]);
}

TEST(BlockManagerTest, PrefixCacheNoCacheOnDifferentTokens) {
    const int num_blocks = 16;
    BlockManager bm(num_blocks, BS);

    auto seq1 = make_seq(2 * BS);  // tokens 0..511
    bm.allocate(seq1);
    EXPECT_EQ(seq1.num_cached_tokens, 0);

    // 完全不同的 token
    std::vector<int32_t> ids2(2 * BS, 99999);
    auto seq2 = make_seq_from(ids2);
    bm.allocate(seq2);
    EXPECT_EQ(seq2.num_cached_tokens, 0); // 没有 cache hit
}

// ============================================================
// Deallocate 后 block 被回收（ref_count 降为 0）
// ============================================================
TEST(BlockManagerTest, DeallocateRecyclesBlocks) {
    const int num_blocks = 4;
    BlockManager bm(num_blocks, BS);

    auto seq1 = make_seq(2 * BS); // 2 blocks
    auto seq2 = make_seq(2 * BS); // 2 blocks (same content → cache hit)

    bm.allocate(seq1);
    EXPECT_EQ(bm.num_free_blocks(), 2);

    bm.allocate(seq2);
    // seq2 cache hit，共享 seq1 的 block → 不需要新 block
    EXPECT_EQ(seq2.num_cached_tokens, 2 * BS);
    EXPECT_EQ(bm.num_free_blocks(), 2); // 没有额外分配

    // 释放 seq1，但 block 仍被 seq2 引用
    bm.deallocate(seq1);
    EXPECT_EQ(bm.num_free_blocks(), 2); // ref_count 从 2 降到 1，不回收
    EXPECT_EQ(bm.num_used_blocks(), 2);

    // 释放 seq2，block 彻底回收
    bm.deallocate(seq2);
    EXPECT_EQ(bm.num_free_blocks(), 4);
    EXPECT_EQ(bm.num_used_blocks(), 0);
}

// ============================================================
// can_append / may_append
// ============================================================
TEST(BlockManagerTest, CanAppendWhenLastBlockNotFull) {
    BlockManager bm(4, BS);

    auto seq = make_seq(BS + 10); // 2 blocks，第 2 个 block 只有 10 tokens
    bm.allocate(seq);

    // 已使用 10/256 slots → may_append 不需要新 block
    EXPECT_TRUE(bm.can_append(seq));

    // 追加 1 个 token，len == BS+11，11 % 256 == 11 → 无需新 block
    seq.append_token(42);
    EXPECT_TRUE(bm.can_append(seq));
    bm.may_append(seq); // 什么都不做（未满）
}

TEST(BlockManagerTest, MayAppendAllocatesNewBlock) {
    BlockManager bm(4, BS);

    // 恰好 1 个完整 block
    auto seq = make_seq(BS);
    bm.allocate(seq);
    EXPECT_EQ(bm.num_free_blocks(), 3);

    // 追加 1 个 token → len = BS+1, (BS+1) % BS == 1 → 需要新 block
    seq.append_token(42);
    EXPECT_EQ(seq.len(), BS + 1);
    EXPECT_TRUE(bm.can_append(seq));
    bm.may_append(seq);

    EXPECT_EQ(static_cast<int>(seq.block_table.size()), 2);
    EXPECT_EQ(bm.num_free_blocks(), 2);
}

TEST(BlockManagerTest, MayAppendRegistersHashWhenBlockFull) {
    BlockManager bm(4, BS);

    // 初始 2*BS - 1 tokens → 2 blocks，第 2 个缺 1 个 token
    auto seq = make_seq(2 * BS - 1);
    bm.allocate(seq);
    EXPECT_EQ(static_cast<int>(seq.block_table.size()), 2);
    EXPECT_EQ(bm.num_free_blocks(), 2);

    // 追加 1 个 token → len = 2*BS, 即第 2 个 block 刚好填满
    seq.append_token(42);
    EXPECT_EQ(seq.len() % BS, 0);

    bm.may_append(seq);

    // 没有分配新 block
    EXPECT_EQ(static_cast<int>(seq.block_table.size()), 2);
    EXPECT_EQ(bm.num_free_blocks(), 2);

    // 验证现在另一个相同前缀的序列可以命中 cache
    // 构造和 seq 前 2*BS 个 token 完全相同的新序列
    std::vector<int32_t> ids2 = seq.token_ids;
    ids2.resize(2 * BS + 50, 777);
    auto seq2 = make_seq_from(ids2);
    bm.allocate(seq2);
    EXPECT_EQ(seq2.num_cached_tokens, 2 * BS);
}

TEST(BlockManagerTest, CanAppendFailsWhenNoFreeBlocks) {
    BlockManager bm(2, BS);

    auto seq1 = make_seq(BS);                 // tokens 0..255
    auto seq2 = make_seq_from(std::vector<int32_t>(BS, 99999)); // 不同内容
    bm.allocate(seq1);
    bm.allocate(seq2);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    // seq1: len = BS, 追加 1 token → rem=1 需要新 block，但没有了
    seq1.append_token(42);
    EXPECT_FALSE(bm.can_append(seq1));
}

// ============================================================
// 边界：单 token 序列
// ============================================================
TEST(BlockManagerTest, SingleTokenSequence) {
    BlockManager bm(4, BS);
    auto seq = make_seq(1);
    EXPECT_EQ(seq.num_blocks(), 1);

    bm.allocate(seq);
    EXPECT_EQ(static_cast<int>(seq.block_table.size()), 1);
    EXPECT_EQ(seq.num_cached_tokens, 0); // 不完整 block 无 hash → 无 cache
    EXPECT_EQ(bm.num_free_blocks(), 3);

    bm.deallocate(seq);
    EXPECT_EQ(bm.num_free_blocks(), 4);
}

// ============================================================
// Prefix Cache：cache miss 后续 block 不跳过
// ============================================================
TEST(BlockManagerTest, CacheMissBreaksChain) {
    const int num_blocks = 16;
    BlockManager bm(num_blocks, BS);

    // seq1: 3 完整 blocks
    auto seq1 = make_seq(3 * BS);
    bm.allocate(seq1);

    // seq2: block[0] 相同，block[1] 不同，block[2] 相同
    std::vector<int32_t> ids2(3 * BS);
    // block 0: 与 seq1 相同
    std::iota(ids2.begin(), ids2.begin() + BS, 0);
    // block 1: 不同
    std::fill(ids2.begin() + BS, ids2.begin() + 2 * BS, 99999);
    // block 2: 内容碰巧与 seq1 的 block[2] 相同（但 prefix 链已断）
    std::iota(ids2.begin() + 2 * BS, ids2.end(), 2 * BS);
    auto seq2 = make_seq_from(ids2);

    bm.allocate(seq2);
    // 只有 block[0] cache hit，block[1] miss 后 block[2] 也不跳过
    EXPECT_EQ(seq2.num_cached_tokens, BS); // 只命中 1 block
}

// ============================================================
// 多次分配释放循环
// ============================================================
TEST(BlockManagerTest, MultipleAllocateDeallocateCycles) {
    BlockManager bm(4, BS);

    for (int round = 0; round < 5; ++round) {
        auto seq = make_seq(2 * BS);
        EXPECT_TRUE(bm.can_allocate(seq));
        bm.allocate(seq);
        EXPECT_EQ(bm.num_free_blocks(), 2);

        bm.deallocate(seq);
        EXPECT_EQ(bm.num_free_blocks(), 4);
    }
}
