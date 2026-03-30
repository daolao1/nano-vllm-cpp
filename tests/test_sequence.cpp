#include "engine/sequence.h"

#include <gtest/gtest.h>

#include <vector>

using namespace nano_vllm;

TEST(SequenceTest, InitializesFromPromptAndSamplingParams) {
    SamplingParams params{};
    params.temperature = 0.7f;
    params.max_tokens = 128;
    params.ignore_eos = true;

    Sequence seq({1, 2, 3}, params);

    EXPECT_EQ(seq.status, SequenceStatus::WAITING);
    EXPECT_EQ(seq.token_ids, std::vector<int32_t>({1, 2, 3}));
    EXPECT_EQ(seq.last_token, 3);
    EXPECT_EQ(seq.num_tokens, 3);
    EXPECT_EQ(seq.num_prompt_tokens, 3);
    EXPECT_EQ(seq.num_cached_tokens, 0);
    EXPECT_FLOAT_EQ(seq.temperature, 0.7f);
    EXPECT_EQ(seq.max_tokens, 128);
    EXPECT_TRUE(seq.ignore_eos);
}

TEST(SequenceTest, SequenceIdsAreUnique) {
    SamplingParams params{};

    Sequence first({1}, params);
    Sequence second({2}, params);

    EXPECT_NE(first.seq_id, second.seq_id);
    EXPECT_LT(first.seq_id, second.seq_id);
}

TEST(SequenceTest, BlockHelpersReflectSequenceLength) {
    SamplingParams params{};
    std::vector<int32_t> tokens(300);
    for (int i = 0; i < 300; ++i) {
        tokens[i] = i;
    }

    Sequence seq(tokens, params);
    seq.num_cached_tokens = Sequence::BLOCK_SIZE;

    EXPECT_EQ(seq.len(), 300);
    EXPECT_EQ(seq.num_blocks(), 2);
    EXPECT_EQ(seq.num_cached_blocks(), 1);
    EXPECT_EQ(seq.last_block_num_tokens(), 44);

    std::vector<int32_t> first_block = seq.block(0);
    std::vector<int32_t> second_block = seq.block(1);

    EXPECT_EQ(first_block.size(), static_cast<size_t>(Sequence::BLOCK_SIZE));
    EXPECT_EQ(first_block.front(), 0);
    EXPECT_EQ(first_block.back(), 255);
    EXPECT_EQ(second_block.size(), 44u);
    EXPECT_EQ(second_block.front(), 256);
    EXPECT_EQ(second_block.back(), 299);
}

TEST(SequenceTest, BlockRejectsInvalidIndex) {
    SamplingParams params{};
    Sequence seq({1, 2, 3}, params);

    EXPECT_THROW(seq.block(-1), std::out_of_range);
    EXPECT_THROW(seq.block(1), std::out_of_range);
}

TEST(SequenceTest, AppendTokenUpdatesState) {
    SamplingParams params{};
    Sequence seq({10, 11}, params);

    seq.append_token(12);
    seq.append_token(13);

    EXPECT_EQ(seq.last_token, 13);
    EXPECT_EQ(seq.num_tokens, 4);
    EXPECT_EQ(seq.num_prompt_tokens, 2);
    EXPECT_EQ(seq.num_completion_tokens(), 2);
    EXPECT_EQ(seq.token_ids, std::vector<int32_t>({10, 11, 12, 13}));
}

TEST(SequenceTest, EmptyPromptIsHandledSafely) {
    SamplingParams params{};
    Sequence seq({}, params);

    EXPECT_EQ(seq.len(), 0);
    EXPECT_EQ(seq.last_token, -1);
    EXPECT_EQ(seq.num_blocks(), 0);
    EXPECT_EQ(seq.last_block_num_tokens(), 0);

    seq.append_token(42);

    EXPECT_EQ(seq.last_token, 42);
    EXPECT_EQ(seq.len(), 1);
    EXPECT_EQ(seq.num_prompt_tokens, 0);
    EXPECT_EQ(seq.num_completion_tokens(), 1);
}