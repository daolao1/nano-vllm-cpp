#include "engine/scheduler.h"

#include <gtest/gtest.h>

#include <numeric>
#include <utility>
#include <vector>

using namespace nano_vllm;

static Config make_config(int num_blocks,
                          int max_num_seqs = 8,
                          int max_num_batched_tokens = 4096,
                          int eos_token_id = 999999) {
    Config config{};
    config.max_num_seqs = max_num_seqs;
    config.max_num_batched_tokens = max_num_batched_tokens;
    config.kvcache_block_size = Sequence::BLOCK_SIZE;
    config.num_kvcache_blocks = num_blocks;
    config.eos_token_id = eos_token_id;
    return config;
}

static Sequence make_seq(int n_tokens,
                         int start_token = 0,
                         int max_tokens = 64) {
    std::vector<int32_t> ids(n_tokens);
    std::iota(ids.begin(), ids.end(), start_token);
    SamplingParams sp{};
    sp.max_tokens = max_tokens;
    return Sequence(std::move(ids), sp);
}

static constexpr int BS = Sequence::BLOCK_SIZE;

TEST(SchedulerTest, PrefillIsPrioritizedOverDecode) {
    Scheduler scheduler(make_config(8, 4));

    scheduler.add(make_seq(10, 0));
    scheduler.add(make_seq(12, 100));

    auto [first_batch, first_is_prefill] = scheduler.schedule();
    ASSERT_TRUE(first_is_prefill);
    ASSERT_EQ(first_batch.size(), 2u);

    scheduler.postprocess(first_batch, {1000, 1001});

    Sequence third = make_seq(6, 200);
    const int64_t third_id = third.seq_id;
    scheduler.add(std::move(third));

    auto [second_batch, second_is_prefill] = scheduler.schedule();
    ASSERT_TRUE(second_is_prefill);
    ASSERT_EQ(second_batch.size(), 1u);
    EXPECT_EQ(second_batch[0]->seq_id, third_id);
    EXPECT_EQ(second_batch[0]->status, SequenceStatus::RUNNING);
}

TEST(SchedulerTest, DecodePreemptsLatestRunningSequenceWhenOutOfBlocks) {
    const int eos = 123456;
    Scheduler scheduler(make_config(4, 3, 4096, eos));

    scheduler.add(make_seq(BS, 0));
    scheduler.add(make_seq(BS, 1000));
    scheduler.add(make_seq(BS, 2000));

    auto [prefill_batch, is_prefill] = scheduler.schedule();
    ASSERT_TRUE(is_prefill);
    ASSERT_EQ(prefill_batch.size(), 3u);

    const int64_t first_id = prefill_batch[0]->seq_id;
    const int64_t second_id = prefill_batch[1]->seq_id;
    const int64_t third_id = prefill_batch[2]->seq_id;
    Sequence* third_seq = prefill_batch[2];

    scheduler.postprocess(prefill_batch, {3000, 4000, 5000});

    auto [decode_batch, decode_is_prefill] = scheduler.schedule();
    ASSERT_FALSE(decode_is_prefill);
    ASSERT_EQ(decode_batch.size(), 2u);
    EXPECT_EQ(decode_batch[0]->seq_id, first_id);
    EXPECT_EQ(decode_batch[1]->seq_id, second_id);
    EXPECT_EQ(third_seq->seq_id, third_id);
    EXPECT_EQ(third_seq->status, SequenceStatus::WAITING);
    EXPECT_TRUE(third_seq->block_table.empty());

    scheduler.postprocess(decode_batch, {eos, eos});

    auto [rescheduled_batch, rescheduled_is_prefill] = scheduler.schedule();
    ASSERT_TRUE(rescheduled_is_prefill);
    ASSERT_EQ(rescheduled_batch.size(), 1u);
    EXPECT_EQ(rescheduled_batch[0]->seq_id, third_id);
}

TEST(SchedulerTest, PostprocessMarksFinishedAndRecyclesBlocks) {
    const int eos = 654321;
    Scheduler scheduler(make_config(2, 1, 4096, eos));

    scheduler.add(make_seq(BS, 0));

    auto [scheduled, is_prefill] = scheduler.schedule();
    ASSERT_TRUE(is_prefill);
    ASSERT_EQ(scheduled.size(), 1u);

    Sequence* finished_seq = scheduled[0];
    scheduler.postprocess(scheduled, {eos});

    EXPECT_EQ(finished_seq->status, SequenceStatus::FINISHED);
    EXPECT_TRUE(finished_seq->block_table.empty());
    EXPECT_TRUE(scheduler.is_finished());

    Sequence next = make_seq(BS * 2, 1000);
    const int64_t next_id = next.seq_id;
    scheduler.add(std::move(next));

    auto [next_batch, next_is_prefill] = scheduler.schedule();
    ASSERT_TRUE(next_is_prefill);
    ASSERT_EQ(next_batch.size(), 1u);
    EXPECT_EQ(next_batch[0]->seq_id, next_id);
}

// ---------------------------------------------------------------------------
// Chunked prefill: long prompts split across multiple prefill steps, then
// transition to decode only after all prompt tokens have been computed.
// ---------------------------------------------------------------------------

TEST(SchedulerTest, ChunkedPrefillSplitsLongPromptAcrossSteps) {
    Config config = make_config(/*num_blocks=*/32, /*max_num_seqs=*/4);
    config.enable_chunked_prefill = true;
    config.max_num_prefill_tokens = 128;  // force multiple chunks for a 300-token prompt
    Scheduler scheduler(config);

    const int prompt_len = 300;
    scheduler.add(make_seq(prompt_len, /*start_token=*/0, /*max_tokens=*/4));
    // Feed a dummy sampled token id that is NOT eos so postprocess doesn't finish the seq.
    const int32_t non_eos = 42;

    // Step 1: first chunk of 128 tokens.
    auto [batch1, is_prefill1] = scheduler.schedule();
    ASSERT_TRUE(is_prefill1);
    ASSERT_EQ(batch1.size(), 1u);
    Sequence* seq = batch1[0];
    EXPECT_EQ(seq->num_tokens_to_process, 128);
    EXPECT_EQ(seq->num_computed_tokens, 0);
    scheduler.postprocess(batch1, {non_eos});
    EXPECT_EQ(seq->num_computed_tokens, 128);
    EXPECT_EQ(seq->num_tokens_to_process, 0);
    EXPECT_EQ(seq->num_tokens, prompt_len);  // mid-chunk: no token appended

    // Step 2: second chunk of 128 tokens.
    auto [batch2, is_prefill2] = scheduler.schedule();
    ASSERT_TRUE(is_prefill2);
    ASSERT_EQ(batch2.size(), 1u);
    EXPECT_EQ(batch2[0], seq);
    EXPECT_EQ(seq->num_tokens_to_process, 128);
    scheduler.postprocess(batch2, {non_eos});
    EXPECT_EQ(seq->num_computed_tokens, 256);
    EXPECT_EQ(seq->num_tokens, prompt_len);

    // Step 3: final prefill chunk of 44 tokens (terminal) -> sampled token IS appended.
    auto [batch3, is_prefill3] = scheduler.schedule();
    ASSERT_TRUE(is_prefill3);
    ASSERT_EQ(batch3.size(), 1u);
    EXPECT_EQ(seq->num_tokens_to_process, prompt_len - 256);
    scheduler.postprocess(batch3, {non_eos});
    EXPECT_EQ(seq->num_computed_tokens, prompt_len);
    EXPECT_EQ(seq->num_tokens, prompt_len + 1);          // terminal chunk appended sample
    EXPECT_EQ(seq->last_token, non_eos);

    // Step 4: next schedule must now be a decode step.
    auto [batch4, is_prefill4] = scheduler.schedule();
    EXPECT_FALSE(is_prefill4);
    ASSERT_EQ(batch4.size(), 1u);
    EXPECT_EQ(batch4[0]->num_tokens_to_process, 0);
}

TEST(SchedulerTest, ChunkedPrefillStaysInDefaultPathWhenDisabled) {
    // Sanity: with chunked prefill disabled, the default path must still
    // process the entire prompt in a single step and set num_tokens_to_process
    // = num_tokens so prepare_prefill's unified path works.
    Scheduler scheduler(make_config(/*num_blocks=*/8));
    const int prompt_len = 10;
    scheduler.add(make_seq(prompt_len));

    auto [batch, is_prefill] = scheduler.schedule();
    ASSERT_TRUE(is_prefill);
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0]->num_tokens_to_process, prompt_len);
    EXPECT_EQ(batch[0]->num_computed_tokens, 0);
}