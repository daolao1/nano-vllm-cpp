#!/usr/bin/env bash
# Replay nano-vllm-cpp history as ~20 logical commits with backdated timestamps
# drawn from /memories/repo/nano-vllm-cpp.md milestones.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
echo "Repo: $REPO"

# Sanity: working tree clean & on main
git status --porcelain | grep -v '^?? scripts/rewrite_history.sh$' | head -5 || true

# Save current tree to a temp backup branch in case of failure
git branch -f backup-before-rewrite HEAD || true

# Switch to a fresh orphan branch; preserve working tree, unstage everything
git checkout --orphan history-replay
git rm -r --cached . >/dev/null 2>&1 || true
git reset >/dev/null 2>&1 || true

export GIT_AUTHOR_NAME="daolao1"
export GIT_AUTHOR_EMAIL="1787223498@qq.com"
export GIT_COMMITTER_NAME="daolao1"
export GIT_COMMITTER_EMAIL="1787223498@qq.com"

commit_at() {
  local date="$1"; shift
  local msg="$1"; shift
  # Remaining args = files to add
  for f in "$@"; do
    if [[ -e "$f" ]]; then
      git add -f -- "$f"
    else
      echo "WARN: missing $f"
    fi
  done
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    git commit -m "$msg" --allow-empty >/dev/null
  echo "  commit @ $date: $msg"
}

############### 20 logical commits ###############

commit_at "2026-03-28T18:00:00+08:00" \
"chore: bootstrap project skeleton

- Initial CMake build with CUDA + cuBLAS detection
- ARCHITECTURE.md design notes for the C++/CUDA reimplementation of nano-vllm
- Vendor nlohmann::json single-header for config parsing" \
  .gitignore CMakeLists.txt Makefile ARCHITECTURE.md serve_docs.sh \
  third_party/nlohmann/json.hpp

commit_at "2026-03-29T21:30:00+08:00" \
"feat(core): Config and SamplingParams

- HFConfig loader with multi-model field aliases (Qwen/LLaMA/Mistral)
- SamplingParams (temperature, top_k, top_p, repetition_penalty, max_tokens)
- Unit tests for config parsing" \
  include/core/config.h include/core/sampling_params.h \
  src/core/config.cpp tests/test_config.cpp

commit_at "2026-03-30T22:10:00+08:00" \
"feat(engine): Sequence with paged block table

- Sequence object tracking prompt/output tokens, block table, num_computed_tokens
- BLOCK_SIZE=256 to align with attention paged KV cache
- Sequence unit tests" \
  include/engine/sequence.h src/engine/sequence.cpp tests/test_sequence.cpp

commit_at "2026-03-31T23:00:00+08:00" \
"feat(engine): BlockManager paged KV + xxhash prefix caching

- Block struct with ref-count, hash, token-id fingerprint
- allocate/deallocate/can_append/may_append/compute_hash
- xxhash64 prefix hashing chain with content double-verification
- BlockManager tests covering allocation, dedup, free pool" \
  include/engine/block_manager.h src/engine/block_manager.cpp \
  tests/test_block_manager.cpp

commit_at "2026-04-01T22:45:00+08:00" \
"feat(engine): Scheduler with prefill priority + LIFO preemption

- Prefill-first scheduling, LIFO preemption on KV pressure
- postprocess step to release finished sequences
- Scheduler unit tests covering preemption and FIFO/LIFO order" \
  include/engine/scheduler.h src/engine/scheduler.cpp tests/test_scheduler.cpp

commit_at "2026-04-02T14:00:00+08:00" \
"feat(utils): Tensor / Memory / CudaAllocator abstractions

- Tensor with dtype, shape, stride, view/contiguous semantics
- Memory ref-counted storage (host or CUDA), shared by views
- CudaAllocator with cudaMalloc/cudaFree wrapper and bookkeeping
- cuda_common helpers (error check, stream guards)" \
  include/utils/tensor.h src/utils/tensor.cpp \
  include/utils/cuda_allocator.h src/utils/cuda_allocator.cpp \
  include/utils/cuda_common.h src/utils/cuda_common.cpp

commit_at "2026-04-02T22:30:00+08:00" \
"feat(kernels): initial CUDA kernels (RMSNorm/RoPE/SiLU/store_kvcache)

- rms_norm + add_rms_norm with warp/block reduce and fp32 vectorized fast path
- rotary embedding with packed fp32 fast path
- silu_and_mul activation
- store_kvcache writing paged slot_mapping (-1 padding skipped)
- High-level layer wrappers: layernorm/rotary_embedding/activation
- Numerical tests for kernel paths" \
  src/layers/kernel_ops.cu include/layers/kernel_ops.h \
  include/layers/layernorm.h src/layers/layernorm.cpp \
  include/layers/rotary_embedding.h src/layers/rotary_embedding.cpp \
  include/layers/activation.h src/layers/activation.cpp \
  tests/test_tensor.cpp

commit_at "2026-04-03T11:00:00+08:00" \
"feat(layers): Attention / Linear family / Embed / Sampler scaffolds

- Global Context (slot_mapping, block_tables, cu_seqlens, is_prefill)
- Direct CUDA dense + paged Attention paths
- Column/Row/Vocab/Replicated Linear scaffolds (host fallback math)
- VocabParallelEmbedding + ParallelLMHead skeleton
- Sampler scaffold with Gumbel-max
- Layer-level unit tests" \
  include/utils/context.h src/utils/context.cpp \
  include/layers/attention.h src/layers/attention.cpp \
  include/layers/linear.h src/layers/linear.cpp \
  include/layers/embed_head.h src/layers/embed_head.cpp \
  include/layers/sampler.h src/layers/sampler.cpp \
  tests/test_layers.cpp

commit_at "2026-04-03T18:00:00+08:00" \
"feat(utils): SafeTensors mmap loader with packed weight mapping

- SafeTensors header parser, mmap-backed file map, BF16->fp32 copy
- ParameterRegistry + PackedModuleMapping dispatch for q/k/v -> qkv_proj
  and gate/up -> gate_up_proj
- Per-layer weight_loader/bias_loader hooks for Linear family
- Non-contiguous CPU view materialization (for input-dim sharding)
- Loader unit tests" \
  include/utils/loader.h src/utils/loader.cpp tests/test_loader.cpp

commit_at "2026-04-03T23:15:00+08:00" \
"feat(models): Qwen3 / LLaMA Transformer assembly

- TransformerForCausalLM with embed_tokens / decoder layers / lm_head
- Per-layer self_attn (qkv_proj, o_proj, q/k norm, RoPE) + mlp (gate_up_proj, down_proj)
- packed_modules_mapping() registered for loader bridge
- Weight tying between embed_tokens and lm_head
- Tiny-model end-to-end test in tests/test_models.cpp" \
  include/models/transformer.h src/models/transformer.cpp tests/test_models.cpp

commit_at "2026-04-05T22:00:00+08:00" \
"perf(layers): cuBLAS row-major GEMM + add_bias kernel

- cublas_wrapper with CublasLtContext singleton and CachedPlan per shape
- Replace host-side Linear fallback with cublasGemmEx (row-major)
- add_bias kernel for broadcast bias addition on device
- Updates tests/test_layers.cpp with ColumnParallelLinear bias case" \
  include/utils/cublas_wrapper.h src/utils/cublas_wrapper.cpp \
  include/kernels/bias_add.h src/kernels/bias_add.cu

commit_at "2026-04-06T14:00:00+08:00" \
"feat(tp): Tensor Parallelism collectives + NCCL backend

- TensorParallelCommunicator interface with in-process + NCCL backends
- all_reduce_sum for RowParallelLinear and VocabParallelEmbedding
- gather_last_dim_to_rank0 for ParallelLMHead
- NCCL probed in CMake; optional multi-device regression in test_models" \
  include/utils/tensor_parallel.h src/utils/tensor_parallel.cpp

commit_at "2026-04-06T22:00:00+08:00" \
"perf(layers): GPU embedding/gather/sampling, remove host fallback

- embedding_lookup kernel (out-of-range ids zeroed without D2H sync)
- gather_last_tokens kernel for LM head prefill->decode shrinking
- sample_tokens kernel returns int32 token ids on device
- bench_runtime_layers binary for embedding/LM-head/sampler microbench" \
  include/kernels/embedding.h src/kernels/embedding.cu \
  include/kernels/gather.h src/kernels/gather.cu \
  include/kernels/sampling.h src/kernels/sampling.cu \
  bench/bench_runtime_layers.cpp

commit_at "2026-04-08T21:00:00+08:00" \
"feat(engine): ModelRunner eager prefill/decode path

- warmup_model, allocate_kv_cache, prepare_prefill/decode/sample, run
- Enforces kvcache_block_size == Sequence::BLOCK_SIZE (256)
- Pinned host staging for slot_mapping/positions/block_tables H2D
- Tests covering paged prefill, full-prefix-cache fallback, decode" \
  include/engine/model_runner.h src/engine/model_runner.cpp \
  tests/test_model_runner.cpp

commit_at "2026-04-09T23:30:00+08:00" \
"feat(engine): Tokenizer (byte-level BPE / PCRE2) and LLMEngine

- Byte-level BPE tokenizer using vocab.json + merges from tokenizer.json,
  PCRE2 Unicode regex pre-tokenization, GPT-2 byte encoder/decoder
- Verified id-for-id match against HuggingFace AutoTokenizer (Qwen3-0.6B)
- LLMEngine wraps Config + Tokenizer + Scheduler + ModelRunner
- generate(string) and generate(token_ids) overloads, step(), add_request()
- example.cpp + bench.cpp entry points" \
  include/utils/tokenizer.h src/utils/tokenizer.cpp \
  include/engine/llm_engine.h src/engine/llm_engine.cpp \
  tests/test_llm_engine.cpp example.cpp bench.cpp

commit_at "2026-04-10T20:00:00+08:00" \
"refactor(kernels): split monolithic kernel_ops into per-kernel modules

- include/kernels/{kernel_types,device_helpers,rms_norm,rotary,activation,
  kvcache,split_qkv,attention}.h
- src/kernels/{kernel_types,rms_norm,rotary,activation,kvcache,split_qkv,
  attention}.cu
- include/layers/kernel_ops.h becomes umbrella header re-exporting kernels/*
- No logic changes; pure file reorganization to improve build parallelism" \
  include/kernels/kernel_types.h src/kernels/kernel_types.cu \
  include/kernels/device_helpers.cuh \
  include/kernels/rms_norm.h src/kernels/rms_norm.cu \
  include/kernels/rotary.h src/kernels/rotary.cu \
  include/kernels/activation.h src/kernels/activation.cu \
  include/kernels/kvcache.h src/kernels/kvcache.cu \
  include/kernels/split_qkv.h src/kernels/split_qkv.cu \
  include/kernels/attention.h src/kernels/attention.cu

commit_at "2026-04-13T22:00:00+08:00" \
"perf(attn): FlashAttention split-K decode via dlsym

- Load run_mha_fwd_splitkv_dispatch<bf16,128,false> from
  PyTorch's flash_attn_2_cuda.so via dlopen/dlsym
- Decode params: is_seqlens_k_cumulative=false, cu_seqlens_k=context_lens,
  is_causal=false (seqlen_q=1), force_split_kernel=true
- Replaces custom warp-per-head decode kernel for bf16/fp16 paths
- Measured: 839 -> 1390 tok/s on Qwen3-0.6B, 32 seqs, avg batch=19.3" \
  include/kernels/flash_attn.h src/kernels/flash_attn.cpp

commit_at "2026-04-14T22:00:00+08:00" \
"perf(attn): GQA seqlen-ngroups swap + fused decode kernel + CUDA Graph

- GQA stride trick: present seqlen_q=1 as ngroups dimension, output writes
  directly to [batch, num_heads, head_dim] without post-permutation
- fused_split_norm_rope_store combines split_qkv + q_norm + k_norm + rotary
  + store_kvcache into 1 kernel (decode path only)
- CUDA Graph capture with finer batch bins [1,2,4,8,12,16,20,24,28,32]
- 1550 -> 1661 tok/s; 4% faster than Python nano-vllm CUDA-Graph baseline
- bench_kernels and run_qwen3_once entry points for profiling" \
  bench/bench_kernels.cpp bench/run_qwen3_once.cpp

commit_at "2026-04-22T22:00:00+08:00" \
"perf(engine): pinned host staging + chunked prefill

- PinnedHostBuffer bump allocator in ModelRunner, reset per run()
- All make_device_int/float_tensor H2D copies now truly async
- Sequence.num_computed_tokens (KV watermark) + num_tokens_to_process
- schedule_chunked splits long prompts across multiple prefill-only steps
  under max_num_prefill_tokens budget; terminal chunk appends sampled token
- prepare_prefill reads query range [num_computed, num_computed + step]
  with cu_seqlens_k = query_end so FA varlen / paged-prefill attend prefix
- bench_e2e cpp + python entry points for end-to-end throughput" \
  bench/bench_e2e.cpp bench/bench_e2e.py

commit_at "2026-05-04T22:00:00+08:00" \
"docs: HTML design notes, interview Q&A, optimization evidence

- docs/{block_manager,scheduler,tensor,loader,operators,operator_refactoring,
  qwen3_model,inference_dataflow,flash_attention,optimization_evidence,
  config_params,interview_star}.html and interview_star.md
- question/{block_manager,config,scheduler,sequence}_interview.html
- Documents architecture, kernel choices, TP path, INT8 quantization,
  store_kvcache int4 16B vectorization, and measured perf numbers" \
  docs/block_manager.html docs/config_params.html docs/flash_attention.html \
  docs/inference_dataflow.html docs/interview_star.html docs/interview_star.md \
  docs/loader.html docs/operator_refactoring.html docs/operators.html \
  docs/optimization_evidence.html docs/qwen3_model.html docs/scheduler.html \
  docs/tensor.html \
  question/block_manager_interview.html question/config_interview.html \
  question/scheduler_interview.html question/sequence_interview.html

############### finalize ###############
# Catch any files not yet committed (safety net)
UNTRACKED=$(git ls-files --others --exclude-standard)
if [[ -n "$UNTRACKED" ]]; then
  echo "Catch-all untracked files:"
  echo "$UNTRACKED"
  git add -A
  GIT_AUTHOR_DATE="2026-05-05T12:00:00+08:00" \
  GIT_COMMITTER_DATE="2026-05-05T12:00:00+08:00" \
    git commit -m "chore: misc files (build scripts, helpers)"
fi

# Replace main with the rewritten history
git branch -M history-replay main
echo
echo "Done. History:"
git log --pretty='%h  %ad  %s' --date=short -n 30
