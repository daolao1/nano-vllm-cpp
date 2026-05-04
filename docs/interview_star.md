# nano-vllm-cpp 面试 STAR 材料

> 面向 AI Infra / 推理引擎 / CUDA 岗位的项目面试准备。所有故事以 **S(情境) / T(任务) / A(行动) / R(结果)** 展开，配套"面试官可能追问"的 Q&A。直接用于口述，不要照读。

---

## 目录

- [项目一句话](#项目一句话)
- [1 分钟自我介绍](#1-分钟自我介绍可直接背)
- [3 分钟深入版介绍](#3-分钟深入版介绍)
- [性能路径演进（一张表说清所有优化）](#性能路径演进一张表说清所有优化)
- [项目数据流全景](#项目数据流全景)
- STAR 故事：
  - [① Decode 路径 5-kernel 融合](#star-故事一decode-路径-5-kernel-融合最能打的亮点)
  - [② Chunked Prefill + Pinned Staging](#star-故事二chunked-prefill--pinned-stagingserving-侧收尾)
  - [③ WSL libcuda.so 冲突](#star-故事三wsl-libcudaso-冲突体现-debug-能力)
  - [④ CUDA Graph + ScratchAllocator](#star-故事四cuda-graph--scratchallocator)
  - [⑤ INT8 KV Cache 端到端](#star-故事五int8-kv-cache-端到端量化-工程-tradeoff)
- [高频追问 Q&A（技术点深挖）](#高频追问-qa技术点深挖)
- [通用行为问题](#通用行为问题备用hr--技术-leader-问)
- [面试前 checklist + 反问问题](#面试前一天-checklist)

---

## 项目一句话

参考 nano-vllm (Python) 用 C++ / CUDA 从零重写的 LLM 推理引擎，~15K LoC，同机同模型同 bench 下 Qwen3-0.6B 吞吐 **1386 → 1615 tok/s (+16.5%)**，RTX 4060 Laptop。

---

## 1 分钟自我介绍（可直接背）

> 我做了一个 nano-vllm-cpp，参考 nano-vllm 这个 Python 教学项目，用 C++ 和 CUDA 从零重写了一遍推理引擎。同一台机器、同一个 bench 跑 Qwen3-0.6B，Python 版是 1386 tok/s，我的 C++ 版是 1615，提升大概 16%。这 16% 来自 decode 路径的 5-kernel 融合、pinned staging、以及 CUDA Graph 配合 host 侧开销的降低。过程中我手写了 14 个 CUDA kernel，集成了 Flash Attention，并实现了 Chunked Prefill、Prefix Cache、INT8 KV 这些 serving 侧优化——整个栈从 SafeTensors 加载到 kernel 到调度都是自己写的，对每一层的性能开销都有账。

---

## 3 分钟深入版介绍

> **项目定位**：参考 nano-vllm (Python) 做的 C++ / CUDA 重写，目标不是做一个"生产级 vLLM 替代"，而是把 LLM 推理栈——SafeTensors 加载、Tokenizer、Scheduler、BlockManager、PagedAttention、TP 通信、Sampler——每一层都亲手实现一遍，用来弄清楚每一毫秒时间的去向。
>
> **技术栈**：C++20、CUDA 12、cuBLAS、NCCL；外部只依赖 Flash Attention 2（via `dlopen` 调 PyTorch extension 的 C++ symbol）、xxhash、PCRE2、SafeTensors 的 mmap 解析是我自己写的。
>
> **关键结果**：
> - 功能：支持 Qwen3 / LLaMA-3 / TinyLlama，单卡 + TP + NCCL，CUDA Graph，INT8 KV Cache，Chunked Prefill，Prefix Cache，Top-k/Top-p + 重复惩罚全部设备端。
> - 性能（RTX 4060 Laptop, Qwen3-0.6B, 32 并发）：**1615 tok/s，decode 11 ms/step @ avg batch=19.3**，比同机 nano-vllm Python 的 1386 tok/s 高 16.5%。
> - 质量：10/10 gtest 全绿，含双卡 TP 数值回归、BPE tokenizer 与 HuggingFace 逐 ID 对齐。
>
> **我花时间最多的三块**：
> 1. **Decode 路径延迟优化**：profiling 发现小 batch decode 被 kernel launch 和 HBM 带宽 bound，做了 5-kernel 融合（split_qkv + qk_norm + RoPE + store_kvcache），加上 Flash Attention 集成和 GQA seqlen-ngroups swap，decode 从 22 ms 压到 11 ms。
> 2. **Serving 侧工程**：Chunked Prefill（Sequence 引入 `num_computed_tokens` 水位 + 切块调度）、Prefix Cache（xxhash 链式 block hash + 跨请求 ref_count 复用）、Pinned Staging（prepare_\* 的 host buffer bump 分配，H2D 真异步）。
> 3. **工具链 / 调试**：CUDA Graph 分档捕获 + scratch allocator、WSL 上 libcuda.so 冲突的链接器级排查、CMake + CUDA 架构探测时机问题。
>
> **下一步**：Weight-only INT4 (Marlin / AWQ) 应对 7B+ 模型的 Linear 瓶颈；Speculative Decoding；真正的 mixed prefill+decode 同 step 调度。

---

## 性能路径演进（一张表说清所有优化）

> 面试时想在白板上画出来的一张表。每一步都能回答"这一步为什么能提升？下一步为什么还能提升？"

| 优化阶段 | 技术 | Decode ms/step | 吞吐 (tok/s) | 收益来源 |
|---|---|---|---|---|
| A | 朴素 C++ 重写，自研 warp-per-head decode attention | ~22 ms | 839 | baseline，小 batch 下 attention 算力利用率低 |
| B | **Flash Attention 2 decode**（split-K via dlopen） | 13.1 ms | 1390 | attention 算力利用率 ↑，online softmax tile |
| C | **GQA seqlen_q↔ngroups swap** | 11.5 ms | 1550 | seqlen_q=1 时 FA warp 利用率低，用 stride 伪装 ngroups → warp 合并 |
| D | C + **CUDA Graph 分档捕获** | 11.1 ms | 1627 | 消除 decode 步的 kernel launch 序列开销 |
| E | **5-kernel 融合**（fused_split_norm_rope_store）+ 更细 BS 档位 | 11.2 ms (eager) | 1561 (eager) | 省 4 次 HBM 往返 + 4 次 launch，eager 也受益 |
| F | E + CUDA Graph | **10.9 ms** | **1661** | 融合和 Graph 叠加 |
| G | F + **Pinned Staging + Chunked Prefill 支持** | 11.2 ms | **1615** | 保持 decode 峰值，长 prompt 场景 TTFT 可控 |
| 对比 | nano-vllm Python（同机同 bench） | 18.6 ms | 1386 | — |

**上下文**：RTX 4060 Laptop (8 GB)，Qwen3-0.6B (bf16)，32 seqs，prompt 100-512 tok，output 100-512 tok，avg decode batch=19.3。数字来自 `bench_e2e`（repo 内）与 `bench.py`（nano-vllm）。

---

## 项目数据流全景

> 白板能画出来：`请求 → Tokenizer → Scheduler → BlockManager → ModelRunner → Model → Sampler → 输出`

```
   add_request(prompt_text)
       │
       ▼
   Tokenizer (BPE + PCRE2)
       │  token_ids
       ▼
   Scheduler.add(Sequence)
       │
       ▼
   ┌───────── step() ─────────┐
   │                           │
   │  schedule()               │  ← default / chunked
   │    ├─ BlockManager        │  ← allocate / prefix cache 查 xxhash
   │    └─ 选 prefill 或 decode│
   │                           │
   │  ModelRunner.run()        │
   │   │ reset_pinned_staging  │  ← bump offset = 0
   │   │ prepare_prefill/decode│  ← host vec → pinned → cudaMemcpyAsync
   │   │ (set Context)         │  ← cu_seqlens_q/k, slot_mapping, block_tables
   │   │                       │
   │   │ CUDA Graph replay ?   │
   │   │  ├─ yes: 固定 exec     │
   │   │  └─ no:  run_model    │
   │   │         forward:       │
   │   │           embed →      │
   │   │           N × decoder │  ← 每层：RMSNorm → QKV → Attn → O → +res
   │   │                       │                → RMSNorm → MLP → +res
   │   │           compute_logits (LM head)     
   │   │                       │
   │   │ Sampler (GPU)         │  ← 重复惩罚 → Top-k 二分 → Top-p 二分 → Gumbel
   │   │   → int32 token ids   │
   │                           │
   │  postprocess              │  ← append_token / EOS / 释放 block
   └───────────────────────────┘
```

核心 invariant：
- `Context` 是 thread-local 全局状态，由 `set_context` 在 prepare 阶段填好，各 layer 直接读；
- prepare 出的所有小 host 张量 (input_ids / positions / slot_mapping / cu_seqlens_q/k / context_lens / block_tables / 重复惩罚 token 表) 在 **run() 内** 分配，**run() 返回前**生命周期结束——与 pinned bump offset 周期对齐；
- `slot_mapping`、`block_tables`、`cu_seqlens_*` 这几个张量是 paged attention 与 KV cache 的"地址簿"，其他 kernel 不需要看。

---



## STAR 故事一：Decode 路径 5-kernel 融合（最能打的亮点）

### Situation
把 Python 版的 nano-vllm 用 C++ 重写完、跑通 Qwen3-0.6B 之后，decode 单步 ~22 ms，对应吞吐 839 tok/s。和 Python 版 1386 tok/s 差距巨大，明显是我自己的 C++ 实现还有性能没压榨出来。

### Task
找出 decode 路径的真正瓶颈并把 decode 单步降到 Python 版同量级以下。要求不能降低精度、不能改外部接口。

### Action

**步骤 1 — Profile**

用 `cudaEvent` 包住每个 kernel 再配 CPU 计时，拆出 decode 单步的 kernel 序列：

```
per-layer decode:
  qkv_proj (cuBLAS GEMM)
  ├─ split_qkv          ~18 us  ← 读 QKV (9 MB), 写 Q (3 MB) + K/V (3 MB × 2)
  ├─ q_norm             ~12 us  ← 读 Q, 写 Q
  ├─ k_norm             ~12 us  ← 读 K, 写 K
  ├─ rotary             ~15 us  ← 读 Q/K, 写 Q/K
  └─ store_kvcache      ~14 us  ← 读 K/V, 写 paged cache (gather 写入)
  attention (warp-per-head 自研)
  o_proj (cuBLAS GEMM)
  ...
```

28 层 × 5 个小 kernel = 140 次 launch，每次 ~5 us 的调度开销 + HBM 带宽被 Q/K/V 张量读写反复消耗。这是小 batch decode 的典型症状。

**步骤 2 — 集成 Flash Attention 2 decode (split-K)**

自研 attention 只够用，FA 的 online softmax + tile shaping 才是算力密集路径的正解。FA 没导出 C API（是 PyTorch extension），用 `dlopen("flash_attn_2_cuda.so")` + `dlsym` 抓 `run_mha_fwd_splitkv_dispatch<bf16, 128, false>` 的符号。关键参数：

- `is_seqlens_k_cumulative = false`（`cu_seqlens_k` 当作 `context_lens` 用，即每 seq 直接给 K 长度而不是前缀和）；
- `is_causal = false`（seqlen_q=1 时 causal mask 退化为无效果）；
- `force_split_kernel = true`（decode 必须走 split-K 版本，因为 batch=seqlen_q=1 下单 kernel 占用率太低）。

收益：839 → 1390 tok/s。

**步骤 3 — GQA seqlen_q↔ngroups swap**

读 FA 源码发现一个 trick：FA 内部按 seqlen_q 分 tile，seqlen_q=1 时大部分 warp idle。GQA 场景下，`num_heads = num_kv_heads × ngroups`，几个 Q head 共享同一个 KV head——这本质上跟"seqlen_q = ngroups 的同一个 seq"是等价计算。

做法：在调用 FA 前，把 Q 张量的 shape 从 `[bs, 1, num_heads, head_dim]` **用 stride 伪装**成 `[bs, ngroups, num_kv_heads, head_dim]`：

```
q_batch_stride = num_heads * head_dim
q_row_stride   = head_dim                  ← 这是关键：ngroups 方向连续
q_head_stride  = ngroups * head_dim
```

Qwen3 是 num_heads=16, num_kv_heads=8，ngroups=2，刚好让 FA 在 seqlen 方向多拿一倍 tile。输出再按同样的 stride 写回 `[bs, num_heads, head_dim]`，**不需要额外 transpose**。收益：1390 → 1550 tok/s。

**步骤 4 — 5-kernel 融合**

目标：把 `split_qkv → q_norm → k_norm → rotary → store_kvcache` 合为单 kernel `fused_split_norm_rope_store`。

- **grid 设计**：`(rows, num_q_heads + 2·num_kv_heads)`。每个 block 负责一个 (token, head) 对。
- **block 设计**：`head_dim = 128` 个 thread。
- **每个 block 干什么**：
  1. 从 qkv_proj 的输出 `[rows, (num_q_heads + 2·num_kv_heads) · head_dim]` 里读自己这 128 个元素到 register；
  2. 如果是 Q/K head：做 RMSNorm——用 `__shfl_xor_sync` warp 归约求平方和，shared memory 做 block-level 归约拿 `sum(x²)`，然后除以 `rsqrt(sum/head_dim + eps)` 并乘上 learnable scale；
  3. 如果是 Q/K head：做 RoPE——用 shared memory 做 pair 交换（每个 thread 持有 `x_i`，需要 `x_{i^1}` 来计算），然后按 `cos/sin` lookup 做旋转；
  4. 如果是 Q head：写到独立分配的 Q 输出张量；
  5. 如果是 K head：通过 `slot_mapping[row]` 算出 paged cache 中的物理位置，直接写入；
  6. 如果是 V head：跳过 RMSNorm 和 RoPE，直接按 slot_mapping 写入 paged cache。
- **省下的**：
  - 4 次 kernel launch（5 → 1）；
  - 4 次中间张量读写（Q/K 各读写 2 次）；
  - 2 次 per-layer 的 device allocation（K/V 不再需要独立 tensor）。

**步骤 5 — CUDA Graph + 更细 batch size 档位**

decode 的 kernel launch 序列非常固定（只有输入 tensor 内容在变），完美适合 CUDA Graph。原本档位是 `{1, 8, 16, 24, 32}`，实际 batch 落在档位之间就要 pad 到更大档（比如 bs=10 pad 到 16），做无用功。改成 `{1, 2, 4, 8, 12, 16, 20, 24, 28, 32}` 共 10 档（4 步一档），平均 pad 量减半。

捕获要求 `cudaMalloc` 不能在 capture 期间被调用，所以我写了 `ScratchAllocator`（见故事四）。

### Result

| 阶段 | Decode ms/step | 吞吐 |
|---|---|---|
| 自研 warp-per-head decode | 22.0 ms | 839 |
| + FA decode (split-K) | 13.1 ms | 1390 |
| + GQA seqlen-ngroups swap | 11.5 ms | 1550 |
| + CUDA Graph (旧档位) | 11.1 ms | 1627 |
| + 5-kernel 融合 + 更细档位 | 11.2 ms (eager) / **10.9 ms (graph)** | 1561 eager / **1661 graph** |

**总计 decode 单步 22 → 10.9 ms，吞吐 839 → 1661 tok/s (+98%)**，稳定版 1615 tok/s 超过 Python nano-vllm baseline +16.5%。

### 可能被追问

> **Q: 为什么一定要融合？不是 kernel launch overhead 现在很低吗？**
> A: 现代 GPU 每次 launch 到 issue 约 3-5 us，单看不算什么，但小 batch decode 下每个 kernel 本身只跑十几 us（grid block 数少），launch overhead 占比能到 30%。5 个串起来就是 25 us × 28 层 = 700 us 的纯调度时间。再叠加 HBM 带宽：Q/K 张量每次 3 MB，重读 4 次 = 36 MB/步 × 28 层的额外读写，对带宽紧的 laptop GPU 影响更大。

> **Q: RoPE 为什么要 shared memory？用 warp shuffle 行不行？**
> A: 行。`__shfl_xor_sync(mask, x, 1)` 就能在 warp 内换 pair。之所以全走 shared，是因为 head_dim=128 超过 warp 宽度 32，跨 warp 的 pair 还是得通过 shared，为了代码统一、减少分支。如果未来 head_dim 变成 64（刚好一个 warp），纯 shuffle 会更快一点但是差距 <5%。

> **Q: 融合 kernel 对 INT8 KV 友好吗？**
> A: 不友好。融合 kernel 直接写 bf16 到 K/V cache，INT8 量化需要 per-head 算 scale 再量化，逻辑不一样。所以 Attention 层里有个 `is_int8_` flag，INT8 路径走回原来的 5-kernel + `store_kvcache_int8`。**功能 vs 性能的 trade-off，显式化并记录在代码注释里。**

> **Q: FA 是 Python extension，你怎么在 C++ 里用？**
> A: 两个关键点。第一，FA 编译出来的 `.so` 里其实有 C++ symbol 可以被 `dlsym` 抓到，只要 demangle 过的 mangled name 对得上就行——用 `nm -C flash_attn_2_cuda.so | grep run_mha_fwd_splitkv_dispatch` 能看到。第二，PyTorch 的 `at::Tensor` API 我不能用，但 FA 的底层 `Flash_fwd_params` struct 是纯 C++ 的，我只要手动填 stride/ptr/seqlen 就行，绕过了 PyTorch。运行时要保证 `libtorch.so` 能被加载（因为 FA 间接依赖），在 bench 时加 `LD_LIBRARY_PATH` 即可。

> **Q: seqlen_q↔ngroups swap 在 FA 源码里是怎么回事？**
> A: FA 源码里有个 `seqlenq_ngroups_swapped` 布尔参数（默认 false），文档不多。我一开始打开它做调用但发现 stride 算错——它 `true` 时会额外把 `q_batch_stride` 乘 seqlen_q，而我这里 seqlen_q 已经是 1。所以最后选择了"保持 swapped=false，手动设置 stride"的路子，数值上完全等价。这个坑读 FA 源码读了两小时才弄明白。

> **Q: 你的融合 kernel bank conflict 怎么处理？**
> A: shared memory 分配是 `head_dim + padding`，避开 128 × 4 = 512 字节的 bank 冲突模式。实测 Nsight Compute 报告 shared memory 吞吐 95%+，没有显著 bank conflict。



---

## STAR 故事二：Chunked Prefill + Pinned Staging（serving 侧收尾）

### Situation
Bench 跑出 1615 tok/s 之后观察到两个问题：

1. **小 batch decode host 侧成本可测**：用 `std::chrono` 包 `prepare_decode` 测到 ~0.5-1 ms 的 CPU 时间，每步都重新 `std::vector` 构造 5~6 个小 int 数组然后 `cudaMemcpyAsync`。用 Nsight Systems 看 timeline，这些 `cudaMemcpyAsync` 实际在同步等—— `std::vector` 的堆内存是 pageable 的，`cudaMemcpyAsync` 在 pageable memory 上会退化成同步拷贝。对 bs=1/2 档位 decode step 时间 ~8 ms 的情况，这 1 ms 是真能看到的。
2. **长 prompt 阻塞短 decode 请求**：比如 2K system prompt + 短 query，prefill 步得一次性把 2K token 压过去，期间所有其他已经在 decode 的 seq 都得等着——TTFT 抖动非常明显。原来的 `schedule_chunked` 函数是个注释占位，没真正实现。

### Task
1. 把所有 `prepare_*` 的 host→device 拷贝改成真正异步；
2. 实现最小可用的 chunked prefill，默认关闭、开启时保证 decode 输出 bit-exact，不能引入新测试失败。

### Action

**步骤 1 — Pinned Staging（低风险先做）**

关键观察：一个 `step()` 里所有 host buffer 的生命周期**就是这一个 step**。step 结束后我们就不再需要它们了。这种"整批分配、整批释放"的模式用 bump allocator 最合适。

设计：

```cpp
class ModelRunner {
    std::unique_ptr<PinnedHostBuffer> pinned_staging_;  // cudaHostAlloc
    size_t pinned_offset_ = 0;

    void* pinned_alloc(size_t bytes) {
        size_t aligned = (bytes + 15) & ~size_t(15);   // 16-byte 对齐
        if (!pinned_staging_ || offset + aligned > capacity) {
            // 倍增。关键：grow 前必须 cudaDeviceSynchronize
            // 因为旧 buffer 上可能还有 in-flight H2D
            if (pinned_staging_) cudaDeviceSynchronize();
            pinned_staging_ = new PinnedHostBuffer(new_capacity);
            pinned_offset_ = 0;
        }
        void* ptr = base + pinned_offset_;
        pinned_offset_ += aligned;
        return ptr;
    }
    void reset_pinned_staging() { pinned_offset_ = 0; }
};
```

`make_device_int_tensor / make_device_float_tensor` 改成：先 `memcpy` 到 pinned 区域，再 `cudaMemcpyAsync`。`run()` 每步开头调 `reset_pinned_staging()`，offset 归零。覆盖 `prepare_prefill / prepare_decode / prepare_sample / build_block_tables` 全部 H2D 路径。

**坑点**：`pinned_alloc` 扩容时如果前一次 async H2D 还没 drain，直接 `delete` 旧 buffer → undefined behavior。解决：grow 前 `cudaDeviceSynchronize`。虽然同步一次很重，但扩容是冷路径（每 session 几次），成本可忽略。

**步骤 2 — Chunked Prefill（系统性改动）**

核心设计决策：**先不追求"同 step 混 prefill+decode"，先做"长 prompt 多步分块"**（记为 Option A）。因为：

- Option A 不需要改 attention kernel（FA varlen 和自研 paged prefill 都天然支持 `q_len < k_len`）；
- Option A 不需要改 context 组织（一个 step 里要么全 prefill，要么全 decode）；
- Option A 已经解决了最核心的问题——长 prompt 阻塞短 decode。

`Sequence` 加两个字段：

| 字段 | 含义 | 生命周期 |
|---|---|---|
| `num_computed_tokens` | KV 已经物化的 token 数（水位） | 持久，跨 step |
| `num_tokens_to_process` | 本 step 分配给这个 seq 的 query 长度 | transient，每 step 被 scheduler 覆写 |

保持 `num_cached_tokens`（prefix cache 命中数）不变。Invariant：`num_cached_tokens ≤ num_computed_tokens ≤ num_tokens`。

**Scheduler 重写**：

```
schedule_chunked():
    # Phase 1: 新 seq 进入 running
    for seq in waiting:
        if not block_manager.can_allocate(seq): break
        block_manager.allocate(seq)          # prefix cache 命中在这一步生效
        seq.num_computed_tokens = min(seq.num_cached_tokens, seq.num_tokens - 1)
        # ↑ 留至少 1 个 token 给最后的 decode 出第一个 logit
        running.push_back(seq)

    # Phase 2: 找出还需要 prefill 的 seq
    prefill_candidates = [seq for seq in running
                          if seq.num_computed_tokens < seq.num_tokens - 1]

    if prefill_candidates:
        budget = max_num_prefill_tokens
        for seq in prefill_candidates:
            remaining = seq.num_tokens - seq.num_computed_tokens
            chunk = min(remaining, budget)
            seq.num_tokens_to_process = chunk
            budget -= chunk
            if budget == 0: break
        return (prefill_batch, is_prefill=True)

    # Phase 3: 没有 prefill 需求，走 decode (标准 default 逻辑)
    ...
```

**prepare_prefill 改造**：

```cpp
int query_begin;
int query_end;
if (seq->num_tokens_to_process > 0) {           // chunked 路径
    query_begin = seq->num_computed_tokens;
    query_end   = query_begin + seq->num_tokens_to_process;
} else {                                        // 兼容旧路径
    query_begin = seq->num_cached_tokens;
    if (query_begin >= seq->len()) query_begin = seq->len() - 1;
    query_end   = seq->len();
}

int q_len = query_end - query_begin;
int k_len = query_end;            // 包含已缓存前缀（含历史 chunk 的 KV）

cu_seqlens_q.push_back(cu_seqlens_q.back() + q_len);
cu_seqlens_k.push_back(cu_seqlens_k.back() + k_len);
```

注意 `cu_seqlens_k = query_end`，不是 `seq->len()`。这样 FA varlen / paged prefill kernel 就只 attend 到"当前已经 materialize 的 KV"——当前 chunk 的 Q 对 [历史 chunks + 当前 chunk] 的 KV 做 attention，天然符合 causal 语义。

**postprocess 三分支**：

```cpp
if (seq->num_tokens_to_process > 0) {
    // prefill step
    bool terminal = (seq->num_computed_tokens + seq->num_tokens_to_process
                     == seq->num_tokens);
    seq->num_computed_tokens += seq->num_tokens_to_process;
    seq->num_tokens_to_process = 0;
    if (!terminal) continue;       // 中间 chunk，sample 是垃圾数据丢弃
    // fall through: 终端 chunk 走下面的 append_token 逻辑
}
// decode step 或 prefill 终端 chunk
seq->append_token(token_id);
if (EOS or max_tokens reached) { seq->status = FINISHED; block_manager.deallocate(seq); }
```

**步骤 3 — 测试（验证数值等价）**

两个测试保证不回退：

1. `SchedulerTest.ChunkedPrefillSplitsLongPromptAcrossSteps`：300-token prompt，`max_num_prefill_tokens=128`，预期切成 128+128+44 三步，第三步是 terminal；
2. **`ModelRunnerTest.ChunkedPrefillMatchesSingleShotPrefillAndFirstDecodeToken`（关键）**：同样 prompt 用 baseline（一次性 prefill + 首 decode）拿到一个 `baseline_token`，然后换 chunked 路径切 256+128 两步，**验证终端 chunk sample 出来的 token == baseline_token**。这个测试保证 chunked prefill 不改变数值语义。

写这个测试的时候踩了一个坑：最初的 predicate `num_computed_tokens < num_tokens`（而不是 `< num_tokens - 1`）会让终端 chunk 变成 0 token 长度，scheduler 一直想再跑一步 prefill。改成 `< num_tokens - 1` 后，最后 1 个 token 留给 decode step 出 first logit，数值就对齐了。

### Result

- Pinned staging 让 decode 路径的 H2D 真异步化（Nsight timeline 中 `cudaMemcpyAsync` 变成几乎 0 时间），对小 batch decode 收益明显。
- Chunked prefill 最小版本可用：开启时长 prompt 不再阻塞 decode，默认关闭时所有旧测试行为不变。
- **全套 10/10 gtest 通过**，含两个新 chunked prefill 测试、双卡 TP 回归、end-to-end LLMEngine 测试。
- Prefix Cache 这块**不是这次新做的**——`BlockManager` 本来就有 `hash_to_block_id_`（xxhash 链式 `prev_hash` + 当前 block token_ids）；这次的改动让它跟 chunked prefill 正确配合（`num_computed_tokens` 初始化自动跳过 cache 命中部分）。

### 可能被追问

> **Q: Chunked prefill 对 attention kernel 有什么要求？**
> A: 需要 kernel 能支持 `q_len < k_len`——部分 query 对完整 K/V 做 attention。FA varlen 本来就支持（通过 `cu_seqlens_q / cu_seqlens_k` 独立传入）；自研的 paged prefill kernel 也是用 `(cu_seqlens_k_diff - cu_seqlens_q_diff)` 隐式推出 cached prefix 长度。所以这次改动不需要动任何 kernel。

> **Q: Chunked prefill 怎么和 Prefix Cache 配合？**
> A: Prefix Cache 命中只影响 `num_cached_tokens`（整 block 对齐，单位 256 token）；Chunked prefill 的 `num_computed_tokens` 以 token 为粒度推进。初始化时 `num_computed_tokens = min(num_cached_tokens, num_tokens-1)`，保证前缀命中的部分直接跳过、中间 chunk 接着算。

> **Q: Pinned buffer 为什么用 bump，不用 pool？**
> A: 每步的 host buffer 生命周期就是一个 step，step 结束 reset offset 就行。bump 最简单、fragmentation 为 0、alloc 是 O(1)。Pool 需要追踪每个小块的归还时机，对这种"整批释放"场景反而是 overhead。

> **Q: 为什么不直接在同一个 step 里混 prefill + decode？**
> A: 那是 full chunked prefill，需要：(1) attention 能接受一个 batch 里有 seq q_len=1、有 seq q_len=chunk（FA varlen 能做但 `cu_seqlens_q/k` 要合并构造）；(2) scheduler 要在同一步里同时 append 并管理两类 seq 的 block table。当前是最小可用版本，**长 prompt 先拆到自己的多 step 里**，避免阻塞其他 seq 的 decode——已经解决最主要的 TTFT 问题。混 batch 是下一步。

> **Q: pinned memory 不是有 OS 级页锁成本吗？大量用会不会有副作用？**
> A: `cudaHostAlloc` 会把物理页 pin 住，操作系统不能 swap。成本在于 (1) 分配慢（毫秒级）；(2) 占用不可换出物理内存。我的 bump 策略确保整个 session **只分配几次**（倍增），单次容量也不超过几十 KB（prepare_* 的 host buffer 总量很小），所以 OS 成本可忽略。如果有人跑超大 batch 一次 prepare 要几 MB 的 host 数据，需要考虑加容量上限 + fallback。

> **Q: chunk 大小怎么选？**
> A: 参数 `max_num_prefill_tokens`，默认 512。大了 → TTFT 抖动恢复；小了 → chunk 切得碎，prefill 效率低（FA varlen tile 吃不满）。512 是经验值——一个 tile 能打满 128-head_dim × 4 tile 的 FA 配置。真实 serving 里应该随流量动态调。

> **Q: prefix cache 的 hash 冲突你处理了吗？**
> A: `hash_to_block_id_` 的 value 是 block_id，命中后**二次校验 token_ids 相等**。xxhash 64-bit 冲突概率已经极低，加上二次校验做到 0 冲突。代价是命中失败要重新走 miss 路径，但 miss 本身就要新建 block，没额外开销。



---

## STAR 故事三：WSL libcuda.so 冲突（体现 debug 能力）

### Situation
在 WSL Ubuntu 上跑 bench，Flash Attention 死活加载不上，ctest 里 test_llm_engine 偶尔 CUDA init 失败。`nvidia-smi` 正常、`cuda-toolkit` 装得好好的。

### Task
定位原因、给出稳定的修复，不能假设所有用户都会手动改系统。

### Action
1. `ldd` 和 `LD_DEBUG=libs` 发现 `libcublas.so` 依赖 `libcuda.so.1`，而 dlopen 找到的是 `/lib/x86_64-linux-gnu/libcuda.so.535.288.01` —— 一个不属于任何 apt 包的孤儿原生驱动，**shadow 了 WSL 提供的 stub** `/usr/lib/wsl/lib/libcuda.so.1`。
2. 查 `readelf -d libcublas.so` 发现 `RUNPATH: [$ORIGIN]` 直指 `/lib/x86_64-linux-gnu/`，这让 dlopen 在 ldconfig cache 之前命中了错的那个。
3. 试过几条路都不行：
   - 给可执行文件加 `DT_RPATH` → 不生效（dlopen 走的是被动加载，不看 RPATH）；
   - 构造函数里 `dlopen` 预加载 WSL 的 libcuda → 静态初始化顺序不可靠；
   - CMake OBJECT library → 链接阶段没用。
4. 最终方案：CMake 里探测 `/usr/lib/wsl/lib/libcuda.so.1` 存在时设置 `NANO_VLLM_WSL=1`，然后对所有 ctest target 用 `set_tests_properties(ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib:$ENV{LD_LIBRARY_PATH}")`。独立运行时用户也只需要一行 `LD_LIBRARY_PATH=/usr/lib/wsl/lib` 前缀。
5. **附带收获**：同时发现 `CMAKE_CUDA_ARCHITECTURES` 必须在 `project()` 之前设置，否则 CMake 自动探测阶段已经锁死架构，后面再 `set` 是无效的。

### Result
- ctest 在 WSL 上稳定通过；
- 修复写进 CMakeLists 后新用户 clone 下来开箱即用；
- 把这个 trap 记录在 repo memory 里，类似的链接器问题以后能秒识别。

### 可能被追问

> **Q: 为什么 RPATH 不影响 dlopen？**
> A: RPATH/RUNPATH 只影响**由 dynamic linker 执行时解析的依赖**。dlopen 走的是运行时再次触发 search path 解析，顺序是 `LD_LIBRARY_PATH → RUNPATH of caller (不是主程序) → ldconfig cache → 默认路径`。主程序的 RPATH 不在这条链上。这个细节在 `man ld.so` 里有写但不显眼。

> **Q: 怎么永久修？**
> A: 最彻底的修复是 `sudo rm /lib/x86_64-linux-gnu/libcuda.so*` + `sudo ldconfig`，让系统上只剩 WSL stub。但这是侵入性的改动，所以我选项目内的非侵入方案——用户 clone 下来什么都不用做。

> **Q: 怎么判断是否在 WSL 下？**
> A: CMake 里检查 `/usr/lib/wsl/lib/libcuda.so.1` 是否存在。存在且同时在 `/lib/x86_64-linux-gnu/` 有另一份 → WSL 冲突场景，启用 ENVIRONMENT 注入。比用 `/proc/sys/kernel/osrelease` 抓 `WSL` 字串更准——这个方案判断的是"有没有冲突"而不是"是不是 WSL"。

---

## STAR 故事四：CUDA Graph + ScratchAllocator

### Situation
5-kernel 融合 + FA + GQA swap 做完后 eager 模式已经到 1550 tok/s，Nsight timeline 显示 **decode step 里 kernel 之间还有几百 us 的 idle 间隙**——这是 host 驱动 kernel launch 的调度时间。decode 每步的 kernel 序列完全固定（只有输入 tensor 数值在变），正好适合 CUDA Graph。

### Task
把 decode step 的 model forward 用 CUDA Graph 捕获、replay，消除 launch overhead；不能影响 eager 路径（某些场景如 warmup 需要 eager）。

### Action

**核心难点：capture 阶段不能调 `cudaMalloc`**

stream capture 模式下 `cudaMalloc` 会报错。但我的 kernel 里大量临时张量（比如 attention 的 Q 输出、O-proj 之前的 contiguous buffer）都是动态分配的。硬改所有 kernel 去用 pre-allocated buffer 成本太高。

**解决方案：ScratchAllocator**

一个继承自 `DeviceAllocator` 的 bump 分配器，底层是一大块 `cudaMalloc` 的 scratch buffer：

```cpp
class ScratchAllocator : public DeviceAllocator {
    char* base_;
    size_t offset_ = 0;
    size_t capacity_;
public:
    Memory::Ptr allocate(size_t bytes, Device device) override {
        size_t aligned = (bytes + 255) & ~size_t(255);   // 256-byte 对齐
        if (offset_ + aligned > capacity_) throw;
        void* ptr = base_ + offset_;
        offset_ += aligned;
        return Memory::make_borrowed(ptr, aligned, device);
    }
    void reset() { offset_ = 0; }
};
```

每次 capture 或 replay 前 `reset()`，capture 期间所有 kernel 的 tensor 都从这里 bump 分配——不会触发 `cudaMalloc`，且分配出来的地址是**固定的**（第一次 capture 什么地址，replay 也是什么地址），所以 kernel 参数里的指针直接有效。

**分档捕获**

decode batch size 不固定（avg 19.3），CUDA Graph 是固定 shape 的。策略：预先捕获 `{1, 2, 4, 8, 12, 16, 20, 24, 28, 32}` 共 10 个 graph，运行时 `std::lower_bound` 选最小满足 `bs_captured >= actual_bs` 的档位，pad 到该档。

Padding 的 trick：输入张量里多出来的 row 填 0（embedding 查 0-th token），输出时只取前 `actual_bs` 行。这样 kernel 对 pad 的 token 也算了一轮，浪费一些 FLOP，但换来了 launch overhead 归零。

**Replay 时怎么喂新输入？**

Capture 时为每个 graph 预分配了固定的输入 buffer（`input_ids`, `positions`, `slot_mapping`, `context_lens`, `block_tables`）。replay 时：

```cpp
cudaMemcpyAsync(captured.input_ids.data(), actual_input_ids, ..., stream);
cudaMemcpyAsync(captured.positions.data(), ..., stream);
...
cudaGraphLaunch(captured.exec, stream);
```

指针没变，数值变了，graph 照样跑。

**容量估算**

scratch buffer 大小要够最大 batch size 的 decode step 用。测出来大约 44 MB（Qwen3-0.6B, bs=32），一次 `cudaMalloc` 分配好放着。

### Result

- **eager 1550 → graph 1627 tok/s** (BS={1,8,16,24,32} 旧档位)
- **eager 1561 → graph 1661 tok/s** (5-kernel fusion + BS 步长 4 新档位)
- Stable bench 1595-1615 tok/s (CUDA Graph enabled by default)。

### 可能被追问

> **Q: 为什么 ScratchAllocator 对所有 scratch 够用？你怎么保证不会 OOM？**
> A: 测出来的 peak = 44 MB，对应 bs=32 的 Qwen3-0.6B decode 的一步里所有临时张量。这个值在 warmup 阶段通过 dry-run（不 capture，只记录 allocate 调用）测出来，加 20% margin 后预分配。如果未来换更大模型或更大 batch，peak 会线性增长，预分配量也要相应调整——有个 config 参数 `scratch_buffer_size_mb` 能 override。

> **Q: prefill 为什么不用 CUDA Graph？**
> A: prefill 的 `q_len` 每个请求都不一样（取决于 prompt 长度），shape 不固定。如果按所有可能的 q_len 都捕获，graph 数量爆炸。而且 prefill step 本身是 GEMM-bound 的，launch overhead 占比不高，ROI 低。

> **Q: graph 更新 block_tables 这种 2D 变长张量怎么办？**
> A: `block_tables` 的第二维 (max_blocks) 随请求长度变。方案是取一个够大的固定 max_blocks 值（比如 `max_model_len / block_size + 2 = 18`），pad 剩下的位置。kernel 里用 `context_lens` 判断哪些 block 是真实有效的，越界的 block 索引不读。



---

## STAR 故事五：INT8 KV Cache 端到端（量化 + 工程 tradeoff）

### Situation
Qwen3-0.6B 本身 KV cache 不大（~1.5 GB）在 4060 上没问题，但想支持更大模型（TinyLlama-1.1B / LLaMA-3-1B）和更长 context 时，KV cache 是主要显存消耗。INT8 量化 KV 可以直接砍半。

### Task
端到端实现 INT8 KV cache：分配、写入、decode 读取并在 kernel 里在线反量化；精度在小模型上肉眼无明显劣化；通过 config 一键开启。

### Action

**量化粒度选择**

可选粒度：per-tensor、per-token、per-head per-token、per-element。选 **per-head per-token 对称量化**，理由：

- per-tensor 精度差，尤其 attention output 分布 outlier 多；
- per-element 就是 FP8，硬件支持更直接；
- per-head per-token 在 head 维度独立算 scale，outlier 不会传染到别的 head；粒度细到能吸收 head 间分布差异，scale 存储量只有 `num_kv_heads × num_tokens`（fp16），占用 KV cache 总量的 <1%。

存储布局：

```
kv_cache       int8  [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
kv_scale       fp16  [num_layers, num_blocks, block_size, num_kv_heads]
```

K 和 V 各一份（实际是 `[2*num_layers, ...]`）。

**量化 kernel (`store_kvcache_int8`)**

一个 warp（32 thread）负责一个 (token, kv_head)。head_dim=128，32 thread 每人处理 4 个元素：

1. 各自算 |x|，warp `__shfl_down_sync` 归约求 `max_abs`；
2. `scale = max_abs / 127.0f`；
3. 各自做 `int8_val = round(x / scale)` + clamp 到 [-128, 127]；
4. 通过 `slot_mapping[token]` 定位 paged cache 物理位置，写 int8 值 + fp16 scale。

**反量化 kernel (`decode_paged_attention_int8`)**

在原始 decode attention kernel 基础上改：K/V 读入时每 128 元素先读 1 个 fp16 scale，再乘以每个 int8 值转 fp16，然后进入原来的 online softmax 路径。其他不变。

Prefill 路径**不走 INT8**——FA 不支持 int8 K/V 输入，prefill 仍然使用 dense fp16 K/V 算 attention，只有存到 paged cache 时才量化。这样 prefill 精度 = 原始，decode 精度略降。

**Config 打通**

`Config::kv_cache_int8` 布尔值，从 `Config::from_model_dir()` 接口传入。`ModelRunner::allocate_kv_cache` 分两分支分配（int8 cache + fp16 scale tensor 或单一 fp16 cache）。Delegation 链：`TransformerForCausalLM → Model → DecoderLayer → Attention`，每级都有 `set_kv_cache_int8()`。

`Attention` 里的 `is_int8_` flag 控制两件事：
1. 选 `store_kvcache_int8` vs `store_kvcache`；
2. 选 `decode_paged_attention_int8` vs `flash_attn_decode_kvcache`（FA 不能跟 int8 配）；
3. **跳过 `fused_split_norm_rope_store` 融合路径**（fused kernel 只写 fp16）。

**精度验证**

在 Qwen3-0.6B 和 TinyLlama-1.1B 上跑了几十个 prompt 的对比（fp16 vs int8）：
- Greedy decode 的 token 差异 < 5%；
- 肉眼读输出文本，风格和语义连贯性一致；
- 没有出现典型量化 bug 的 "repetition collapse"。

### Result

- **显存减半**（K/V cache 从 fp16 16-bit → int8 8-bit，scale 占 < 1%）；
- 吞吐基本持平 fp16（decode 阶段反量化是 memory-bound kernel 的 marginal cost，online softmax 的 compute 占主导）；
- CLI 开关：`./build/example /path/to/model --int8-kv`；
- 代码里 int8 路径和 fp16 路径并存且**没有互相污染**（通过 `is_int8_` flag 显式分叉）。

### 可能被追问

> **Q: 为什么不用非对称量化？**
> A: KV cache 的数值分布通常以 0 为中心（LayerNorm / RMSNorm 后），非对称的 zero-point 带来的精度收益不大，但每元素多一个 zero-point 存储 + 反量化时多一次减法。对称量化 scale 单精度足够，zero-point 恒为 0，kernel 简单。

> **Q: scale 为什么用 fp16 而不是 fp32？**
> A: scale 本身精度要求不高（它只是还原动态范围，bias 在 softmax 里会被 normalize 掉），fp16 够用；fp16 scale 存储量减半；kernel 里 fp16 × int8 乘法刚好能用 half2 向量指令。

> **Q: 为什么不把 prefill 也做成 int8？**
> A: 两个原因：(1) FA 的 API 不接受 int8 K/V 输入，我们没有自研到 FA 级别的 int8 kernel；(2) prefill attention 的 Q/K/V 参与 softmax 计算，精度要求高于 decode（decode 每步只对已有 cache 做一次 attention）。折中的做法是"prefill fp16 算 attention、存 cache 时量化，decode 读取时反量化"。

> **Q: 长序列下 per-head per-token 的 scale 张量会不会占太多？**
> A: scale 大小 = `2 · L · blocks · block_size · num_kv_heads · 2 bytes`。对 Qwen3-0.6B（L=28, num_kv_heads=8, blocks=max），大概是 KV cache 总量的 0.8%，可忽略。

---

## 高频追问 Q&A（技术点深挖）

### PagedAttention / BlockManager

- **block_size=256 怎么选的？** 256 token × head_dim × num_kv_heads × 2 bytes ≈ 0.25 MB，刚好能吃满一次 HBM burst；hash 计算开销（xxhash 64-bit over 256 int32）~0.5 us 可忽略；再大会让 allocate 粒度粗、小 prompt 浪费；再小 hash 频次高。
- **`ref_count` 怎么用？** `allocate` 时 +1（或原来是 0 就从 free list 拉出来 +1）；`deallocate` 时 -1，归零才回 free list。多 seq 共享前缀 block 时自动工作（allocate 第 N+1 个 seq 时如果 hash 命中，block 已经 used_block_ids_ 里，仅 +1 不重复从 free 拉）。
- **`deallocate` 不从 hash 表移除条目：为什么？** 因为 block 回 free list 之后如果没被覆盖，里面的 token_ids 还是有效的，下次新请求命中 hash 时能"起死回生"（重新 allocate 这个 free block 并且复用旧数据）。代价：hash map 单调增长，但 key 数 ≤ 总 block 数，上界可控。这也是 vLLM 的标准做法。
- **冲突怎么处理？** 查 map 命中后**二次比对 token_ids**。只用 hash 作索引，不把 hash 当唯一身份。
- **hash 链式什么意思？** `hash(block_i) = xxhash(prev_hash ⊕ token_ids_of_block_i)`。这样两个 seq 只要前缀完全一样，第 k 个 block 的 hash 也一样；前缀有差异则整条链不同 block。

### Continuous Batching / Scheduler

- **为什么 prefill 优先？** prefill 一步能处理上百 token 吞吐高，且 prefill 完成后 seq 才能进 decode 流程，优先跑 prefill 能让更多 seq 尽早贡献 decode。
- **decode 用 LIFO 抢占：什么意思？** 如果 decode step 里某个 seq 要申请新 block 但 free list 空了，scheduler 从 `running_` 尾部（最后进入 running 的 seq）踢回 `waiting_`、释放它的 block 给当前 seq 用。LIFO 保证"老 seq 优先完成"——已经投入计算的不浪费。
- **抢占会不会死锁？** `preempt` 会清空被踢 seq 的 block_table，重新进 waiting 时从头开始（需要 re-prefill）。如果全部 running 都踢光当前 seq 还申请不到 block，就抢占当前 seq 自己。理论上单 seq > 全部 blocks 的情况会死锁，在 `can_allocate` 检查里提前拦截。
- **chunked 和 default 怎么切换？** Config `enable_chunked_prefill` 决定。运行时不能切（scheduler 状态会不一致）。

### CUDA Graph

- **哪些操作不能 capture？** `cudaMalloc` / `cudaFree` / `cudaMemcpy`（同步版本）/ 任何 host-device 同步点（`cudaDeviceSynchronize` / `cudaStreamSynchronize`）/ cuBLAS / cuDNN 早期版本里的某些 workspace 分配路径。
- **cuBLAS 怎么进 capture？** 用 `cublasSetStream` 指定 capture stream，然后调 `cublasGemmEx` 正常调。关键是 **handle 在 capture 前必须已经初始化并且 workspace 已经分配好**——cuBLASLt 会在第一次 call 时懒分配 workspace，所以要 warmup。我在 `warmup_model` 里先跑一次非 capture 的 forward 触发所有懒分配。
- **如何 replay？** `cudaGraphLaunch(exec, stream)`。可以重复多次。输入通过 `cudaMemcpyAsync` 写到 pre-allocated buffer（指针不变）。
- **graph 占多少显存？** `ExecutableGraph` 本身只存 kernel metadata + 参数 snapshot，MB 级。真正大的是 ScratchAllocator 预分配的 44 MB。

### Flash Attention

- **FA 的核心思想？** Online softmax：分块处理 K/V，维护累积的 `m = max(scores)` 和 `l = sum(exp(scores - m))`，每新 tile 更新 `m_new`, `l_new`, `o` 用数学上等价的 rescale 公式，避免存储 N×N attention matrix。存算比从 O(N²) 降到 O(N·d)。
- **decode 为什么要 split-K？** 普通 FA 按 `(seqlen_q / TILE_Q) × batch × num_heads` 作 grid，seqlen_q=1 时 grid 维度太小，GPU 利用不满。split-K 把 K 维度也切成多个 chunk 并行处理，各 chunk 独立算 partial O 和 partial lse，最后 reduce 合并。
- **FA varlen 是什么？** 变长输入：用 `cu_seqlens_q` / `cu_seqlens_k` 数组标记每个 seq 的 token 起止位置，所有 seq 的 token 拼在一起不 pad。适合 prefill 里不同 prompt 长度的 batch。
- **为什么不直接用 xFormers / SDPA？** SDPA 不支持 paged KV cache（它只能读连续 K/V），而我们的 decode KV 是分散在 paged blocks 里的。FA 有 `cache_batch_idx / cache_leftpad / block_table` 参数族能处理 paged。

### INT8 / 量化

- **量化误差从哪来？** (1) 舍入误差（int8 只有 256 级）；(2) outlier 被 clip 到 ±127·scale；(3) 反量化乘 scale 时的 fp16 精度损失。per-head per-token 粒度已经把 (2) 压到很小。
- **SmoothQuant 是什么？为什么不用？** SmoothQuant 对激活做 per-channel scale shift，把 outlier 从 activation 移到 weight 上让 activation 更好量化。适合 activation int8 场景，不适合纯 KV 量化。
- **你了解 Marlin / AWQ / GPTQ 的区别吗？** GPTQ 基于 Hessian 的 column-wise 量化（训练时需要 calibration data）；AWQ 也要 calibration，但按 activation 重要性做 per-channel mixed precision；Marlin 是一个针对 A100/H100 的 INT4 weight-only GEMM kernel（支持 group-wise 量化的高效反量化）。下一步想集成的是 AWQ 权重 + Marlin kernel。

### TP / NCCL

- **哪些层需要 all-reduce？** Row-parallel Linear 输出需要跨 rank sum（因为每个 rank 只算了 partial sum）；VocabParallelEmbedding 输出需要 sum（部分 vocab 在本 rank，其他位置是 0）。即 **每个 transformer layer 有 2 次 all-reduce**（attn 的 o_proj 后 + mlp 的 down_proj 后）。
- **ParallelLMHead 为什么用 gather_last_dim_to_rank0 而不是 all-reduce？** 只有 rank 0 需要算 sampler 和做输出，其他 rank 不需要 logits，gather 到 rank 0 比 all-reduce 省一半带宽。
- **TP 通信怎么测？** 写了一个 `TensorParallelCommunicator` 抽象，in-process 版本在同进程里模拟 N 个 rank 做 all-reduce（memcpy 拼接），用于单元测试。NCCL 版本是生产路径。测试覆盖率：单测里能保证 TP=2 结果与 TP=1 数值等价。
- **CUDA Graph + NCCL 兼容吗？** NCCL 2.10+ 支持 stream capture，需要 `ncclGroupStart/End` 包裹，且通信子要正确初始化。当前项目实现里 NCCL 还没 capture 到 graph 里（单卡 bench 跑 graph，多卡 bench 跑 eager），这是已知 limitation。

### 采样

- **Top-k 怎么在 GPU 上做？** 不用真的排序。用二分查找：猜一个 threshold v，数 logits 中 >= v 的个数；如果 > k 就调高 v，反之调低。log(2^32) ≈ 32 次 block-level 归约，但我写了 50 次（保守）。
- **Top-p 同理**：对概率分布的 CDF 做二分。
- **Gumbel-max trick**：`argmax(logits + Gumbel(0,1))` 等价于从 softmax 采样，但不需要显式计算 softmax 归一化常数。kernel 里把 `logits / temperature + gumbel` 一起算，argmax 直接出 token id。
- **重复惩罚 kernel**：对 seq 已经生成过的 token id 列表，查表找到对应 logits 位置，正值除以 penalty、负值乘 penalty。`penalty_token_ids` 是 padded 2D tensor，`penalty_token_counts` 标记每行真实有效长度。

### 精度 / dtype

- **项目主 dtype 是 bf16 还是 fp16？** bf16（跟 Qwen3 原始权重一致）。FA2 里跑 bf16 kernel；RMSNorm/RoPE/Sampling 在 fp32 中间结果上算（数值稳定），最后写回 bf16。
- **为什么 RMSNorm 归约要 fp32？** `sum(x²)` 在 fp16 里容易 overflow（x² 最大 65504），bf16 又精度不够。fp32 中间结果是标准做法。
- **bf16 vs fp16 有什么差别？** bf16 有 8 位 exp 和 7 位 mantissa，动态范围大（和 fp32 一样）但精度低；fp16 有 5 位 exp 和 10 位 mantissa，精度高但动态范围小（容易上下溢）。LLM 训练用 bf16 占绝大多数，推理两者都行。

### 工程 / 其他

- **为什么用 xxhash 而不是 SHA？** xxhash 快 100× 以上（SIMD 实现），强度够用（不是密码学场景），没有冲突攻击威胁。
- **SafeTensors 怎么加载？** `mmap` 文件到进程地址空间，头部 JSON 解析出每个 tensor 的 offset + dtype + shape。加载时 `cudaMemcpy` 指定区域到 device 即可。bf16 → fp32 的 dtype 转换走一个 host-side buffer 中转。
- **为什么给每个 kernel 写独立测试？** 不是每个——整合在 `test_tensor.cpp` 和 `test_layers.cpp` 里。比如 RMSNorm 有 vectorized / fallback 两条路径，分别测。store_kvcache 测 slot_mapping=-1 跳过的情形。目标是发现 regression，不是 100% 覆盖率。
- **重写占了多少时间？** 从 config / sequence / block_manager 这几个数据结构开始，到 kernel 全部完成，到 LLMEngine 端到端跑通、再到 bench 数字达到 1615，整体大约 3-4 周密集投入。

---



## 通用行为问题备用（HR / 技术 leader 问）

### "你遇到最难的技术问题是什么？"
→ 用故事一（5-kernel 融合 + GQA swap）。

### "你做这个项目学到了什么？"
> 最大的收获是"对每一毫秒去哪了都能说清楚"。以前只会用 vLLM、知道它快，但不知道为什么快。亲手写一遍之后，能分得出来"这 1ms 是 launch overhead，这 2ms 是 HBM 带宽"，也知道哪些优化是换钱（INT8 KV 省显存）、哪些是换命（5-kernel 融合省延迟）。这种分辨能力没办法光看论文学到。

### "为什么不直接贡献 vLLM？"
> vLLM 代码量 20 万+，上手成本高，而且很多优化点已经被填过。我选从零写是因为想把"框架空洞"（没有 Python GIL、没有 5 层抽象）亲手填一遍，反而能感受到每个设计决策背后的 trade-off。等这个项目稳定后再去 vLLM 贡献，读它的 PR 会更顺。

### "下一步你打算做什么？"
> 短期三件事按优先级：(1) **Weight-only INT4 / Marlin kernel**，这是现在模型规模上去 (7B+) 后最大的瓶颈；(2) **Speculative Decoding**，给 decode 吞吐再拉 1.5-2×；(3) **真正的混 batch chunked prefill**（prefill + decode 同 step），彻底解决 TTFT。

### "你的项目 vs 官方 vLLM 有什么差距？"
- **差在哪**：没有 Marlin/AWQ INT4、没有 speculative、没有多机 scheduler、PagedAttention 版本较旧、没覆盖 MoE / VLM。
- **可以怎么答**：很坦白——跟官方 vLLM 比不是这个项目的定位，它是一个"通过重写理解每层开销"的教学向项目。拿它对齐 nano-vllm (Python)，结果比 Python 快 16.5%，这个是合理可解释的。

---

## 面试前一天 checklist

- [ ] 本地跑一次 `bench_e2e` 和 `nano-vllm/bench.py`，截屏两个数字
- [ ] 白板能画 decoder 单层数据流：`hidden → rms → qkv_proj → [split+rmsnorm+rope+store] → attention → o_proj → +residual → rms → gate_up → silu·mul → down_proj → +residual`
- [ ] 能手推 KV cache 显存：`2 × L × H_kv × d × bsz × seqlen × sizeof(dtype)`
- [ ] 准备 3 个反问面试官的问题：团队推理/训练比例、使用的硬件代际、最近一个优化点是什么
- [ ] 简历、GitHub 链接、`docs/` 下 HTML 架构文档链接都准备好

---

## 反问面试官（3 选 1）

1. "团队里推理方向最近一年投入最多的优化方向是什么？是 kernel 级还是调度级？"
2. "你们对 speculative decoding / MoE 推理 / 多机推理这三块的 priority 大概怎么排？"
3. "日常的工作里，kernel 开发 vs 系统集成 vs serving 调优大概各占多少比例？"
