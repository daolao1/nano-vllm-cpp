# nano-vllm C++ 重写：架构文档 & 操作手册

> 本文档基于 Python 版 nano-vllm 逐行分析而成，作为 C++ 重写的完整蓝图。
> 所有数据结构、算法、控制流均从源码精确还原。

---

## 目录

- [一、项目总览](#一项目总览)
- [二、目录结构设计](#二目录结构设计)
- [三、依赖与构建系统](#三依赖与构建系统)
- [四、核心数据结构](#四核心数据结构)
- [五、模块详细设计](#五模块详细设计)
  - [5.1 Config 配置](#51-config-配置)
  - [5.2 SamplingParams 采样参数](#52-samplingparams-采样参数)
  - [5.3 Sequence 序列对象](#53-sequence-序列对象)
  - [5.4 BlockManager 页式 KV Cache 管理](#54-blockmanager-页式-kv-cache-管理)
  - [5.5 Scheduler 调度器](#55-scheduler-调度器)
  - [5.6 ModelRunner 模型执行器](#56-modelrunner-模型执行器)
  - [5.7 LLMEngine 引擎主类](#57-llmengine-引擎主类)
  - [5.8 模型层 Qwen3](#58-模型层-qwen3)
  - [5.9 算子层](#59-算子层)
  - [5.10 工具层](#510-工具层)
- [六、完整数据流](#六完整数据流)
- [七、实施步骤](#七实施步骤)
- [八、C++ 特有设计考量](#八c-特有设计考量)

---

## 一、项目总览

nano-vllm 是一个 ~1000 行的极简 LLM 推理引擎，实现了完整的 vLLM 核心功能：

| 特性 | 实现方式 |
|------|----------|
| PagedAttention | block_size=256, Triton kernel 写 KV Cache, flash_attn 读 |
| Prefix Caching | xxhash64 哈希链 + 内容双重验证 |
| Continuous Batching | Prefill 优先 + LIFO 抢占的 Scheduler |
| CUDA Graph | 预录制 bs=[1,2,4,8,16,...,512], 共享 memory pool |
| Tensor Parallelism | NCCL all_reduce/gather + SharedMemory 控制面 |
| Gumbel-max 采样 | probs / Exp(1) → argmax, torch.compile 加速 |

**支持模型**：Qwen3（架构上可扩展到任何 Transformer 解码器）

---

## 二、目录结构设计

```
nano-vllm-cpp/
├── CMakeLists.txt
├── include/
│   ├── config.h                    # Config 配置
│   ├── sampling_params.h           # SamplingParams
│   ├── engine/
│   │   ├── sequence.h              # Sequence 序列对象
│   │   ├── block_manager.h         # BlockManager 页管理
│   │   ├── scheduler.h             # Scheduler 调度器
│   │   ├── model_runner.h          # ModelRunner 模型执行
│   │   └── llm_engine.h            # LLMEngine 引擎主类
│   ├── layers/
│   │   ├── attention.h             # Attention + KV Cache 写入
│   │   ├── activation.h            # SiluAndMul
│   │   ├── layernorm.h             # RMSNorm
│   │   ├── linear.h                # 各种并行 Linear
│   │   ├── rotary_embedding.h      # RoPE
│   │   ├── embed_head.h            # VocabParallelEmbedding + ParallelLMHead
│   │   └── sampler.h               # Sampler
│   ├── models/
│   │   └── qwen3.h                 # Qwen3 模型定义
│   └── utils/
│       ├── context.h               # 全局 Context
│       └── loader.h                # SafeTensors 权重加载
├── src/                            # 对应 .cpp/.cu 实现
│   ├── engine/
│   ├── layers/
│   ├── models/
│   └── utils/
└── kernels/
    └── store_kvcache.cu            # KV Cache 写入 CUDA kernel
```

---

## 三、依赖与构建系统

### 必需依赖

| 库 | 用途 | C++ 替代/对接方式 |
|----|------|-------------------|
| **PyTorch (libtorch)** | Tensor 运算、CUDA 管理 | libtorch C++ API 或纯 CUDA |
| **flash-attn** | FlashAttention kernel | [flash-attn C API](https://github.com/Dao-AILab/flash-attention) 或 cuDNN fMHA |
| **Triton** | KV Cache 写入 kernel | 改写为 CUDA kernel（简单替代） |
| **NCCL** | 多卡通信 | 直接用 NCCL C API |
| **xxhash** | Prefix Cache 哈希 | [xxHash C 库](https://github.com/Cyan4973/xxHash)（header-only） |
| **safetensors** | 权重加载 | 自行解析（格式很简单，见下文） |
| **sentencepiece/tiktoken** | Tokenizer | [sentencepiece C++](https://github.com/google/sentencepiece) |
| **nlohmann/json** | HF config.json 解析 | nlohmann/json header-only |

### CMake 配置建议

```cmake
cmake_minimum_required(VERSION 3.18)
project(nano_vllm_cpp CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(Torch REQUIRED)        # libtorch
find_package(CUDA REQUIRED)
# 或者不用 libtorch，手写 Tensor + cuBLAS/cuBLASLt
```

**关键选择**：是否使用 libtorch？
- **用 libtorch**：开发速度快，Tensor API 几乎与 Python 一一对应；但引入大依赖
- **纯 CUDA**：更轻量，但要自己管理内存、GEMM（cuBLAS）、类型转换等

**推荐方案**：先用 libtorch 快速实现，后续可逐步替换为纯 CUDA。

---

## 四、核心数据结构

### 4.1 类依赖图

```
LLMEngine
  ├── Config
  ├── Tokenizer (sentencepiece)
  ├── Scheduler
  │     ├── BlockManager
  │     │     └── Block[]
  │     └── Sequence[]  (waiting/running 队列)
  └── ModelRunner
        ├── Model (Qwen3ForCausalLM)
        │     ├── Qwen3Model
        │     │     ├── VocabParallelEmbedding
        │     │     ├── Qwen3DecoderLayer[]
        │     │     │     ├── RMSNorm (input_layernorm)
        │     │     │     ├── Qwen3Attention
        │     │     │     │     ├── QKVParallelLinear
        │     │     │     │     ├── RMSNorm (q_norm, k_norm)
        │     │     │     │     ├── RotaryEmbedding
        │     │     │     │     ├── Attention (+ k_cache, v_cache 引用)
        │     │     │     │     └── RowParallelLinear (o_proj)
        │     │     │     ├── RMSNorm (post_attention_layernorm)
        │     │     │     └── Qwen3MLP
        │     │     │           ├── MergedColumnParallelLinear (gate_up_proj)
        │     │     │           ├── SiluAndMul
        │     │     │           └── RowParallelLinear (down_proj)
        │     │     └── RMSNorm (final norm)
        │     └── ParallelLMHead
        ├── Sampler
        ├── KV Cache Tensor [2, L, num_blocks, block_size, kv_heads, head_dim]
        └── CUDA Graphs (可选)
```

---

## 五、模块详细设计

### 5.1 Config 配置

```cpp
struct Config {
    std::string model;                      // HF 模型目录路径
    int max_num_batched_tokens = 16384;     // 每次 step 最大 token 数
    int max_num_seqs = 512;                 // 每批最大序列数
    int max_model_len = 4096;               // 最长序列长度
    float gpu_memory_utilization = 0.9f;    // GPU 显存利用率上限
    int tensor_parallel_size = 1;           // TP 并行度
    bool enforce_eager = false;             // true = 禁用 CUDA Graph
    int kvcache_block_size = 256;           // PagedAttention 页大小
    int num_kvcache_blocks = -1;            // 由 ModelRunner 计算
    int eos_token_id = -1;                  // 由 LLMEngine 赋值

    // 从 HF config.json 加载的模型参数
    struct HFConfig {
        int hidden_size;
        int num_attention_heads;
        int num_key_value_heads;
        int num_hidden_layers;
        int intermediate_size;
        int vocab_size;
        int max_position_embeddings;
        int head_dim;                       // 可选, = hidden_size / num_attention_heads
        float rms_norm_eps;
        float rope_theta;
        bool tie_word_embeddings;
        bool attention_bias;                // Qwen3 = false
        std::string hidden_act;             // "silu"
        std::string torch_dtype;            // "bfloat16"
    } hf_config;
};
```

**初始化校验**：
1. `model` 目录存在
2. `kvcache_block_size % 256 == 0`
3. `1 <= tensor_parallel_size <= 8`
4. 解析 `model/config.json` 填充 `hf_config`
5. `max_model_len = min(max_model_len, hf_config.max_position_embeddings)`
6. `max_num_batched_tokens >= max_model_len`

---

### 5.2 SamplingParams 采样参数

```cpp
struct SamplingParams {
    float temperature = 1.0f;       // 采样温度，必须 > 1e-10
    int max_tokens = 64;            // 最大生成 token 数
    bool ignore_eos = false;        // 是否忽略 EOS 终止
};
```

---

### 5.3 Sequence 序列对象

```cpp
enum class SequenceStatus { WAITING, RUNNING, FINISHED };

class Sequence {
public:
    static constexpr int BLOCK_SIZE = 256;

    // 构造函数
    Sequence(std::vector<int32_t> token_ids, SamplingParams params);

    // 核心字段
    int64_t seq_id;                          // 全局唯一 ID（原子递增）
    SequenceStatus status = SequenceStatus::WAITING;
    std::vector<int32_t> token_ids;          // 完整 token 序列（深拷贝）
    int32_t last_token;                      // 最新 token（decode 时只传这个）
    int32_t num_tokens;                      // = token_ids.size()
    int32_t num_prompt_tokens;               // prompt 长度（不变）
    int32_t num_cached_tokens = 0;           // prefix cache 命中的 token 数
    std::vector<int32_t> block_table;        // 物理 block ID 列表
    float temperature;
    int max_tokens;
    bool ignore_eos;

    // 属性方法
    int len() const { return num_tokens; }
    bool is_finished() const { return status == SequenceStatus::FINISHED; }
    int num_completion_tokens() const { return num_tokens - num_prompt_tokens; }
    int num_cached_blocks() const { return num_cached_tokens / BLOCK_SIZE; }
    int num_blocks() const { return (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE; }
    int last_block_num_tokens() const { return num_tokens - (num_blocks() - 1) * BLOCK_SIZE; }

    // 获取第 i 个 block 的 token_ids 切片
    std::vector<int32_t> block(int i) const {
        int start = i * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, num_tokens);
        return {token_ids.begin() + start, token_ids.begin() + end};
    }

    // 追加新 token
    void append_token(int32_t token_id) {
        token_ids.push_back(token_id);
        last_token = token_id;
        num_tokens++;
    }
};
```

**关键设计**：序列 ID 使用全局原子计数器 `static std::atomic<int64_t> counter`。

**序列化（多卡场景）**：跨进程传输 Sequence 时，decode 阶段只传 `last_token` 而不传完整 `token_ids`（优化带宽）。C++ 中可用简单的自定义序列化替代 pickle。

---

### 5.4 BlockManager 页式 KV Cache 管理

这是 PagedAttention 的核心。

#### Block 结构

```cpp
struct Block {
    int block_id;
    int ref_count = 0;              // 引用计数
    uint64_t hash = UINT64_MAX;     // xxhash64 值，UINT64_MAX = 未计算
    std::vector<int32_t> token_ids; // 该 block 存储的 token

    void update(uint64_t h, const std::vector<int32_t>& tids) {
        hash = h;
        token_ids = tids;
    }
    void reset() {
        ref_count = 1;
        hash = UINT64_MAX;
        token_ids.clear();
    }
};
```

#### BlockManager 类

```cpp
class BlockManager {
public:
    BlockManager(int num_blocks, int block_size);

    // 哈希计算（核心：哈希链）
    static uint64_t compute_hash(const std::vector<int32_t>& token_ids,
                                  uint64_t prefix = UINT64_MAX);

    // 查询能否为序列分配 blocks
    bool can_allocate(const Sequence& seq) const;

    // 为序列分配 blocks（含 prefix cache 查找）
    void allocate(Sequence& seq);

    // 释放序列的所有 blocks
    void deallocate(Sequence& seq);

    // 查询 decode 后能否追加 block
    bool can_append(const Sequence& seq) const;

    // decode 后维护 block（可能分配新 block 或注册 hash）
    void may_append(Sequence& seq);

private:
    int block_size_;
    std::vector<Block> blocks_;
    std::unordered_map<uint64_t, int> hash_to_block_id_;  // hash → block_id
    std::deque<int> free_block_ids_;
    std::unordered_set<int> used_block_ids_;

    void allocate_block(int block_id);
    void deallocate_block(int block_id);
};
```

#### 关键算法：compute_hash（哈希链）

```
compute_hash(token_ids, prefix):
    h = xxhash64()
    if prefix != UINT64_MAX:
        h.update(prefix 的 8 字节小端表示)
    h.update(token_ids 数组的字节表示)
    return h.intdigest()
```

**设计**：每个完整 block（256 tokens）的 hash = xxh64(前一个 block 的 hash + 当前 block 的 token_ids)。这形成哈希链，使得相同前缀的不同请求可以复用 KV Cache。

#### 关键算法：allocate（前缀缓存分配）

```
allocate(seq):
    assert seq.block_table 为空
    h = UINT64_MAX
    cache_miss = false
    for i in 0..seq.num_blocks-1:
        token_ids = seq.block(i)
        if len(token_ids) == BLOCK_SIZE:
            h = compute_hash(token_ids, h)
        else:
            h = UINT64_MAX    // 最后一个不完整 block 不计算 hash

        block_id = hash_to_block_id[h]  // 查找缓存
        if block_id 不存在 或 blocks[block_id].token_ids != token_ids:
            cache_miss = true

        if cache_miss:
            // 分配新 block
            block_id = free_block_ids.front()
            allocate_block(block_id)
        else:
            // cache hit！
            seq.num_cached_tokens += BLOCK_SIZE
            if block_id 在 used_block_ids 中:
                blocks[block_id].ref_count++
            else:
                allocate_block(block_id)  // 被 evict 后又需要

        if h != UINT64_MAX:
            blocks[block_id].update(h, token_ids)
            hash_to_block_id[h] = block_id

        seq.block_table.push_back(block_id)
```

关键点：一旦发生 cache_miss，后续所有 block 都标记为 miss（不会跳过"空洞"复用）。

#### 关键算法：deallocate

```
deallocate(seq):
    for block_id in reversed(seq.block_table):
        blocks[block_id].ref_count--
        if ref_count == 0:
            deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

#### 关键算法：may_append（decode 后维护）

```
may_append(seq):
    last_block = blocks[block_table.back()]
    if len(seq) % BLOCK_SIZE == 1:
        // 新 token 溢出到新 block，旧 block 已满（hash 已注册）
        allocate 新 block, append 到 block_table
    elif len(seq) % BLOCK_SIZE == 0:
        // 最后一个 block 刚好填满，注册 hash
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = block_table.size() > 1 ? blocks[block_table[-2]].hash : UINT64_MAX
        h = compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        hash_to_block_id[h] = last_block.block_id
    else:
        // 未满，什么都不做
```

#### can_append 的判断

```
can_append(seq):
    // 仅当 len(seq) % BLOCK_SIZE == 1 时需要 1 个新 block
    need = (seq.len() % BLOCK_SIZE == 1) ? 1 : 0
    return free_block_ids.size() >= need
```

---

### 5.5 Scheduler 调度器

```cpp
class Scheduler {
public:
    Scheduler(const Config& config);

    void add(Sequence seq);
    bool is_finished() const;

    // 返回 (调度的序列列表, 是否为 prefill)
    std::pair<std::vector<Sequence*>, bool> schedule();

    // 后处理：追加 token、检查终止、回收 block
    void postprocess(const std::vector<Sequence*>& seqs,
                     const std::vector<int32_t>& token_ids);

private:
    void preempt(Sequence* seq);

    int max_num_seqs_;
    int max_num_batched_tokens_;
    int eos_;
    BlockManager block_manager_;
    std::deque<Sequence> waiting_;
    std::deque<Sequence*> running_;  // 指向存活的 Sequence
};
```

#### 核心算法：schedule()

```
schedule():
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0

    // ===== 阶段 1：Prefill =====
    while waiting 非空 && num_seqs < max_num_seqs:
        seq = waiting.front()
        if num_batched_tokens + seq.len() > max_num_batched_tokens:
            break
        if !block_manager.can_allocate(seq):
            break
        num_seqs++
        block_manager.allocate(seq)
        num_batched_tokens += seq.len() - seq.num_cached_tokens  // 减去 prefix cache
        seq.status = RUNNING
        waiting.pop_front()
        running.push_back(seq)
        scheduled_seqs.push_back(seq)

    if scheduled_seqs 非空:
        return (scheduled_seqs, true)  // prefill

    // ===== 阶段 2：Decode =====
    while running 非空 && num_seqs < max_num_seqs:
        seq = running.pop_front()
        while !block_manager.can_append(seq):
            if running 非空:
                preempt(running.pop_back())  // LIFO 抢占：弹出最晚加入的
            else:
                preempt(seq)                 // 连自己都放不下
                break
        else:
            num_seqs++
            block_manager.may_append(seq)
            scheduled_seqs.push_back(seq)

    assert scheduled_seqs 非空
    running.insert_front(reversed(scheduled_seqs))
    return (scheduled_seqs, false)  // decode
```

**关键设计**：
1. **Prefill 优先级高于 Decode**：有新请求就先 prefill
2. **Decode 的 LIFO 抢占**：显存不足时，弹出 running 队尾（最晚加入的）序列
3. `num_batched_tokens` 中扣除了 prefix cache 命中的 token 数

#### postprocess()

```
postprocess(seqs, token_ids):
    for (seq, token_id) in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (!seq.ignore_eos && token_id == eos) || seq.num_completion_tokens() == seq.max_tokens:
            seq.status = FINISHED
            block_manager.deallocate(seq)
            running.remove(seq)
```

---

### 5.6 ModelRunner 模型执行器

这是最复杂的模块，负责准备输入、运行模型、管理 KV Cache 和 CUDA Graph。

```cpp
class ModelRunner {
public:
    ModelRunner(const Config& config, int rank, /* 通信机制 */);

    // 核心接口
    std::vector<int32_t> run(const std::vector<Sequence*>& seqs, bool is_prefill);
    void exit();

private:
    // 初始化流程
    void warmup_model();
    void allocate_kv_cache();
    void capture_cudagraph();

    // 输入准备
    void prepare_prefill(const std::vector<Sequence*>& seqs,
                         Tensor& input_ids, Tensor& positions);
    void prepare_decode(const std::vector<Sequence*>& seqs,
                        Tensor& input_ids, Tensor& positions);
    Tensor prepare_sample(const std::vector<Sequence*>& seqs);

    // 模型执行
    Tensor run_model(const Tensor& input_ids, const Tensor& positions, bool is_prefill);

    // 多卡通信
    void write_shm(const std::string& method, /* args */);
    void read_shm(/* out args */);
    void loop();  // worker rank 的事件循环
    void call(const std::string& method, /* args */);

    Config config_;
    int rank_;
    int world_size_;
    Qwen3ForCausalLM model_;
    Sampler sampler_;
    Tensor kv_cache_;  // [2, L, num_blocks, block_size, kv_heads, head_dim]

    // CUDA Graph 相关
    bool enforce_eager_;
    std::vector<int> graph_bs_;                // [1,2,4,8,16,32,...,512]
    std::unordered_map<int, CUDAGraph> graphs_;
    Tensor graph_input_ids_, graph_positions_;  // graph 捕获时的输入 tensor
    Tensor graph_slot_mapping_, graph_context_lens_, graph_block_tables_;
    Tensor graph_outputs_;
};
```

#### 5.6.1 初始化流程

```
ModelRunner.__init__(config, rank, events):
    1. NCCL init_process_group("nccl", "tcp://localhost:2333", world_size, rank)
    2. cuda.set_device(rank)
    3. 设置默认 dtype = hf_config.torch_dtype (bfloat16)
    4. 创建 Qwen3ForCausalLM 模型（在 GPU 上）
    5. load_model(model, config.model)  // 加载 SafeTensors 权重
    6. 创建 Sampler
    7. warmup_model()
    8. allocate_kv_cache()
    9. if !enforce_eager: capture_cudagraph()
    10. 若多卡：rank 0 创建 SharedMemory，rank>0 attach 后进入 loop()
```

#### 5.6.2 warmup_model()

```
warmup_model():
    clear CUDA cache, reset peak memory stats
    num_seqs = min(max_num_batched_tokens / max_model_len, max_num_seqs)
    创建 num_seqs 个全零序列，长度 = max_model_len
    调用 run(seqs, true)  // 一次完整 prefill，触发所有内存分配
    clear CUDA cache      // 只为测量 peak memory
```

#### 5.6.3 allocate_kv_cache()

```
allocate_kv_cache():
    (free, total) = cuda.mem_get_info()
    used = total - free
    peak = cuda.peak_memory
    current = cuda.current_memory

    kv_heads_per_rank = num_kv_heads / world_size
    head_dim = hf_config.head_dim 或 hidden_size / num_attention_heads
    block_bytes = 2(K+V) * num_layers * block_size * kv_heads_per_rank * head_dim * dtype_bytes

    num_blocks = int(total * gpu_memory_utilization - used - peak + current) / block_bytes
    assert num_blocks > 0

    kv_cache = Tensor([2, num_layers, num_blocks, block_size, kv_heads_per_rank, head_dim])

    // 遍历模型所有 Attention 层，将 k_cache/v_cache 指向 kv_cache 的对应切片
    layer_id = 0
    for module in model.modules():
        if hasattr(module, k_cache) && hasattr(module, v_cache):
            module.k_cache = kv_cache[0, layer_id]  // [num_blocks, block_size, kv_heads, head_dim]
            module.v_cache = kv_cache[1, layer_id]
            layer_id++
```

#### 5.6.4 prepare_prefill()

```
prepare_prefill(seqs) -> (input_ids, positions):
    // 收集所有序列的输入（跳过 prefix cache 部分）
    input_ids = []
    positions = []
    cu_seqlens_q = [0]      // query 累积长度
    cu_seqlens_k = [0]      // key 累积长度（含 cached）
    slot_mapping = []
    block_tables = null

    for seq in seqs:
        seqlen = seq.len()
        // 只取 num_cached_tokens 之后的 token（跳过已缓存的）
        input_ids.extend(seq.token_ids[seq.num_cached_tokens : seqlen])
        positions.extend(range(seq.num_cached_tokens, seqlen))

        seqlen_q = seqlen - seq.num_cached_tokens  // 实际需要计算的 query 长度
        seqlen_k = seqlen                           // 完整 key 长度（含 cached）
        cu_seqlens_q.append(cu_seqlens_q.back() + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k.back() + seqlen_k)

        // 计算物理 slot mapping（从 cached blocks 之后开始）
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            block_id = seq.block_table[i]
            start = block_id * block_size
            if i != seq.num_blocks - 1:
                end = start + block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(range(start, end))

    // 如果 cu_seqlens_k > cu_seqlens_q（有 prefix cache），需要 block_tables
    if cu_seqlens_k.back() > cu_seqlens_q.back():
        block_tables = prepare_block_tables(seqs)  // [num_seqs, max_num_blocks]

    // 所有 tensor：pin_memory + async cuda transfer
    set_context(is_prefill=true, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, slot_mapping, null, block_tables)
    return (input_ids, positions)
```

**Tensor 形状（Prefill）**：
| Tensor | Shape | 说明 |
|--------|-------|------|
| input_ids | [sum_query_lens] | 1D varlen 格式，所有序列拼接 |
| positions | [sum_query_lens] | 对应位置编码 |
| cu_seqlens_q | [num_seqs + 1] | int32, query 累积长度 |
| cu_seqlens_k | [num_seqs + 1] | int32, key 累积长度 |
| slot_mapping | [num_new_tokens] | 非 cached token 对应的物理 slot |
| block_tables | [num_seqs, max_blocks] | int32, 仅 prefix cache 时存在 |

#### 5.6.5 prepare_decode()

```
prepare_decode(seqs) -> (input_ids, positions):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    for seq in seqs:
        input_ids.append(seq.last_token)                    // 只取最新 token
        positions.append(seq.len() - 1)                      // 位置
        context_lens.append(seq.len())                       // 已有 token 总数
        // 最新 token 对应的物理 slot
        slot_mapping.append(
            seq.block_table.back() * block_size + seq.last_block_num_tokens - 1
        )

    block_tables = prepare_block_tables(seqs)  // [batch_size, max_num_blocks]
    set_context(is_prefill=false, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)
    return (input_ids, positions)
```

**Tensor 形状（Decode）**：
| Tensor | Shape | 说明 |
|--------|-------|------|
| input_ids | [batch_size] | 每个序列 1 个 token |
| positions | [batch_size] | |
| slot_mapping | [batch_size] | |
| context_lens | [batch_size] | |
| block_tables | [batch_size, max_blocks] | |

#### 5.6.6 run_model()

```
run_model(input_ids, positions, is_prefill):
    if is_prefill 或 enforce_eager 或 batch_size > 512:
        // 直接 eager 执行
        hidden = model.forward(input_ids, positions)
        return model.compute_logits(hidden)
    else:
        // CUDA Graph 执行
        bs = input_ids.size(0)
        // 找最小的 >= bs 的预录制 graph
        graph_bs = next(x for x in [1,2,4,8,16,...] if x >= bs)
        graph = graphs[graph_bs]

        // 填充 graph_vars（注意：要保持 tensor 地址不变，只修改内容）
        graph_vars.input_ids[:bs] = input_ids
        graph_vars.positions[:bs] = positions
        graph_vars.slot_mapping.fill(-1)          // -1 = 不写 cache
        graph_vars.slot_mapping[:bs] = context.slot_mapping
        graph_vars.context_lens.zero()
        graph_vars.context_lens[:bs] = context.context_lens
        graph_vars.block_tables[:bs, :actual_cols] = context.block_tables

        graph.replay()

        return model.compute_logits(graph_vars.outputs[:bs])
```

#### 5.6.7 capture_cudagraph()

```
capture_cudagraph():
    max_bs = min(max_num_seqs, 512)
    max_num_blocks = ceil(max_model_len / block_size)

    // 所有 graph 共享的 tensor（地址固定）
    input_ids = zeros(max_bs, int64)
    positions = zeros(max_bs, int64)
    slot_mapping = zeros(max_bs, int32)
    context_lens = zeros(max_bs, int32)
    block_tables = zeros(max_bs, max_num_blocks, int32)
    outputs = zeros(max_bs, hidden_size, model_dtype)

    graph_bs = [1, 2, 4, 8, 16, 32, ..., max_bs]
    graph_pool = null

    // 从最大 bs 到最小 bs 逆序 capture（共享 memory pool）
    for bs in reversed(graph_bs):
        graph = CUDAGraph()
        set_context(decode, slot_mapping[:bs], context_lens[:bs], block_tables[:bs])

        outputs[:bs] = model(input_ids[:bs], positions[:bs])    // warmup

        with cuda_graph_capture(graph, graph_pool):
            outputs[:bs] = model(input_ids[:bs], positions[:bs])  // capture

        if graph_pool == null:
            graph_pool = graph.pool()
        graphs[bs] = graph
        cuda_synchronize()
        reset_context()

    保存 graph_vars = {input_ids, positions, slot_mapping, context_lens, block_tables, outputs}
```

**关键设计**：
- 从大到小 capture，因为最大 bs 分配的 pool 可以容纳小 bs
- `slot_mapping` 填 -1 表示不写 KV cache（kernel 中检查 -1 跳过）
- 所有 graph 共享 tensor 地址，replay 时只修改内容

#### 5.6.8 run()

```
run(seqs, is_prefill) -> list[int]:
    input_ids, positions = is_prefill ? prepare_prefill(seqs) : prepare_decode(seqs)
    temperatures = (rank == 0) ? prepare_sample(seqs) : null
    logits = run_model(input_ids, positions, is_prefill)
    token_ids = (rank == 0) ? sampler(logits, temperatures).tolist() : null
    reset_context()
    return token_ids
```

#### 5.6.9 多卡通信协议

```
// Rank 0 广播：
write_shm(method_name, args...):
    data = serialize([method_name, args...])
    shm.buf[0:4] = len(data) 小端
    shm.buf[4:4+len] = data
    for event in events: event.set()

// Rank >0 接收：
read_shm():
    event.wait()
    n = shm.buf[0:4] 小端解读
    [method_name, args...] = deserialize(shm.buf[4:4+n])
    event.clear()
    return (method_name, args)

// Rank >0 事件循环：
loop():
    while true:
        (method, args) = read_shm()
        call(method, args...)
        if method == "exit": break

// 统一调用接口（rank 0 负责调度）：
call(method, args...):
    if world_size > 1 && rank == 0:
        write_shm(method, args...)    // 广播给 workers
    self.method(args...)              // 自己也执行
```

C++ 实现建议：用 POSIX 共享内存 + eventfd/条件变量替代 Python 的 SharedMemory + Event。

---

### 5.7 LLMEngine 引擎主类

```cpp
class LLMEngine {
public:
    // 构造函数
    LLMEngine(const std::string& model, /* kwargs */);
    ~LLMEngine();  // atexit 注册 exit

    // 添加请求
    void add_request(const std::string& prompt, const SamplingParams& params);
    void add_request(const std::vector<int32_t>& token_ids, const SamplingParams& params);

    // 单步执行
    // 返回 (已完成的序列输出, 处理的 token 数)
    std::pair<std::vector<Output>, int> step();

    bool is_finished() const;

    // 批量生成
    std::vector<Output> generate(const std::vector<std::string>& prompts,
                                  const SamplingParams& params);

    struct Output {
        std::string text;
        std::vector<int32_t> token_ids;
    };

private:
    Config config_;
    Tokenizer tokenizer_;
    Scheduler scheduler_;
    ModelRunner model_runner_;
    std::vector<std::thread> worker_threads_;  // TP workers
};
```

#### generate() 主循环

```
generate(prompts, sampling_params):
    // 1. 编码并添加请求
    for (prompt, sp) in zip(prompts, sampling_params):
        token_ids = tokenizer.encode(prompt)
        add_request(token_ids, sp)

    // 2. 循环直到完成
    outputs = {}
    while !is_finished():
        (output, num_tokens) = step()
        // output 包含已完成序列的 (seq_id, completion_token_ids)
        for (seq_id, token_ids) in output:
            outputs[seq_id] = token_ids

    // 3. 按 seq_id 排序（保证输出顺序 = 输入顺序）
    sorted_outputs = sort_by_seq_id(outputs)

    // 4. decode 输出文本
    return [{ text: tokenizer.decode(ids), token_ids: ids } for ids in sorted_outputs]
```

#### step()

```
step():
    (seqs, is_prefill) = scheduler.schedule()
    token_ids = model_runner.call("run", seqs, is_prefill)
    scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = is_prefill ? sum(seq.len() for seq in seqs) : -len(seqs)
    return (outputs, num_tokens)
```

---

### 5.8 模型层 Qwen3

#### 类层级

```
Qwen3ForCausalLM
  ├── Qwen3Model (model)
  │     ├── VocabParallelEmbedding (embed_tokens)
  │     ├── Qwen3DecoderLayer[] (layers)
  │     └── RMSNorm (norm)
  └── ParallelLMHead (lm_head)
```

#### Qwen3Attention

```
Qwen3Attention:
    fields:
        qkv_proj: QKVParallelLinear(hidden_size → q_size + 2*kv_size)
        o_proj:   RowParallelLinear(num_heads * head_dim → hidden_size)
        rotary_emb: RotaryEmbedding
        attn:     Attention
        q_norm:   RMSNorm(head_dim)    // 仅 qkv_bias=false (Qwen3)
        k_norm:   RMSNorm(head_dim)    // 仅 qkv_bias=false (Qwen3)

    forward(positions, hidden_states):
        qkv = qkv_proj(hidden_states)    // [N, q_size + 2*kv_size]
        q, k, v = split(qkv, [q_size, kv_size, kv_size], dim=-1)
        q = q.reshape(-1, num_heads, head_dim)
        k = k.reshape(-1, num_kv_heads, head_dim)
        v = v.reshape(-1, num_kv_heads, head_dim)
        if !qkv_bias:
            q = q_norm(q)     // per-head RMSNorm
            k = k_norm(k)
        q, k = rotary_emb(positions, q, k)
        o = attn(q, k, v)    // FlashAttention + PagedAttention
        output = o_proj(o.flatten(1, -1))
        return output
```

#### Qwen3MLP

```
Qwen3MLP:
    gate_up_proj: MergedColumnParallelLinear(hidden → [intermediate, intermediate])
    down_proj:    RowParallelLinear(intermediate → hidden)
    act_fn:       SiluAndMul

    forward(x):
        gate_up = gate_up_proj(x)
        x = act_fn(gate_up)        // silu(gate) * up
        x = down_proj(x)
        return x
```

#### Qwen3DecoderLayer

```
Qwen3DecoderLayer:
    input_layernorm:          RMSNorm
    self_attn:                Qwen3Attention
    post_attention_layernorm: RMSNorm
    mlp:                      Qwen3MLP

    forward(positions, hidden_states, residual):
        // Pre-Norm 残差连接
        if residual == null:
            // 第一层
            hidden_states, residual = input_layernorm(hidden_states), hidden_states
        else:
            // 后续层：融合 Add + RMSNorm
            hidden_states, residual = input_layernorm(hidden_states, residual)
        hidden_states = self_attn(positions, hidden_states)
        hidden_states, residual = post_attention_layernorm(hidden_states, residual)
        hidden_states = mlp(hidden_states)
        return (hidden_states, residual)
```

**关键优化**：`add_rms_forward` 将残差加法和 RMSNorm 融合为一次操作。

#### Qwen3Model

```
Qwen3Model:
    forward(input_ids, positions):
        hidden_states = embed_tokens(input_ids)
        residual = null
        for layer in layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = norm(hidden_states, residual)  // final add+norm
        return hidden_states
```

#### Qwen3ForCausalLM

```
Qwen3ForCausalLM:
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    __init__(config):
        model = Qwen3Model(config)
        lm_head = ParallelLMHead(vocab_size, hidden_size)
        if tie_word_embeddings:
            lm_head.weight = model.embed_tokens.weight  // 权重共享

    forward(input_ids, positions):
        return model(input_ids, positions)

    compute_logits(hidden_states):
        return lm_head(hidden_states)
```

---

### 5.9 算子层

#### 5.9.1 Attention + KV Cache 写入

**KV Cache 写入 CUDA Kernel**（替代原 Triton kernel）：

```cuda
// 每个线程处理一个 token 的 KV 写入
__global__ void store_kvcache_kernel(
    const float* key,           // [N, num_heads * head_dim]
    int key_stride,             // key 的行 stride
    const float* value,         // [N, num_heads * head_dim]
    int value_stride,
    float* k_cache,             // [num_blocks * block_size, num_heads * head_dim] (展平)
    float* v_cache,
    const int* slot_mapping,    // [N], 每个 token 对应的物理 slot
    int D                       // = num_heads * head_dim
) {
    int idx = blockIdx.x;       // token index
    int slot = slot_mapping[idx];
    if (slot == -1) return;     // -1 表示不写（CUDA Graph padding 用）

    // 按 D 元素复制
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        k_cache[slot * D + d] = key[idx * key_stride + d];
        v_cache[slot * D + d] = value[idx * value_stride + d];
    }
}
```

**Attention 前向逻辑**：

```
Attention.forward(q, k, v):
    context = get_context()  // 全局 Context 对象

    // 1. 写入 KV Cache
    if k_cache 非空:
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    // 2. 注意力计算
    if context.is_prefill:
        if context.block_tables != null:
            // 有 prefix cache：用 paged KV cache
            o = flash_attn_varlen(q, k_cache, v_cache,
                                   cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k,
                                   scale, causal=true, block_table)
        else:
            // 无 prefix cache：直接用输入 k,v
            o = flash_attn_varlen(q, k, v,
                                   cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k,
                                   scale, causal=true)
    else:
        // Decode: 每个序列只有 1 个 query token
        o = flash_attn_with_kvcache(q.unsqueeze(1),     // [bs, 1, heads, head_dim]
                                     k_cache, v_cache,
                                     cache_seqlens=context.context_lens,
                                     block_table=context.block_tables,
                                     scale, causal=true)
    return o
```

**C++ 实现选择**：
- 推荐使用 FlashAttention 的 C API（在其仓库的 `csrc/` 目录下）
- 或使用 cuDNN 的 fused Multi-Head Attention API
- 如果不想依赖外部库，可以先实现朴素版（但性能差很多）

#### 5.9.2 RMSNorm

```
RMSNorm:
    weight: [hidden_size]   // 可学习参数，初始为全 1
    eps: float = 1e-6

    // 标准 RMSNorm
    rms_forward(x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=true)
        x *= rsqrt(var + eps)
        return x.to(orig_dtype) * weight

    // 融合 Add + RMSNorm（关键优化）
    add_rms_forward(x, residual):
        orig_dtype = x.dtype
        x = x.float() + residual.float()   // 融合 add
        residual = x.to(orig_dtype)         // 更新 residual
        var = x.pow(2).mean(dim=-1, keepdim=true)
        x *= rsqrt(var + eps)
        return (x.to(orig_dtype) * weight, residual)

    forward(x, residual=null):
        if residual == null: return rms_forward(x)
        else: return add_rms_forward(x, residual)
```

#### 5.9.3 RoPE (Rotary Position Embedding)

```
RotaryEmbedding.__init__(head_size, rotary_dim, max_position, base):
    assert rotary_dim == head_size
    inv_freq = 1.0 / (base ^ (arange(0, rotary_dim, 2) / rotary_dim))
    t = arange(0, max_position)
    freqs = einsum("i,j -> ij", t, inv_freq)      // [max_pos, rotary_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    cos_sin_cache = cat(cos, sin, dim=-1).unsqueeze(1)  // [max_pos, 1, rotary_dim]

apply_rotary_emb(x, cos, sin):
    x1, x2 = chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return cat(y1, y2, dim=-1).to(x.dtype)

RotaryEmbedding.forward(positions, query, key):
    cos_sin = cos_sin_cache[positions]              // 按位置索引
    cos, sin = chunk(cos_sin, 2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return (query, key)
```

#### 5.9.4 激活函数

```
SiluAndMul.forward(x):
    gate, up = chunk(x, 2, dim=-1)
    return silu(gate) * up
    // silu(x) = x * sigmoid(x)
```

#### 5.9.5 Sampler（Gumbel-max trick）

```
Sampler.forward(logits, temperatures):
    // 1. 温度缩放
    logits = logits.float() / temperatures.unsqueeze(1)
    // 2. Softmax
    probs = softmax(logits, dim=-1)
    // 3. Gumbel-max 采样（等价于 multinomial 采样，但更高效）
    exp_samples = exponential_distribution(1).clamp_min(1e-10)  // 每个位置独立采样
    scores = probs / exp_samples
    return argmax(scores, dim=-1)
```

**Gumbel-max trick 原理**：从分类分布 `Cat(p1,...,pk)` 采样等价于 `argmax(log(pi) + Gi)` 其中 `Gi ~ Gumbel(0,1)`，而 `Gumbel = -log(Exp(1))`。实现中用 `probs / Exp(1)` 的 argmax 达到同样效果。

#### 5.9.6 张量并行 Linear 层

```
// 列并行：output 维切分
ColumnParallelLinear(input_size, output_size):
    local_output_size = output_size / tp_size
    weight: [local_output_size, input_size]
    forward(x): F.linear(x, weight, bias)
    // 无 all_reduce，通常与 RowParallel 配对

// 行并行：input 维切分
RowParallelLinear(input_size, output_size):
    local_input_size = input_size / tp_size
    weight: [output_size, local_input_size]
    forward(x): y = F.linear(x, weight, bias_if_rank0); all_reduce(y)

// QKV 合并列并行
QKVParallelLinear(hidden, head_size, num_heads, num_kv_heads):
    local_q_size = (num_heads / tp_size) * head_size
    local_kv_size = (num_kv_heads / tp_size) * head_size
    output_size = local_q_size + 2 * local_kv_size
    weight: [output_size, hidden]
    // 权重加载时按 q/k/v shard 填充对应区域

// Gate+Up 合并列并行
MergedColumnParallelLinear(input_size, output_sizes=[intermediate, intermediate]):
    local_output = sum(output_sizes) / tp_size
    weight: [local_output, input_size]
    // 权重加载时按 shard_id (0=gate, 1=up) 填充对应区域
```

#### 5.9.7 Embedding & LM Head

```
VocabParallelEmbedding(vocab_size, hidden_dim):
    local_vocab = vocab_size / tp_size
    vocab_start = local_vocab * rank
    vocab_end = vocab_start + local_vocab
    weight: [local_vocab, hidden_dim]

    forward(x):
        if tp_size > 1:
            mask = (x >= vocab_start) & (x < vocab_end)
            x = mask * (x - vocab_start)     // 映射到本地 offset
        y = embedding(x, weight)
        if tp_size > 1:
            y = mask.unsqueeze(1) * y         // mask 掉非本 rank
            all_reduce(y)
        return y

ParallelLMHead(vocab_size, hidden_dim): 继承 VocabParallelEmbedding
    forward(x):
        if is_prefill:
            // 只取每个序列最后一个 token 的 hidden state
            last_indices = cu_seqlens_q[1:] - 1
            x = x[last_indices]
        logits = linear(x, weight)    // [bs, local_vocab]
        if tp_size > 1:
            // rank 0 gather 所有 rank 的 logits，沿 vocab 维拼接
            all_logits = gather(logits, rank=0)
            logits = cat(all_logits, dim=-1)  // [bs, full_vocab]
        return logits
```

---

### 5.10 工具层

#### 5.10.1 Context（全局注意力上下文）

```cpp
struct Context {
    bool is_prefill = false;
    Tensor cu_seqlens_q;     // [num_seqs + 1], int32
    Tensor cu_seqlens_k;     // [num_seqs + 1], int32
    int max_seqlen_q = 0;
    int max_seqlen_k = 0;
    Tensor slot_mapping;     // [N], int32
    Tensor context_lens;     // [batch_size], int32 (decode only)
    Tensor block_tables;     // [batch_size, max_blocks], int32
};

// 全局单例
Context& get_context();
void set_context(bool is_prefill, ...);
void reset_context();
```

**设计目的**：避免在每个 forward 签名中传递大量注意力元数据。Attention 层直接从全局 Context 读取。

C++ 中可用 thread_local 全局变量（单线程推理）或传引用（如果多线程）。

#### 5.10.2 SafeTensors 权重加载

**SafeTensors 文件格式**：
```
[8 字节 header_size (uint64 小端)]
[header_size 字节 JSON header]
[对齐的二进制 tensor 数据]
```

JSON header 示例：
```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [896, 896],
    "data_offsets": [0, 1605632]
  },
  ...
}
```

**权重加载算法**：

```
load_model(model, path):
    packed_modules_mapping = model.packed_modules_mapping
    // 例如: {"q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), ...}

    for file in glob(path + "/*.safetensors"):
        for weight_name in safetensors_keys(file):
            // 检查是否需要 packed 加载
            for (key, (target, shard_id)) in packed_modules_mapping:
                if key in weight_name:
                    param_name = weight_name.replace(key, target)
                    param = model.get_parameter(param_name)
                    param.weight_loader(param, tensor, shard_id)
                    break
            else:
                // 直接加载
                param = model.get_parameter(weight_name)
                param.weight_loader(param, tensor)
```

**packed_modules_mapping 的作用**：
HF 模型的 SafeTensors 中，Q/K/V 是独立的权重。但 nano-vllm 将它们合并成 `qkv_proj` 一个权重（性能更好）。加载时需要将 HF 的 `q_proj.weight` 填入 `qkv_proj.weight` 的对应区域。

---

## 六、完整数据流

```
用户调用 generate(["你好", "介绍一下自己"], SamplingParams(temperature=0.6))
│
├── tokenizer.encode("你好") → [151644, 108386, 151645, ...]
├── tokenizer.encode("介绍一下自己") → [151644, 111308, 99392, ...]
│
├── add_request → Sequence(token_ids, params) → scheduler.waiting
│
└── while !is_finished():
      step():
      │
      │  ┌── scheduler.schedule() ──────────────────────────────────────┐
      │  │                                                              │
      │  │  首次：waiting 有 2 个序列 → Prefill                          │
      │  │  allocate blocks (prefix cache 查找)                         │
      │  │  返回 ([seq1, seq2], is_prefill=true)                        │
      │  └──────────────────────────────────────────────────────────────┘
      │
      │  ┌── model_runner.run(seqs, is_prefill=true) ──────────────────┐
      │  │                                                              │
      │  │  prepare_prefill:                                            │
      │  │    input_ids = [所有 prompt tokens 拼接]                      │
      │  │    positions = [0,1,2,...,len1-1, 0,1,...,len2-1]            │
      │  │    cu_seqlens_q = [0, len1, len1+len2]                      │
      │  │    slot_mapping = [物理 slot 0,1,2,...]                      │
      │  │    set_context(...)                                          │
      │  │                                                              │
      │  │  model.forward(input_ids, positions):                        │
      │  │    embed_tokens(input_ids)           → [N, hidden_size]      │
      │  │    for layer in layers:                                      │
      │  │      RMSNorm + Attention:                                    │
      │  │        qkv_proj → q,k,v                                     │
      │  │        q_norm, k_norm (Qwen3 特有)                           │
      │  │        RoPE(positions, q, k)                                 │
      │  │        store_kvcache(k, v → k_cache, v_cache)               │
      │  │        flash_attn_varlen(q, k, v, cu_seqlens...) → o        │
      │  │        o_proj(o)                                             │
      │  │      RMSNorm + MLP:                                          │
      │  │        gate_up_proj → silu_and_mul → down_proj               │
      │  │    final RMSNorm                                             │
      │  │                                                              │
      │  │  compute_logits:                                             │
      │  │    取每个序列最后一个 hidden → lm_head → logits              │
      │  │                                                              │
      │  │  sampler(logits, temperatures):                              │
      │  │    logits / temperature → softmax → Gumbel-max → token_ids   │
      │  │                                                              │
      │  │  返回 [next_token_1, next_token_2]                           │
      │  └──────────────────────────────────────────────────────────────┘
      │
      │  ┌── scheduler.postprocess ────────────────────────────────────┐
      │  │  seq1.append_token(next_token_1)                             │
      │  │  seq2.append_token(next_token_2)                             │
      │  │  检查 EOS / max_tokens → 未结束 → 继续                       │
      │  └──────────────────────────────────────────────────────────────┘
      │
      │  后续 step()：waiting 空 → Decode 模式
      │
      │  ┌── scheduler.schedule() ──────────────────────────────────────┐
      │  │  running 有 [seq1, seq2] → Decode                            │
      │  │  block_manager.can_append → may_append                       │
      │  │  返回 ([seq1, seq2], is_prefill=false)                       │
      │  └──────────────────────────────────────────────────────────────┘
      │
      │  ┌── model_runner.run(seqs, is_prefill=false) ─────────────────┐
      │  │  prepare_decode:                                             │
      │  │    input_ids = [last_token_1, last_token_2]                  │
      │  │    positions = [pos1, pos2]                                  │
      │  │    slot_mapping, context_lens, block_tables                  │
      │  │                                                              │
      │  │  CUDA Graph replay (if enabled):                             │
      │  │    填充 graph_vars → graph.replay()                          │
      │  │                                                              │
      │  │  或 eager forward:                                           │
      │  │    model.forward → compute_logits → sampler                  │
      │  │                                                              │
      │  │  返回 [next_token_1, next_token_2]                           │
      │  └──────────────────────────────────────────────────────────────┘
      │
      │  postprocess → append → 检查终止
      │
      └── seq1.is_finished && seq2.is_finished → 退出循环

tokenizer.decode(seq1.completion_token_ids) → "你好！我是..."
tokenizer.decode(seq2.completion_token_ids) → "我是一个AI助手..."
返回 [{"text": "...", "token_ids": [...]}, ...]
```

---

## 七、实施步骤

### 第 1 步：基础设施（估计 2-3 天，已完成）

**目标**：Config + SamplingParams + Sequence + 构建系统

实现内容：
1. `Config`：解析 `config.json`（用 nlohmann/json）
2. `SamplingParams`：简单 struct
3. `Sequence`：完整状态机
4. CMakeLists.txt + 编译通过

验证：`test_config` 覆盖 Config / SamplingParams；`test_sequence` 覆盖 Sequence 构造、block() / append_token()、空 prompt 边界。

### 第 2 步：BlockManager（估计 2-3 天）

**目标**：完整的页式 KV Cache 管理

实现内容：
1. `Block` 结构
2. `BlockManager`：allocate / deallocate / can_append / may_append
3. `compute_hash`：集成 xxhash

验证：
- 分配 N 个 block，再释放，free 数量正确
- 两个相同 prefix 的序列，第二个分配时 `num_cached_tokens > 0`
- deallocate 后 block 被回收

### 第 3 步：Scheduler（估计 1-2 天）

**目标**：完整的调度逻辑

实现内容：
1. `Scheduler`：schedule / postprocess / preempt
2. 集成 BlockManager

验证：
- 添加多个请求，schedule 按 prefill 优先返回
- 模拟显存不足，触发 preempt，被抢占序列回到 waiting
- postprocess 正确标记 FINISHED 并释放 block

### 第 4 步：Tensor + CUDA 运算（估计 3-5 天）

**目标**：选择并集成 Tensor 库

方案 A：使用 libtorch C++ API
- 安装 libtorch
- 直接用 `torch::Tensor`
- CUDA 操作、内存管理全部由 libtorch 处理

方案 B：自己实现轻量 Tensor（更硬核）
- 自己管理 GPU 内存
- cuBLAS 做 GEMM
- 自己写 CUDA kernel

**推荐方案 A**，后续可逐步替换热点 kernel。

### 第 5 步：算子层实现（估计 5-7 天）

按以下顺序实现，每个实现后单独验证：

1. **RMSNorm**（含 add_rms_forward 融合版）
2. **RoPE**（预计算 cos/sin cache）
3. **SiluAndMul**
4. **store_kvcache kernel**（CUDA kernel，替代 Triton）
5. **Attention**（集成 FlashAttention C API）
6. **Linear 层家族**（Column/Row/QKV/MergedColumn Parallel）
7. **VocabParallelEmbedding + ParallelLMHead**
8. **Sampler**（Gumbel-max trick）

验证：每个算子写单元测试，与 PyTorch 参考实现对比输出精度。

### 第 6 步：权重加载（估计 2-3 天）

**目标**：从 SafeTensors 文件加载权重

实现内容：
1. SafeTensors 解析器（解析 JSON header + mmap 数据）
2. `packed_modules_mapping` 处理
3. `weight_loader` 函数在各 Linear 层中实现
4. BF16 → 工作精度转换

验证：加载 Qwen3-0.6B 权重，打印每层 weight 的 sum/mean，与 Python 版对比。

### 第 7 步：模型组装（估计 2-3 天）

**目标**：完整的 Qwen3ForCausalLM

实现内容：
1. `Qwen3Attention`
2. `Qwen3MLP`
3. `Qwen3DecoderLayer`
4. `Qwen3Model`
5. `Qwen3ForCausalLM`

验证：给定相同输入，C++ 模型输出与 Python 版对比（logits 误差 < 1e-3）。

### 第 8 步：ModelRunner（估计 3-5 天）

**目标**：完整的模型执行器

实现内容：
1. `prepare_prefill` / `prepare_decode`（Context 设置）
2. `run_model`（eager 模式先做）
3. `warmup_model` + `allocate_kv_cache`
4. `run` 整体流程
5. CUDA Graph capture/replay（可后做）

验证：
- 单序列 prefill → 正确 logits
- 多序列 prefill → 正确 logits（对比 varlen 拼接结果）
- Decode → 每步生成合理 token

### 第 9 步：LLMEngine + Tokenizer（估计 2 天）

**目标**：完整的推理引擎

实现内容：
1. 集成 sentencepiece / tiktoken tokenizer
2. `LLMEngine`：generate / step / add_request
3. 吞吐量统计

验证：端到端推理，输出与 Python 版 nano-vllm 一致（给定相同 prompt + temperature=高值按分布比较）。

### 第 10 步：多卡支持（估计 3-5 天，可选）

**目标**：Tensor Parallelism

实现内容：
1. NCCL 初始化（init_process_group）
2. 共享内存通信协议（read_shm / write_shm）
3. all_reduce / gather 在各层中的调用
4. VocabParallelEmbedding 多卡逻辑
5. Worker 进程 / 线程管理

验证：2 卡推理结果与单卡一致。

### 第 11 步：CUDA Graph（估计 2-3 天，可选）

**目标**：Decode 阶段加速

实现内容：
1. Graph capture（逆序 bs）
2. Graph replay（slot_mapping fill -1, 取前 bs 行）
3. Graph pool 共享

验证：CUDA Graph 模式输出与 eager 模式一致，decode 延迟降低。

---

## 八、C++ 特有设计考量

### 8.1 内存管理

| Python 做法 | C++ 建议 |
|-------------|---------|
| GC 自动回收 | Sequence 用 std::unique_ptr 管理生命周期 |
| pin_memory + non_blocking | cudaHostAlloc + cudaMemcpyAsync |
| torch.empty | libtorch at::empty 或 cudaMalloc |
| mmap SafeTensors | mmap 或 std::ifstream |

### 8.2 全局 Context

Python 用全局变量 `_CONTEXT`。C++ 建议：
- 单线程：全局 `Context` 实例
- 多线程：`thread_local Context` 或传引用

### 8.3 序列化（多卡）

Python 用 pickle。C++ 建议：
- 简单自定义二进制格式（写 header + 数据）
- 或用 flatbuffers / protobuf（如果需要扩展性）
- **优化**：decode 阶段只传 `last_token` + `block_table`

### 8.4 动态批次大小

Python 版直接 list 拼接。C++ 要注意：
- Prefill 的 varlen 格式：所有序列的 token 拼成一个 1D tensor
- Decode 直接是 [batch_size] 的 tensor
- CUDA Graph 的 padding：slot_mapping 填 -1，多余行 zero

### 8.5 torch.compile 替代

Python 版大量使用 `@torch.compile`。C++ 中：
- 手写 CUDA kernel（RMSNorm, RoPE, SiluAndMul, Sampler）
- 对性能要求不高的可用 libtorch 连接 cuBLAS
- 考虑 kernel fusion（如 add_rms_norm 可以手写一个融合 kernel）

### 8.6 错误处理

建议在以下边界加检查：
- Config 初始化（路径存在、参数合法）
- BlockManager（断言 ref_count、free blocks 够用）
- SafeTensors 加载（权重名匹配、shape 正确）
- CUDA 操作（cudaGetLastError）

### 8.7 性能关键路径

按优先级排序需要优化的 kernel：
1. **GEMM**（MatMul）：用 cuBLAS/cuBLASLt，占 80%+ 推理时间
2. **FlashAttention**：用 FlashAttention C API
3. **store_kvcache**：简单内存拷贝 kernel（已给出 CUDA 实现）
4. **RMSNorm / RoPE**：手写 CUDA kernel 或用 libtorch
5. **Gumbel-max sampling**：cuRAND + argmax kernel
