# DeepSeek-V4-Flash 128K · 16×8 H100 · RoCE 训练方案

> 2026-07-23 逐条对照 [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) `main` 分支与 Megatron-LM 上游代码核实。注意：官方 README（examples/models/deepseek_v4/README.md）中 "Context parallel / long-context (≥64K): TODO" 是 2026-06-02 的旧状态——THD 打包（[Megatron-LM #5011](https://github.com/NVIDIA/Megatron-LM/pull/5011)，2026-06-26 merged）与 DSv4 Context Parallel（[Megatron-LM #5087](https://github.com/NVIDIA/Megatron-LM/pull/5087)，2026-07-03 merged）均已合入，以代码和 `.dev.commit` 为准。

## 1. 结论：两阶段菜单

"100% 可行"只能建立在官方已验证的路径上。当前官方**已端到端验证**的 DSv4-Flash 全参训练是：SBHD、bf16/Adam、4K 上下文、TP1/PP4/EP8/CP1（GB300 32 卡验证；H100 需 ≥64 卡使 fp32 master 权重可分片）。128K + CP16 的**代码支持已合入并带上游单测/功能测试**，但没有任何公开记录在 128×H100 规模上跑通过。因此菜单分两阶段，阶段 A 是保底，阶段 B 是目标，之间用第 5 节的验收阶梯衔接。

### 阶段 A（保底，官方验证路径的直接放大，先跑通再谈 128K）

| 项目 | 配置 |
|---|---:|
| GPU | 128 × H100 80GB（16 节点 × 8） |
| Recipe | `deepseek_v4_flash_no_mtp_sft_config`（或需要 MTP 时 `deepseek_v4_flash_sft_config`） |
| TP / PP / EP / CP | 1 / 4 / 8 / 1 |
| DP | 4（128 ÷ (PP4·CP1)）— fp32 master 可分片，规避 32 卡 DP=1 的 OOM |
| 序列长度 | 4096（recipe 默认），SBHD、不打包 |
| 精度 / 优化器 | bf16 / Adam（**不要**用 MXFP8 或 Muon 做 SFT：官方确认 iter-2 分别 NaN / assert） |
| mHC | `use_fused_mhc` 由 recipe 自动判定（H100 上自动为 False，无需手动设置） |

### 阶段 B（目标：128K，代码已支持，须通过第 5 节验收后才可长跑）

| 项目 | 配置 |
|---|---:|
| TP | 1（DSv4 hybrid attention 硬性要求） |
| PP | 8 |
| CP | 16（每 CP group 横跨 2 节点） |
| EP | 8 |
| DP | 1（分布式优化器在 DP×CP=16 上分片 master 权重） |
| 序列长度 | 131072，THD（packed） |
| 单卡本地序列 | 8192 tokens（`max_seqlen_per_dp_cp_rank=8192`） |
| CP 数据切分 | `cp_partition_mode="contiguous"`（DSv4 CP 唯一合法值） |
| 打包调度 | `sequence_packing_scheduler="dp_balanced"`（DSv4+CP 校验强制要求非空） |
| MBS / GBS | 1 / 1 起步，稳定后按吞吐提高 GBS（DP=1 下 GBS 即梯度累积步数） |
| 精度 | bf16 / Adam |
| 重计算 | 先 recipe 默认 selective `["moe_act","mhc"]`；OOM 再 `full/uniform/1` |
| MTP | 关闭（no_mtp recipe） |
| Indexer loss | 0（recipe 默认 `dsa_indexer_loss_coeff=0.0`；H100/SM90 上 fused DSA + dense indexer loss 明确不支持，保持关闭） |
| DSA kernel fusion | `apply_dsa_kernel_fusion=False`（recipe 默认；#5087 虽放开 SM90，但 CP+ratio-4 indexer 需特定 cuDNN Frontend wrapper，首轮不要开） |
| CUDA Graph | 关闭（`cuda_graph_impl="none"`，recipe 默认） |

混合长度 SFT 数据可选 Dynamic CP（官方示例 `examples/training_features/long_context/`）：`model.dynamic_context_parallel=True`、`model.sequence_packing_scheduler="default_dynamic_cp"`、`model.max_seqlen_per_dp_cp_rank=8192`、`model.min_dynamic_context_parallel_size=1`。全部样本都接近 128K 时静态 CP=16 更简单可控，作为首选。

## 2. 阶段 B 的配置方式

**不要只用 CLI 覆盖 PP。** 官方 recipe 在函数体内按 PP=4 计算 DSv4 的 pipeline layout（`set_deepseek_v4_pipeline_model_parallel_layout`），CLI 事后把 PP 改成 8 不会重算 layout。正确做法是在自己的 checkout 里加一个 recipe 变体（基于 `src/megatron/bridge/recipes/deepseek/h100/deepseek_v4.py`）：

```python
def deepseek_v4_flash_no_mtp_sft_128gpu_h100_128k_config() -> ConfigContainer:
    cfg = deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config()

    # 并行拓扑：TP1 / PP8 / CP16 / EP8，DP=1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 16

    # 128K + THD + DSv4 CP 硬性要求
    cfg.model.seq_length = 131072
    cfg.dataset.seq_length = 131072
    cfg.model.cp_partition_mode = "contiguous"
    cfg.model.sequence_packing_scheduler = "dp_balanced"
    cfg.model.max_seqlen_per_dp_cp_rank = 8192

    # PP 改动后必须重算 DSv4 pipeline layout
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 1

    # 首轮不存 optimizer state（分布式 Adam 全量存档是数 TB 级）
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.load_optim = False
    return cfg
```

注意事项：

* 数据必须走打包（THD）路径产生 `cu_seqlens`（SFT 数据集开启打包；`gpt_step` 会按 CP 切分打包批次）。打包相关字段名以你固定的 Bridge commit 里 `default_squad_config` / dataset config 实际字段为准，不要照抄本文。
* `set_deepseek_v4_pipeline_model_parallel_layout` 对 PP=8 的层排布（Flash 为 43 层 + 可选 MTP）需在 dry-run 中确认各 stage 层数均衡；官方已用它跑过 Pro 的 PP=16。
* 权重必须是转换后的 bf16 Megatron checkpoint（`conversion.sh` 导入，~570 GB）；不能直接拿 FP8/MXFP4 推理量化权重当全参训练起点。HF 下载缓存 + 导入权重合计约 750 GB，`HF_HOME` 放共享存储。

## 3. 必须固定的软件版本

```bash
cd Megatron-Bridge
git rev-parse HEAD                      # 记录 Bridge commit
./scripts/switch_mcore.sh dev           # 切到 .dev.commit 固定的 Megatron-LM dev commit
./scripts/switch_mcore.sh status        # 确认 = bfa33263ca06e6974410d0ea871b25e21c5aee85
uv sync
```

* `.dev.commit` 当前指向 `bfa33263`，即 [Megatron-LM #5087](https://github.com/NVIDIA/Megatron-LM/pull/5087)（DSv4 CP）的 merge commit，天然包含 #5011（THD）。**不要**用容器里自带的 Megatron-LM，也不要 `git pull` 浮动更新——Bridge 与 MCore 版本偏移曾直接导致 `dsv4_hybrid` 无法注册（内部 bug 6232716）。
* 依赖：`fast_hadamard_transform` 必须从 [Dao-AILab git 源](https://github.com/Dao-AILab/fast-hadamard-transform) 安装（PyPI sdist 不完整）；如启用 THD 的 DSA kernel，需 `nvidia-cudnn-frontend[cutedsl]>=1.24.0` 且 **`pip install --no-deps`**（全量安装会遮蔽容器 CUDA、破坏 TE）。
* 记录 `transformer_engine`、`torch`、`nvidia.cudnn` 版本，与 NCCL 日志一起归档。

## 4. RoCE 环境与启动

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_PXN_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=<bootstrap_ethernet_interface>
export NCCL_IB_HCA=<validated_connectx_hca_list>
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

`NCCL_IB_GID_INDEX / NCCL_IB_TC / NCCL_IB_SL / NCCL_ALGO / NCCL_PROTO / NCCL_CROSS_NIC` 只有在网络团队给出经过验证的 RoCEv2 GID、PFC/ECN、traffic class 与多 rail 配置时才固定。NCCL 日志必须显示 `NET/IB`（RoCE RDMA），出现 `NET/Socket` 回退即视为环境不合格。

启动以 `examples/models/deepseek_v4/slurm_sft.sh` 为模板改造（HARDWARE=hopper、16 节点 × 8 卡）：

```bash
torchrun \
  --nnodes=16 \
  --nproc-per-node=8 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  scripts/training/run_recipe.py \
  --recipe deepseek_v4_flash_no_mtp_sft_128gpu_h100_128k_config \
  --dataset local-jsonl dataset.dataset_root=<path_to_long_context_jsonl>
```

## 5. 上线前四级验收（不可跳级）

### 第一级：网络

按 1、2、4、16 节点依次执行 `nccl-tests`（`all_reduce_perf`、`all_gather_perf`、`reduce_scatter_perf`、`sendrecv_perf`）。要求 `#wrong = 0`、无 WARN/timeout/socket 回退、各节点带宽无离群、GPU-NIC 亲和正确。

### 第二级：CP 正确性（在你们固定的软件栈上重跑上游测试）

#5087 已带 CP 单测/功能测试，在集群容器内原样跑通：

```text
tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_attention_cp.py
tests/unit_tests/transformer/experimental_attention_variant/test_csa_cp_utils.py
tests/unit_tests/transformer/experimental_attention_variant/test_csa_cp_layout_kernels.py
tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_native_parity.py
tests/unit_tests/test_sequence_packing.py
```

再做一次小规模端到端对照：2 GPU、16K context，CP=1 vs CP=2，比较 forward loss、参数梯度、CSA/HCA/SWA 输出、checkpoint save/load。DSv4 同时含 CSA、HCA、SWA 与 mHC，不能拿普通 GQA/MLA 的 CP 测试代替。

### 第三级：显存阶梯（保持 PP=8 不变）

```text
32K  / CP=4   （本地序列 8192）
64K  / CP=8   （本地序列 8192）
128K / CP=16  （本地序列 8192）
```

128K 验收要求：每卡 `max_memory_allocated ≤ 72 GiB`；reserved 不持续增长；完成 fwd/bwd/optimizer step；无 OOM/NaN/Inf；所有 rank 本地序列 ≈ 8192；日志确认 THD + contiguous CP 切分生效。

### 第四级：稳定性

```text
1 → 10 → 100 → 1000 steps → save → restart → load → 再 100 steps
```

要求：loss、grad norm 有限；无 iter-2 NaN；恢复后 loss 轨迹连续；峰值显存预留 ≥ 8 GiB；RoCE 无重传风暴、NCCL timeout 或 rank stall。

## 6. 失败时的降级路径

CP=16 峰值超 72 GiB 时按顺序执行，不要先加 TP（DSv4 禁止 TP>1），也不要开 CPU offload：

1. selective → full recompute（`recompute_granularity=full, recompute_method=uniform, recompute_num_layers=1`）；
2. 仍超 → `PP=4 / CP=32`（每卡本地序列 4096，DP×CP=32 分片更细，但 CP group 横跨 4 节点，RoCE 压力更大，仅作显存保底）；
3. CP 正确性测试不过、或 iter-2 NaN、或 CP=32 仍超 72 GiB → 停止 128K 全参长跑，回到阶段 A 并升级 Solution Team 处理；此时不存在仅靠调配置就能保证正确的方案。

## 7. 最终判断

* **阶段 A（4K、TP1/PP4/EP8/CP1、bf16/Adam、128×H100）是当前唯一可以称为"100% 可行"的菜单**：它是官方端到端验证配置在满足 ≥64 卡 H100 约束下的直接放大。
* 阶段 B（128K、TP1/PP8/CP16/EP8、THD+contiguous CP）自 2026-07-03 起代码路径完整且带上游测试，是当前硬件下拓扑最合理的目标菜单；但在 128×H100 规模上没有公开验证记录，"可行"必须由第 5 节的网络、数值、显存、1000-step 四级验收共同证明，不能由静态配置保证。

---

## Sources

- [Megatron-Bridge · examples/models/deepseek_v4/README.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/deepseek_v4/README.md)
- [Megatron-Bridge · examples/models/deepseek_v4/slurm_sft.sh / slurm_pretrain.sh](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)
- [Megatron-Bridge · src/megatron/bridge/recipes/deepseek/h100/deepseek_v4.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/deepseek/h100/deepseek_v4.py)
- [Megatron-Bridge · examples/training_features/long_context/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/training_features/long_context)
- [Megatron-LM PR #5011 — Packed Sequence (THD) support for DSv4 Hybrid Attention（2026-06-26 merged）](https://github.com/NVIDIA/Megatron-LM/pull/5011)
- [Megatron-LM PR #5087 — DSv4 Context Parallel support（2026-07-03 merged）](https://github.com/NVIDIA/Megatron-LM/pull/5087)
- [Megatron-LM issue #4468 — DSv4 capability tracking](https://github.com/NVIDIA/Megatron-LM/issues/4468)
- [context_parallel package — Megatron Core docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- 内部记录：NVBugs 6232716（Bridge/MCore 版本偏移导致 `dsv4_hybrid` 注册失败）；Solution Team Slack thread（Dong ↔ Alexandros）
