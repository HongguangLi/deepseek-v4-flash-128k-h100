# DeepSeek-V4-Flash 128K 全参训练 — 16×8 H100-80GB / RoCE

配套 [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 官方
[`examples/models/deepseek_v4`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)
的**可运行扩展仓库**：官方 example 覆盖到 32 GPU / 4K 上下文，本仓库在其之上提供
128×H100、128K 上下文（CP16 + THD）的 recipe 与 Slurm 启动脚本，全部复用官方入口
（`run_recipe.py` / `run_conversion.py`），不修改 Megatron-Bridge 源码。

> 2026-07-23 逐条对照 Megatron-Bridge `main` 与 Megatron-LM 上游代码核实。官方 README 中
> "Context parallel / long-context (≥64K): TODO" 是 2026-06-02 的旧状态——THD 打包
> （[#5011](https://github.com/NVIDIA/Megatron-LM/pull/5011)，2026-06-26 merged）与 DSv4
> Context Parallel（[#5087](https://github.com/NVIDIA/Megatron-LM/pull/5087)，2026-07-03
> merged）均已合入，以代码和 `.dev.commit` 为准。

## 方案结论：两阶段

"100% 可行"只能建立在官方已验证的路径上。当前官方**已端到端验证**的 DSv4-Flash 全参训练
是 4K / SBHD / bf16+Adam / TP1·PP4·EP8·CP1（H100 需 ≥64 卡使 fp32 master 权重可分片）。
128K + CP16 的代码支持已合入并带上游测试，但没有公开记录在 128×H100 规模上跑通过。因此：

- **阶段 A（保底，可称 100% 可行）**：4K 基线 = 官方验证配置在 128 卡上的直接放大（DP=32），
  用于验证权重导入、软件栈、RoCE、拓扑——与长上下文无关的一切。
- **阶段 B（目标）**：128K / TP1·PP8·CP16·EP8 / THD+contiguous，每卡本地序列 8192 tokens，
  分布式优化器在 DP×CP=16 上分片 master 权重。必须通过下文 Step 2/3/6/7 的
  网络、数值、显存、稳定性四级验收后才可长跑——"可行"由验收证明，不由静态配置保证。

DSv4 CP 的三个硬性要求（与普通 attention 的 ring/hierarchical CP 完全不同，
`cp_comm_type=a2a+p2p` 之类的配置不适用）：THD（packed）输入、
`cp_partition_mode="contiguous"`、非空 `sequence_packing_scheduler`。
本仓库 recipe 已内置，无需手工设置。

## 仓库结构

```text
├── README.md                                  本文件（方案 + 运行手册）
├── recipes/
│   └── dsv4_flash_h100_128gpu.py              5 个 recipe（见下表），派生自官方 no_mtp SFT recipe
└── scripts/
    ├── run_dsv4_recipe.py                     训练入口：注册本仓库 recipe 后转调官方 run_recipe.py
    ├── check_stack.sh                         软件栈校验（dev pin / CP 代码 / 依赖）
    ├── env_roce.sh                            RoCE / NCCL 环境模板
    ├── nccl_level1.sh                         第一级验收：nccl-tests（1/2/4/16 节点）
    ├── slurm_import.sh                        Phase 0：HF → Megatron bf16 权重导入（一次性）
    ├── slurm_stage_a_4k.sh                    阶段 A：4K 保底冒烟（官方验证路径放大）
    └── slurm_stage_b_128k.sh                  阶段 B：显存阶梯 32K/64K/128K + 128K 长跑
```

## Recipes

全部派生自官方 `deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config`（bf16/Adam、
selective recompute、indexer loss=0、H100 自动关 fused-mHC），只改拓扑/序列/CP 字段，
并在改 PP 后重算 DSv4 pipeline layout。

| Recipe | Seq | TP/PP/CP/EP | DP | 每卡本地序列 | 用途 |
|---|---:|---|---:|---:|---|
| `dsv4_flash_sft_h100_128gpu_4k_baseline_config` | 4K | 1/4/1/8 | 32 | 4096 | 阶段 A 保底（官方验证路径） |
| `dsv4_flash_sft_h100_128gpu_32k_cp4_config` | 32K | 1/8/4/8 | 4 | 8192 | 显存阶梯 1/3 |
| `dsv4_flash_sft_h100_128gpu_64k_cp8_config` | 64K | 1/8/8/8 | 2 | 8192 | 显存阶梯 2/3 |
| `dsv4_flash_sft_h100_128gpu_128k_cp16_config` | 128K | 1/8/16/8 | 1 | 8192 | **目标配置** |
| `dsv4_flash_sft_h100_128gpu_128k_cp32_fallback_config` | 128K | 1/4/32/8 | 1 | 4096 | 显存保底（+full recompute） |

CP>1 的 recipe 自动携带 DSv4 CP 硬性要求（Megatron-LM [#5087](https://github.com/NVIDIA/Megatron-LM/pull/5087)）：
`cp_partition_mode="contiguous"` + `sequence_packing_scheduler="dp_balanced"` +
`max_seqlen_per_dp_cp_rank=seq/cp`；启动脚本同时传 `dataset.enable_offline_packing=true`（THD 输入）。

## 前置条件

1. **容器**：含 DSv4 依赖的 NeMo FW 容器（或按官方 `docker/Dockerfile.ci` 自建），
   内含 `/opt/Megatron-Bridge`、`fast_hadamard_transform`、预编译 `helpers_cpp`。
2. **Megatron-LM 必须切到 Bridge 固定的 dev commit**（含 THD [#5011](https://github.com/NVIDIA/Megatron-LM/pull/5011) 与 DSv4 CP [#5087](https://github.com/NVIDIA/Megatron-LM/pull/5087)）：

   ```bash
   cd /opt/Megatron-Bridge
   ./scripts/switch_mcore.sh dev     # -> bfa33263ca06e6974410d0ea871b25e21c5aee85
   uv sync
   ```

3. **存储**：`HF_HOME`（~200GB 下载缓存）与 `MEGATRON_DIR`（~570GB bf16 导入产物）放共享存储。
4. **本仓库**放共享存储并通过 `CONTAINER_MOUNTS` 挂进容器，路径填给 `REPO_DIR`。
5. 编辑 `scripts/env_roce.sh`：`NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` 填集群真实值。
6. 各 slurm 脚本头部的 `#SBATCH --account/--partition` 与 `CONTAINER_IMAGE` 改成集群实际值。

## 执行顺序（严格按 0→8 执行，前一步不通过不进下一步）

先做一次性配置（只做一次）：

1. 每个 `scripts/slurm_*.sh` 头部：改 `#SBATCH --account` / `--partition`；
2. `scripts/env_roce.sh`：填 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 集群真实值；
3. 准备好三个路径并在下面所有命令前 export：

```bash
export CONTAINER_IMAGE=/path/to/nemo_fw.sqsh          # 含 DSv4 依赖的容器
export CONTAINER_MOUNTS=/lustre:/lustre               # 必须盖住下面三个目录
export REPO_DIR=/lustre/deepseek-v4-flash-128k-h100   # 本仓库(共享存储上的 clone)
export WORKSPACE=/lustre/dsv4                         # 结果/模型目录
export DATASET_ROOT=/lustre/data/longctx_jsonl        # 客户长上下文 JSONL 数据
```

### Step 0 — 软件栈校验（1 节点，几分钟）

```bash
srun -N1 --container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS \
    bash -lc "cd /opt/Megatron-Bridge && ./scripts/switch_mcore.sh dev && uv sync && bash $REPO_DIR/scripts/check_stack.sh"
```

**通过**：输出 `PASS`（mcore = `bfa33263…`、`csa_cp_utils` 可导入、`fast_hadamard_transform` 在位）。
**失败**：容器不含 DSv4 依赖 → 换 NeMo FW 容器或按官方 `docker/Dockerfile.ci` 自建，不要带病继续。

### Step 1 — recipe 干跑（1 节点，不花 GPU 时）

对 5 个 recipe 逐个做 `--dryrun`，验证 config 构建、pipeline layout、字段名在你的容器版本上全部成立：

```bash
srun -N1 --container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS \
    bash -lc "cd /opt/Megatron-Bridge && for r in \
      dsv4_flash_sft_h100_128gpu_4k_baseline_config \
      dsv4_flash_sft_h100_128gpu_32k_cp4_config \
      dsv4_flash_sft_h100_128gpu_64k_cp8_config \
      dsv4_flash_sft_h100_128gpu_128k_cp16_config \
      dsv4_flash_sft_h100_128gpu_128k_cp32_fallback_config; do \
        echo \"== \$r ==\"; \
        uv run --no-sync python $REPO_DIR/scripts/run_dsv4_recipe.py \
            --recipe \$r --step_func gpt_step --dataset mock --dryrun; \
    done"
```

**通过**：5 个全部构建成功。**失败**：报错会指明字段名/layout 问题，先修 recipe 再往下走。

### Step 2 — 第一级验收：网络（1/2/4/16 节点各一次）

```bash
for n in 1 2 4 16; do sbatch -N $n scripts/nccl_level1.sh; done
```

**通过**：每项 `#wrong=0`；无 WARN/timeout；日志显示 `NET/IB` 而非 `NET/Socket`；各节点带宽无离群。
**失败**：交给网络团队，训练不要开始。

### Step 3 — 第二级验收：DSv4 CP 正确性（1 节点 8 GPU）

在固定软件栈上原样重跑上游 #5087 的测试：

```bash
srun -N1 --gpus-per-node=8 --container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS \
    bash -lc "cd /opt/Megatron-Bridge/3rdparty/Megatron-LM && \
      uv run --no-sync torchrun --nproc_per_node=8 -m pytest -x -q \
        tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_attention_cp.py \
        tests/unit_tests/transformer/experimental_attention_variant/test_csa_cp_utils.py \
        tests/unit_tests/transformer/experimental_attention_variant/test_csa_cp_layout_kernels.py \
        tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_native_parity.py \
        tests/unit_tests/test_sequence_packing.py"
```

（分布式测试的具体调起方式以 Megatron-LM `tests/` 内约定为准；个别用例需要特定 world size 时按其 skip 条件拆开跑。）
**通过**：全绿。**失败**：说明你们的容器栈与上游 CI 有差异，升级 Solution Team，**不要**启动 128K。

### Step 4 — Phase 0：权重导入（2 节点，一次性）

```bash
sbatch scripts/slurm_import.sh
```

**通过**：`$WORKSPACE/models/DeepSeek-V4-Flash/iter_0000000/.metadata` 存在（~570GB bf16）。
重复提交会自动跳过。

### Step 5 — 阶段 A：4K 保底冒烟（16 节点，50 iter，约 1-2 小时）

```bash
sbatch scripts/slurm_stage_a_4k.sh
```

**通过**：loss 有限且下降、无 NaN、NCCL 无 WARN/Socket 回退。
这一步验证的是权重导入/软件栈/16 节点 RoCE/PP·EP 拓扑——与长上下文无关的一切。
**失败**：问题一定不在 CP/128K，在基础栈里；修好前不进 Step 6。

### Step 6 — 第三级验收：显存阶梯（16 节点，每档 20 iter，依次通过）

```bash
STEP=32k  sbatch scripts/slurm_stage_b_128k.sh
STEP=64k  sbatch scripts/slurm_stage_b_128k.sh
STEP=128k sbatch scripts/slurm_stage_b_128k.sh
```

**每档通过标准**：完成 fwd/bwd/optimizer step；无 OOM/NaN/Inf；128K 档另加：
每卡 `max_memory_allocated ≤ 72 GiB`、reserved 不持续增长、所有 rank 本地序列 ≈ 8192、
日志确认 THD + contiguous CP 切分生效。
**128K 档超 72GiB**：先在该 recipe 上叠 full recompute
（追加 `model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1`）；
仍超 → `STEP=cp32`；再不行 → 停，回阶段 A 并升级 Solution Team。

### Step 7 — 第四级验收：稳定性（16 节点）

同一配置按 1 → 10 → 100 → 1000 iter 递增：

```bash
STEP=128k TRAIN_ITERS=1    sbatch scripts/slurm_stage_b_128k.sh
STEP=128k TRAIN_ITERS=10   sbatch scripts/slurm_stage_b_128k.sh
STEP=128k TRAIN_ITERS=100  sbatch scripts/slurm_stage_b_128k.sh
STEP=128k TRAIN_ITERS=1000 SAVE_CKPT=1 RUN_NAME=dsv4_128k_stab sbatch scripts/slurm_stage_b_128k.sh
```

1000-iter 跑完后做 save→restart→load 连续性验证：**用同一个 `RUN_NAME` 重提交**、把
`TRAIN_ITERS` 加 100，脚本会自动从该目录的最新 checkpoint 续跑（`checkpoint.save` 与
`checkpoint.load` 指向同一目录）：

```bash
STEP=128k TRAIN_ITERS=1100 SAVE_CKPT=1 RUN_NAME=dsv4_128k_stab sbatch scripts/slurm_stage_b_128k.sh
```

**通过**：loss/grad norm 有限；无 iter-2 NaN；恢复后 loss 轨迹连续（对比 restart 前后
log 的 loss 曲线无跳变）；峰值显存预留 ≥ 8 GiB；RoCE 无重传风暴/NCCL timeout/rank stall。

### Step 8 — 正式长跑（16 节点）

Step 7 全部通过后提交正式训练。与阶梯冒烟的区别只有四个参数：固定的 `RUN_NAME`
（决定 checkpoint 目录，续跑靠它）、正式的 `TRAIN_ITERS`/`LR_WARMUP_ITERS`、
提高的 `GBS`（DP=1 下即梯度累积步数，吞吐换显存无代价，建议 8–32）：

```bash
STEP=128k \
RUN_NAME=dsv4_flash_128k_prod \
SAVE_CKPT=1 SAVE_INTERVAL=250 \
TRAIN_ITERS=5000 LR=5e-6 LR_WARMUP_ITERS=100 GBS=16 \
DATASET_ROOT=/lustre/data/longctx_jsonl \
sbatch scripts/slurm_stage_b_128k.sh
```

（`TRAIN_ITERS`/`LR`/`GBS` 按客户数据量与训练目标定，上面是示例值；
`SAVE_INTERVAL=250` 即约每 250 iter 存一次，`most_recent_k=1` 只保留最新一份 ~570GB。）

- **断点续跑 / 追加训练**：原样重提交同一条命令即可（同 `RUN_NAME` 自动从最新 checkpoint
  恢复）；要多训一段就只改大 `TRAIN_ITERS`。
- **时限接力**：集群单作业时限不够时，用同名作业串行排队：
  `sbatch --job-name=dsv4prod --dependency=singleton ...` 连提多次，每段自动续上一段。
- 至此才可以对客户说这套配置是经过验证的。

## 验收标准（汇总）

- **网络**：`#wrong=0`、无 WARN/timeout、日志 `NET/IB` 而非 `NET/Socket`。
- **128K 显存**：每卡 `max_memory_allocated ≤ 72 GiB`，reserved 不持续增长，所有 rank 本地序列 ≈ 8192。
- **稳定性**：loss/grad norm 有限、无 iter-2 NaN、checkpoint 恢复后 loss 连续、峰值预留 ≥ 8 GiB。
- **降级顺序**：selective→full recompute → `STEP=cp32` → 仍失败则停止 128K 长跑，回阶段 A 并升级 Solution Team。

## 性能调优（"能跑但 128K 奇慢"时按序执行）

先量化再调优：记录当前每 iter 耗时与 `tokens/s/GPU`（= GBS×131072 ÷ iter 时间 ÷ 128），
每改一项只改一项，对比数字。

### 吞吐经验值（16×8 H100-80GB，DSv4-Flash ~285B 全参 SFT）

估算方法：TPS ≈ 128×989 TFLOPs × MFU ÷ (6 × 激活参数量 × 重计算系数)；
按激活 ~30B、128K+CP+RoCE+新 kernel 的现实 MFU 3–15% 估：

| 序列长度 | 集群 TPS 经验带 | 判读 |
|---|---:|---|
| 4K–8K（阶段 A） | 200K–450K | <150K 需排查 |
| 32K / CP4 | 90K–220K | <60K 需排查 |
| 64K / CP8 | 40K–120K | <30K 需排查 |
| **128K / CP16** | **15K–60K** | 及格 ≥15K，调优后健康目标 25K–50K；<10K 一定是配置问题 |

这是工程估算带，不是官方 benchmark（DSv4-Flash 128K on H100 没有公开官方数据）；
需要背书数字时走 Solution Team。实测远低于带（如个位数 K）几乎总能归因到:
GBS 太小（流水线气泡）、full recompute、CP 档位不当、MTP 没关、未开 fused DSA、
或 NCCL 回退 `NET/Socket`——即下面各节，按收益排序。

**调优纪律**：一次只改一项；每项用 5-step 探针验收；每次记录三个数——
稳态 iter 时间、TPS、每卡 `max_memory_allocated` 峰值。预测值与实测的偏差本身就是
诊断信息（如 GBS 提高后 TPS 低于气泡公式预测 → CP 通信占比超预期）。

叠加预期示例（实测案例：32K / CP16 / PP8 / full recompute / GBS=1 起点 3.8K TPS）：

| 叠加项 | 预计 TPS |
|---|---:|
| GBS=32（气泡 87.5%→18%） | ~25K |
| + selective recompute（省 ~1/4 计算） | ~31K |
| + SM90 fused DSA（×1.3 估） | **~40K** |

这大约是 unfused→fused 过渡期在 CP16 下的天花板；再往上需要更长的 local seq
（更低 CP 档）+ Dynamic CP，见下文。

### 1. GBS 太小 → 流水线气泡吃掉大部分算力（最常见，零成本）

流水线气泡占比 ≈ (PP−1)/(m+PP−1)，m = 梯度累积步数 = GBS（DP=1 时）。

| 配置 | GBS=1 | GBS=8 | GBS=16 | GBS=32 |
|---|---:|---:|---:|---:|
| PP4（cp32 档） | **75% 空转** | 27% | 16% | 9% |
| PP8（cp16 档） | **87.5% 空转** | 47% | 30% | 18% |

冒烟脚本默认 GBS=1 只用于验证能跑；测速/长跑必须提高：

```bash
STEP=cp32 GBS=16 TRAIN_ITERS=20 sbatch scripts/slurm_stage_b_128k.sh
```

GBS 是梯度累积，不改变单 microbatch 显存；但注意 **1F1B 下 stage-0 同时持有
`min(GBS, PP)` 份 microbatch 的激活存根**——full recompute 下每份只是各层输入
（~百 MB 级），增量约 1–2 GiB，且在 GBS=PP 处封顶（再大只加时间不加显存）。
显存贴边（峰值距容量 <3 GiB）的配置，先用 **GBS=PP 跑 5-step 探针**确认峰值，
过了再上 16/32（与 GBS=PP 显存相同）；OOM 则退 GBS=PP/2。
lr/warmup 按等效 batch 相应调整。

### 2. full → selective recompute（省 ~1/4 计算）

full recompute 每步白算一次前向。官方 recipe 的 selective 组合：

```yaml
recompute_granularity: selective
recompute_modules: ["moe_act", "mhc"]
recompute_method: null
recompute_num_layers: null
```

selective 每 microbatch 存的激活比 full 多，与 GBS 的在途份数（`min(GBS,PP)`）
相乘——所以顺序是 **先小 GBS + selective 探针，过了再抬 GBS**。full 只在
显存验收不过时作为降级手段。

### 3. 关闭 MTP（SFT 不需要）

MTP 层白吃一层的计算、显存和 loss head。用官方 no_mtp 方式关闭
（`deepseek_v4_flash_no_mtp_sft_config` 的做法：`mtp_num_layers=None` 并把
`csa_compress_ratios` 裁剪回 `num_layers`），**不要只把 loss 系数置 0**——那样层还在算。
本仓库 recipes 已默认 no_mtp。

### 4. CP 档位：显存允许时用更低的 CP（更长 local seq）

**红线：还在 unfused DSA/CSA 路径上时，不要为了效率降 CP。** unfused backward 会
物化随 local seq 增长的大临时张量（实测案例：64K/CP16、local 4096 时单笔申请
45.86 GiB 直接 OOM）。降 CP = 提 local seq = 撞同一堵墙。先完成第 5 步（fused DSA）
再回来降 CP。

fused 栈上的原则：local seq 越长计算/通信比越好，CP 组跨的节点越少越好。
128K 上 CP32 是显存保底档（local 4096、CP 组跨 4 节点、常叠 full recompute），
显存过验收就回 CP16（`STEP=128k`：local 8192、CP 组只跨 2 节点、selective）。
若 CP16 差一点点显存，先试 CP16 + full recompute（仍快于 CP32+full）：

```bash
STEP=128k GBS=16 EXTRA_OVERRIDES="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 model.recompute_modules=null" sbatch scripts/slurm_stage_b_128k.sh
```

### 5. 打开 H100 的 fused DSA kernel（上游 #5087 新放开）

官方 recipe `apply_dsa_kernel_fusion=False` 是 SM100-only 时代的老默认；
[#5087](https://github.com/NVIDIA/Megatron-LM/pull/5087) 已将 fused DSA 放开到
SM90（Hopper）。这一项同时打两个靶：长上下文下 attention/indexer 的计算大头下降，
且 **unfused backward 物化的大临时张量（45.86 GiB 一类）随之消失**——它既是吞吐项
也是解锁更低 CP 档 / 更长上下文的容量项。
前提：容器内 `pip install --no-deps "nvidia-cudnn-frontend[cutedsl]>=1.24.0"`
（全量安装会遮蔽容器 CUDA）。先 100 iter 与未开启时做 loss 对照再长跑：

```bash
STEP=128k GBS=16 EXTRA_OVERRIDES="model.apply_dsa_kernel_fusion=true" sbatch scripts/slurm_stage_b_128k.sh
```

### 6. 通信重叠

```bash
EXTRA_OVERRIDES="comm_overlap.overlap_moe_expert_parallel_comm=true"
```

EP all-to-all 与计算重叠；与 ddp 的 `overlap_grad_reduce/overlap_param_gather`
（Adam recipe 路线默认已开）叠加使用。逐项开、逐项对比。

### 7. 数据是混合长度 → Dynamic CP

静态 CP32 会把 4K 的短样本也切成 32 份跑，纯浪费。若 SFT 数据长短混合，
改用官方 Dynamic CP（`examples/training_features/long_context/`）：

```bash
EXTRA_OVERRIDES="model.dynamic_context_parallel=true model.sequence_packing_scheduler=default_dynamic_cp model.max_seqlen_per_dp_cp_rank=8192 model.min_dynamic_context_parallel_size=1"
```

短样本自动 local_cp_size=1，只有真 128K 样本才占满 CP 组。

### 8. 网络自查

16 节点 `all_gather_perf`/`reduce_scatter_perf` 的 busbw 应接近 RoCE 线速；
训练日志绝不能出现 `NET/Socket`。带宽差一个量级时先修网络再谈调优。

### 其他说明

- **flashattention / liger 用不了是预期行为**，不是缺陷：DSv4 的 CSA/DSA/SWA/mHC 走
  TE/cuDNN 专用 kernel（`fast_hadamard_transform`、cuDNN FE、flash-mla），
  llama-factory 一类通用栈没有这些结构的实现。效率来自并行拓扑 + 上面的 fusion，
  换框架解决不了，反而失去唯一支持 DSv4 CP 的实现。
- CUDA Graph 对 DSv4 THD 路径上游仍是 TODO，不要开。
- 以上都做完仍不达标：抓一次 `nsys` profile（或开 mcore timing log）定位热点，
  连同 tokens/s 数字升级给 Solution Team / perf 团队。

## 已知约束（来自官方 README / 上游代码）

- DSv4 要求 **TP=1**（MLA TP 与 hybrid attention 路径不兼容），用 PP/EP/CP 扩展。
- **不要**用 MXFP8 或 Muon 做 SFT：官方确认分别在 iter-2 NaN / assert（上游 blocker）。
- 全参训练必须从 bf16 导入产物启动，不能直接用 FP8/MXFP4 推理量化权重。
- 不存 optimizer state（脚本已固定 `save_optim=false`）；每个模型 checkpoint ~570GB。
- H100 上 `use_fused_mhc` 由 recipe 自动置 False（fused mHC 为 sm_100 专用）。
- 混合长度 SFT 数据可选 Dynamic CP（官方示例 `examples/training_features/long_context/`）；
  全部样本都接近 128K 时首选本仓库的静态 CP=16。

## 参考来源

- [Megatron-Bridge · examples/models/deepseek_v4（README / slurm_sft.sh / slurm_pretrain.sh / conversion.sh）](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)
- [Megatron-Bridge · src/megatron/bridge/recipes/deepseek/h100/deepseek_v4.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/deepseek/h100/deepseek_v4.py)
- [Megatron-Bridge · examples/training_features/long_context/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/training_features/long_context)
- [Megatron-LM PR #5011 — Packed Sequence (THD) support for DSv4 Hybrid Attention（2026-06-26 merged）](https://github.com/NVIDIA/Megatron-LM/pull/5011)
- [Megatron-LM PR #5087 — DSv4 Context Parallel support（2026-07-03 merged）](https://github.com/NVIDIA/Megatron-LM/pull/5087)
- [Megatron-LM issue #4468 — DSv4 capability tracking](https://github.com/NVIDIA/Megatron-LM/issues/4468)
- [context_parallel package — Megatron Core docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
