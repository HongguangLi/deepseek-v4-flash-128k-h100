# DeepSeek-V4-Flash 128K 全参训练 — 16×8 H100-80GB / RoCE

配套 [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 官方
[`examples/models/deepseek_v4`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)
的**可运行扩展仓库**：官方 example 覆盖到 32 GPU / 4K 上下文，本仓库在其之上提供
128×H100、128K 上下文（CP16 + THD）的 recipe 与 Slurm 启动脚本，全部复用官方入口
（`run_recipe.py` / `run_conversion.py`），不修改 Megatron-Bridge 源码。

完整方案与论证见 [deepseek_v4_flash_128k_h100_roce_plan.md](deepseek_v4_flash_128k_h100_roce_plan.md)。

## 仓库结构

```text
├── README.md                                  本文件
├── deepseek_v4_flash_128k_h100_roce_plan.md   方案文档（菜单、验收标准、降级路径）
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
STEP=128k TRAIN_ITERS=1000 SAVE_CKPT=1 sbatch scripts/slurm_stage_b_128k.sh
```

1000-iter 跑完后做 save→restart→load 连续性验证（加载上一步保存的 checkpoint 再跑 100 iter）：

```bash
STEP=128k TRAIN_ITERS=1100 SAVE_CKPT=1 sbatch scripts/slurm_stage_b_128k.sh
# 并在提交前把上一次的 checkpoint 目录传给 checkpoint.load（编辑脚本 SAVE_OVERRIDES 或加 CLI 覆盖）
```

**通过**：loss/grad norm 有限；无 iter-2 NaN；恢复后 loss 轨迹连续；峰值显存预留 ≥ 8 GiB；
RoCE 无重传风暴/NCCL timeout/rank stall。

### Step 8 — 正式长跑

Step 7 全部通过后，把 `TRAIN_ITERS`、lr/warmup、`SAVE_INTERVAL` 调成正式训练值，
继续用 `STEP=128k SAVE_CKPT=1` 提交。至此才可以对客户说这套配置是经过验证的。

## 验收标准（摘要，详见方案文档）

- **网络**：`#wrong=0`、无 WARN/timeout、日志 `NET/IB` 而非 `NET/Socket`。
- **128K 显存**：每卡 `max_memory_allocated ≤ 72 GiB`，reserved 不持续增长，所有 rank 本地序列 ≈ 8192。
- **稳定性**：loss/grad norm 有限、无 iter-2 NaN、checkpoint 恢复后 loss 连续、峰值预留 ≥ 8 GiB。
- **降级顺序**：selective→full recompute → `STEP=cp32` → 仍失败则停止 128K 长跑，回阶段 A 并升级 Solution Team。

## 已知约束（来自官方 README / 上游代码）

- DSv4 要求 **TP=1**（MLA TP 与 hybrid attention 路径不兼容），用 PP/EP/CP 扩展。
- **不要**用 MXFP8 或 Muon 做 SFT：官方确认分别在 iter-2 NaN / assert（上游 blocker）。
- 全参训练必须从 bf16 导入产物启动，不能直接用 FP8/MXFP4 推理量化权重。
- 不存 optimizer state（脚本已固定 `save_optim=false`）；每个模型 checkpoint ~570GB。
- H100 上 `use_fused_mhc` 由 recipe 自动置 False（fused mHC 为 sm_100 专用）。
