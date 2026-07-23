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

## Quickstart（按序执行，不可跳级）

```bash
# 0. 软件栈校验（容器内）
srun -N1 --container-image=$IMG --container-mounts=$MOUNTS \
    bash $REPO_DIR/scripts/check_stack.sh

# 1. 第一级验收: 网络（1/2/4/16 节点各一次）
for n in 1 2 4 16; do sbatch -N $n scripts/nccl_level1.sh; done

# 2. 第二级验收: 在固定软件栈上重跑上游 DSv4 CP 测试（单节点容器内）
#    tests/unit_tests/transformer/experimental_attention_variant/
#      test_dsv4_hybrid_attention_cp.py  test_csa_cp_utils.py
#      test_csa_cp_layout_kernels.py     test_dsv4_hybrid_native_parity.py
#    tests/unit_tests/test_sequence_packing.py

# 3. Phase 0: 权重导入（一次性, 2 节点, TP1/PP1/EP16 官方验证拓扑）
sbatch scripts/slurm_import.sh

# 4. 阶段 A: 4K 保底冒烟（16 节点, 50 iter）
REPO_DIR=/workspace/harvey1477 sbatch scripts/slurm_stage_a_4k.sh

# 5. 第三级验收: 显存阶梯（每档 20 iter, 依次通过）
STEP=32k  REPO_DIR=... DATASET_ROOT=/data/longctx sbatch scripts/slurm_stage_b_128k.sh
STEP=64k  REPO_DIR=... DATASET_ROOT=/data/longctx sbatch scripts/slurm_stage_b_128k.sh
STEP=128k REPO_DIR=... DATASET_ROOT=/data/longctx sbatch scripts/slurm_stage_b_128k.sh

# 6. 第四级验收 + 长跑: 1 → 10 → 100 → 1000 iter → save/restart 连续性
STEP=128k TRAIN_ITERS=1000 SAVE_CKPT=1 REPO_DIR=... DATASET_ROOT=... \
    sbatch scripts/slurm_stage_b_128k.sh
```

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
