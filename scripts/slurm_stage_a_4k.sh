#!/bin/bash
# 阶段 A(保底冒烟): DeepSeek-V4-Flash 全参 SFT, 4K / SBHD,
# TP1 PP4 CP1 EP8 -> DP=32, 16 节点 x 8 H100。
#
# 这是官方已验证配置的直接放大, 目的是在进入 128K 之前验证:
# 权重导入、软件栈(dev pin)、RoCE 16 节点通信、PP/EP 拓扑、loss 正常下降。
# 默认 50 iter 冒烟, 不存 checkpoint。
#
#   sbatch scripts/slurm_stage_a_4k.sh

#SBATCH --job-name=dsv4-stage-a-4k
#SBATCH --account=my_account
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/dsv4_stage_a_%j.out
#SBATCH --error=logs/dsv4_stage_a_%j.err
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

# ---- 路径 ----
WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:?本仓库在容器内的挂载路径, 例如 /workspace/harvey1477}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash}
MEGATRON_DIR=${MEGATRON_DIR:-${WORKSPACE}/models/${MODEL_VARIANT}}
PRETRAINED_CKPT="${MEGATRON_DIR}/iter_0000000"   # scripts/slurm_import.sh 的产物

# ---- 运行参数 ----
RECIPE_NAME=dsv4_flash_sft_h100_128gpu_4k_baseline_config
DATASET=${DATASET:-squad}          # squad | gsm8k | local-jsonl(另设 DATASET_ROOT)
DATASET_ROOT=${DATASET_ROOT:-}
TRAIN_ITERS=${TRAIN_ITERS:-50}
LR=${LR:-5e-6}
LOG_INTERVAL=1

CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to your .sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}   # 必须包含 REPO_DIR、MEGATRON_DIR 所在的共享存储

# ---- 拓扑 ----
NNODES=${SLURM_JOB_NUM_NODES:-16}
NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29571}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
[ -n "$CONTAINER_MOUNTS" ] && SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"

DATASET_ARGS="--dataset $DATASET"
[ -n "$DATASET_ROOT" ] && DATASET_ARGS="$DATASET_ARGS dataset.dataset_root=$DATASET_ROOT"

echo "=== DSv4-Flash Stage A: 4K baseline, $NNODES x $NPROC_PER_NODE GPU, recipe=$RECIPE_NAME ==="

$SRUN_CMD bash -lc "
    set -euo pipefail
    source $REPO_DIR/scripts/env_roce.sh
    export NCCL_DEBUG=WARN
    cd /opt/Megatron-Bridge
    uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=\$SLURM_PROCID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        $REPO_DIR/scripts/run_dsv4_recipe.py \
        --recipe $RECIPE_NAME \
        --step_func gpt_step \
        $DATASET_ARGS \
        train.train_iters=$TRAIN_ITERS \
        optimizer.lr=$LR \
        scheduler.lr_warmup_iters=10 \
        scheduler.lr_decay_iters=$TRAIN_ITERS \
        validation.eval_iters=2 \
        dataset.do_test=false \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CKPT \
        checkpoint.save=null checkpoint.load=null \
        logger.log_interval=$LOG_INTERVAL
"

echo "=== Stage A 完成。验收: loss 有限且下降、无 NaN、NCCL 无 WARN/Socket 回退 ==="
