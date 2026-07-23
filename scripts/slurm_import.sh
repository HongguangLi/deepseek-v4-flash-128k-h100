#!/bin/bash
# Phase 0: HF -> Megatron bf16 权重导入(训练前只需做一次)。
#
# 官方在 H100-80GB 上验证过的 Flash 导入拓扑是 TP1/PP1/EP16 (16 GPU = 2 节点),
# 见 examples/models/deepseek_v4/README.md "Parallelism Configurations"。
# 导入产物是 torch_dist checkpoint, 训练时可 re-shard 到 PP4/PP8 等拓扑。
#
# 磁盘规划(README): HF 缓存 ~150-200GB(FP8/MXFP4), bf16 导入产物 ~570GB。
# HF_HOME 与 MEGATRON_DIR 都放共享存储。
#
#   sbatch scripts/slurm_import.sh

#SBATCH --job-name=dsv4-import
#SBATCH --account=my_account
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/dsv4_import_%j.out
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash}
HF_MODEL_ID="deepseek-ai/${MODEL_VARIANT}"
MEGATRON_DIR=${MEGATRON_DIR:-${WORKSPACE}/models/${MODEL_VARIANT}}

TP=1; PP=1; EP=16   # H100 官方验证过的导入拓扑

CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to your .sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
# export HF_HOME=/path/to/shared/HF_HOME   # 必须共享存储, 缓存 ~200GB 下载
# export HF_TOKEN=hf_xxx                   # 仓库 gated 时需要

NNODES=${SLURM_JOB_NUM_NODES:-2}
MASTER_PORT=${MASTER_PORT:-29571}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
[ -n "$CONTAINER_MOUNTS" ] && SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"

if [ -e "${MEGATRON_DIR}/iter_0000000/.metadata" ]; then
    echo "导入产物已存在: ${MEGATRON_DIR}/iter_0000000 — 跳过。"
    exit 0
fi

$SRUN_CMD bash -lc "
    set -euo pipefail
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    cd /opt/Megatron-Bridge
    uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node=8 --nnodes=$NNODES --node_rank=\$SLURM_PROCID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/conversion/run_conversion.py import \
        --device gpu \
        --hf-model '$HF_MODEL_ID' \
        --megatron-path '$MEGATRON_DIR' \
        --tp $TP --pp $PP --ep $EP \
        --torch-dtype bfloat16 \
        --trust-remote-code
"
echo "导入完成: ${MEGATRON_DIR}/iter_0000000"
