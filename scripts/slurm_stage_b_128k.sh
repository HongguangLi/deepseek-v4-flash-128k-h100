#!/bin/bash
# 阶段 B: 长上下文显存阶梯 + 128K 目标训练。16 节点 x 8 H100。
#
# 用 STEP 选择阶梯档位(依次通过, 不可跳级):
#   STEP=32k   -> 32K  / CP4  / PP8  (每卡本地序列 8192)
#   STEP=64k   -> 64K  / CP8  / PP8  (每卡本地序列 8192)
#   STEP=128k  -> 128K / CP16 / PP8  (每卡本地序列 8192)  <- 目标配置
#   STEP=cp32  -> 128K / CP32 / PP4 + full recompute      <- 仅显存保底
#
# 128K 档验收: 每卡 max_memory_allocated <= 72GiB; reserved 不持续增长;
# 无 OOM/NaN/Inf; 日志确认 THD + contiguous CP 切分生效。
#
#   STEP=32k  sbatch scripts/slurm_stage_b_128k.sh
#   STEP=128k TRAIN_ITERS=1000 SAVE_CKPT=1 sbatch scripts/slurm_stage_b_128k.sh

#SBATCH --job-name=dsv4-stage-b
#SBATCH --account=my_account
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/dsv4_stage_b_%j.out
#SBATCH --error=logs/dsv4_stage_b_%j.err
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

# ---- 阶梯档位 ----
STEP=${STEP:-128k}
case "$STEP" in
    32k)  RECIPE_NAME=dsv4_flash_sft_h100_128gpu_32k_cp4_config ;;
    64k)  RECIPE_NAME=dsv4_flash_sft_h100_128gpu_64k_cp8_config ;;
    128k) RECIPE_NAME=dsv4_flash_sft_h100_128gpu_128k_cp16_config ;;
    cp32) RECIPE_NAME=dsv4_flash_sft_h100_128gpu_128k_cp32_fallback_config ;;
    *) echo "ERROR: STEP 必须是 32k|64k|128k|cp32"; exit 1 ;;
esac

# ---- 路径 ----
WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:?本仓库在容器内的挂载路径}
MODEL_VARIANT=${MODEL_VARIANT:-DeepSeek-V4-Flash}
MEGATRON_DIR=${MEGATRON_DIR:-${WORKSPACE}/models/${MODEL_VARIANT}}
PRETRAINED_CKPT="${MEGATRON_DIR}/iter_0000000"

# ---- 数据: CP>1 必须走打包(THD)路径 ----
# 长上下文 JSONL 数据目录(客户自备); 阶梯冒烟也可用 DATASET=squad(短样本会被打包拼满)
DATASET=${DATASET:-local-jsonl}
DATASET_ROOT=${DATASET_ROOT:-}
if [ "$DATASET" = "local-jsonl" ] && [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET=local-jsonl 需要设置 DATASET_ROOT"; exit 1
fi

# ---- 运行参数 ----
TRAIN_ITERS=${TRAIN_ITERS:-20}       # 阶梯冒烟默认 20 iter; 稳定性阶段调到 1000
LR=${LR:-5e-6}
SAVE_CKPT=${SAVE_CKPT:-0}            # 长跑时置 1(每个 checkpoint ~570GB, 只保留最新)
SAVE_INTERVAL=${SAVE_INTERVAL:-300}
LOG_INTERVAL=1

CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to your .sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}

# ---- 拓扑 ----
NNODES=${SLURM_JOB_NUM_NODES:-16}
NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29571}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
[ -n "$CONTAINER_MOUNTS" ] && SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"

DATASET_ARGS="--dataset $DATASET"
[ -n "$DATASET_ROOT" ] && DATASET_ARGS="$DATASET_ARGS dataset.dataset_root=$DATASET_ROOT"

RUN_NAME=dsv4_flash_${STEP}_${SLURM_JOB_ID:-manual}
if [ "$SAVE_CKPT" = "1" ]; then
    SAVE_OVERRIDES="checkpoint.save=${WORKSPACE}/results/${RUN_NAME}/checkpoints checkpoint.save_interval=$SAVE_INTERVAL checkpoint.most_recent_k=1"
else
    SAVE_OVERRIDES="checkpoint.save=null checkpoint.load=null"
fi

echo "=== DSv4-Flash Stage B: STEP=$STEP recipe=$RECIPE_NAME iters=$TRAIN_ITERS ==="

$SRUN_CMD bash -lc "
    set -euo pipefail
    source $REPO_DIR/scripts/env_roce.sh
    cd /opt/Megatron-Bridge
    uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=\$SLURM_PROCID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        $REPO_DIR/scripts/run_dsv4_recipe.py \
        --recipe $RECIPE_NAME \
        --step_func gpt_step \
        $DATASET_ARGS \
        dataset.enable_offline_packing=true \
        train.train_iters=$TRAIN_ITERS \
        optimizer.lr=$LR \
        scheduler.lr_warmup_iters=10 \
        scheduler.lr_decay_iters=$TRAIN_ITERS \
        validation.eval_iters=2 \
        dataset.do_test=false \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CKPT \
        checkpoint.load_optim=false \
        $SAVE_OVERRIDES \
        logger.log_interval=$LOG_INTERVAL
"

echo "=== Stage B ($STEP) 完成。检查每卡峰值显存(<=72GiB)、无 NaN、CP 切分日志 ==="
