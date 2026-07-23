#!/bin/bash
# 第一级验收: 网络。按 1、2、4、16 节点各提交一次:
#   sbatch -N 1  scripts/nccl_level1.sh
#   sbatch -N 2  scripts/nccl_level1.sh
#   sbatch -N 4  scripts/nccl_level1.sh
#   sbatch -N 16 scripts/nccl_level1.sh
#
# 通过标准: 每项 #wrong = 0; 无 NCCL WARN/timeout/Socket 回退;
# 各节点带宽无明显离群; GPU-NIC 亲和正确。

#SBATCH --job-name=dsv4-nccl-l1
#SBATCH --account=my_account
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/nccl_l1_%j.out
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

# 容器需带 nccl-tests(NGC PyTorch / NeMo FW 容器均含或可自行编译)
CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to your .sqsh}

source "$(dirname "$0")/env_roce.sh"

# 每卡 buffer 扫到 8GB, 覆盖 128K 训练时的大消息尺寸
ARGS="-b 8 -e 8G -f 2 -g 1"

for t in all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf; do
    echo "==== $t ($SLURM_JOB_NUM_NODES nodes) ===="
    srun --mpi=pmix --container-image="$CONTAINER_IMAGE" \
        bash -lc "$t $ARGS"
done

echo "Done. 人工检查: #wrong=0、无 WARN、日志显示 NET/IB 而非 NET/Socket。"
