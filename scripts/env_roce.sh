#!/usr/bin/env bash
# RoCE / NCCL 环境模板 — 由各训练脚本 source。
# <> 占位符必须替换为集群真实值, 不要照抄。

# ---- 通用(官方 DSv4 recipe 同款) ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_PXN_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---- RoCE(按集群实际填写) ----
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<bootstrap_ethernet_interface>}
export NCCL_IB_HCA=${NCCL_IB_HCA:-<validated_connectx_hca_list>}

# 验收阶段开 INFO, 长跑改 WARN
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}

# 下列参数只有在网络团队给出经过验证的 RoCEv2 GID/PFC/ECN/TC/多rail 配置时才固定,
# 否则保持不设置:
#   NCCL_IB_GID_INDEX  NCCL_IB_TC  NCCL_IB_SL  NCCL_ALGO  NCCL_PROTO  NCCL_CROSS_NIC
#
# 合格判据: NCCL 日志必须显示 NET/IB (RoCE RDMA);
# 出现 NET/Socket 回退即视为环境不合格, 停止训练排查网络。
