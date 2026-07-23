# DeepSeek-V4-Flash full-parameter SFT recipes for 128 x H100-80GB (16 nodes x 8), RoCE.
#
# 全部派生自官方 recipe `deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config`
# (Megatron-Bridge: src/megatron/bridge/recipes/deepseek/h100/deepseek_v4.py),
# 只覆盖并行拓扑、序列长度与 DSv4 CP 必需字段。其余全部保持官方默认值:
# bf16/Adam、selective recompute(["moe_act","mhc"])、alltoall dispatcher、
# dsa_indexer_loss_coeff=0、H100 上自动关闭 fused-mHC、CUDA Graph 关闭。
#
# 软件栈要求: Megatron-LM 必须处于 Bridge `.dev.commit` 固定的 dev commit
# (bfa33263ca06e6974410d0ea871b25e21c5aee85, 含 #5011 THD 与 #5087 DSv4 CP)。
# 先跑 scripts/check_stack.sh 校验, 再提交任何训练任务。

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config,
)


WORLD_SIZE = 128  # 16 nodes x 8 H100-80GB


def _dsv4_flash_sft_h100_128gpu(pp: int, cp: int, seq_length: int):
    """公共派生逻辑: 只改拓扑/序列/CP 字段; 改 PP 后必须重算 pipeline layout。"""
    cfg = deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config()

    assert WORLD_SIZE % (pp * cp) == 0, "PP*CP 必须整除 128"
    cfg.model.tensor_model_parallel_size = 1  # DSv4 hybrid attention 硬性要求 TP=1
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.context_parallel_size = cp
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length

    if cp > 1:
        # DSv4 hybrid attention 的 CP 硬性要求 (Megatron-LM #5087):
        # THD(packed) 输入 + contiguous 切分 + 非空打包调度器。
        # 注意: 数据集必须走打包路径(启动脚本传 dataset.enable_offline_packing=true)。
        cfg.model.cp_partition_mode = "contiguous"
        cfg.model.sequence_packing_scheduler = "dp_balanced"
        assert seq_length % cp == 0, "seq_length 必须能被 CP 整除"
        cfg.model.max_seqlen_per_dp_cp_rank = seq_length // cp

    # 官方 recipe 在函数体内按 PP=4 计算 DSv4 pipeline layout;
    # 任何 PP 改动之后都必须重算, 否则 stage 层排布是错的。
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    cfg.train.micro_batch_size = 1
    # 首轮不落 optimizer state: 285B 模型的分布式 Adam 全量存档是数 TB 级。
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.load_optim = False
    return cfg


def dsv4_flash_sft_h100_128gpu_4k_baseline_config():
    """阶段 A(保底): 4K / SBHD / TP1 PP4 CP1 EP8 -> DP=32。

    官方已端到端验证路径(bf16/Adam/4K, H100 需 >=64 卡使 fp32 master 分片)
    的直接放大。用途: 验证权重导入、软件栈、RoCE 16 节点通信、PP/EP 拓扑、
    loss 正常下降 —— 与长上下文无关的一切。必须最先跑通。
    """
    cfg = _dsv4_flash_sft_h100_128gpu(pp=4, cp=1, seq_length=4096)
    cfg.train.global_batch_size = 128  # DP=32, 每 rank 4 步梯度累积
    return cfg


def dsv4_flash_sft_h100_128gpu_32k_cp4_config():
    """显存阶梯 1/3: 32K / CP4 / PP8 -> DP=4。每卡本地序列 8192。"""
    cfg = _dsv4_flash_sft_h100_128gpu(pp=8, cp=4, seq_length=32768)
    cfg.train.global_batch_size = 4
    return cfg


def dsv4_flash_sft_h100_128gpu_64k_cp8_config():
    """显存阶梯 2/3: 64K / CP8 / PP8 -> DP=2。每卡本地序列 8192。"""
    cfg = _dsv4_flash_sft_h100_128gpu(pp=8, cp=8, seq_length=65536)
    cfg.train.global_batch_size = 2
    return cfg


def dsv4_flash_sft_h100_128gpu_128k_cp16_config():
    """目标配置(阶段 B): 128K / CP16 / PP8 -> DP=1。每卡本地序列 8192。

    分布式优化器在 DP x CP = 16 个 rank 上分片 fp32 master 权重。
    DP=1 时 global_batch_size 即梯度累积步数, 稳定后按吞吐提高。
    """
    cfg = _dsv4_flash_sft_h100_128gpu(pp=8, cp=16, seq_length=131072)
    cfg.train.global_batch_size = 1
    return cfg


def dsv4_flash_sft_h100_128gpu_128k_cp32_fallback_config():
    """显存保底: 128K / CP32 / PP4 -> DP=1。每卡本地序列 4096 + full recompute。

    仅当 CP16 峰值显存 > 72 GiB 且 full recompute 仍不够时使用;
    CP group 横跨 4 节点, RoCE 压力显著增大。
    """
    cfg = _dsv4_flash_sft_h100_128gpu(pp=4, cp=32, seq_length=131072)
    cfg.train.global_batch_size = 1
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    return cfg


__all__ = [
    "dsv4_flash_sft_h100_128gpu_4k_baseline_config",
    "dsv4_flash_sft_h100_128gpu_32k_cp4_config",
    "dsv4_flash_sft_h100_128gpu_64k_cp8_config",
    "dsv4_flash_sft_h100_128gpu_128k_cp16_config",
    "dsv4_flash_sft_h100_128gpu_128k_cp32_fallback_config",
]
