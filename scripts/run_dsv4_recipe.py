#!/usr/bin/env python3
"""注册本仓库的 DSv4 H100 recipe, 然后转调官方 run_recipe.py。

官方 scripts/training/run_recipe.py 的 --recipe 只从
megatron.bridge.recipes.<family>.h100 命名空间解析 recipe 函数名。
本入口在调用前把 recipes/dsv4_flash_h100_128gpu.py 中的配置注入
megatron.bridge.recipes.deepseek.h100, 之后所有行为(数据集选择、CLI 覆盖、
pipeline layout 同步、CP/打包不变量校验、训练循环)与官方入口完全一致。

用法(与官方 run_recipe.py 相同, 仅 --recipe 可选本仓库的名字):

  torchrun --nnodes=16 --nproc-per-node=8 ... \
      scripts/run_dsv4_recipe.py \
      --recipe dsv4_flash_sft_h100_128gpu_128k_cp16_config \
      --step_func gpt_step \
      --dataset local-jsonl dataset.dataset_root=/data/longctx \
      dataset.enable_offline_packing=true \
      checkpoint.pretrained_checkpoint=/workspace/models/DeepSeek-V4-Flash/iter_0000000

环境变量:
  BRIDGE_ROOT  Megatron-Bridge 根目录(默认 /opt/Megatron-Bridge)
"""

import importlib
import os
import sys

BRIDGE_ROOT = os.environ.get("BRIDGE_ROOT", "/opt/Megatron-Bridge")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 官方入口脚本目录(run_recipe.py / recipe_runner.py / recipe_metadata.py)
sys.path.insert(0, os.path.join(BRIDGE_ROOT, "scripts", "training"))
# 本仓库根目录(recipes/ 包)
sys.path.insert(0, REPO_ROOT)

import recipes.dsv4_flash_h100_128gpu as local_recipes  # noqa: E402

# 注入官方解析命名空间, 使 --recipe <name> 可以找到本仓库的配置函数。
_h100_pkg = importlib.import_module("megatron.bridge.recipes.deepseek.h100")
for _name in local_recipes.__all__:
    setattr(_h100_pkg, _name, getattr(local_recipes, _name))

import run_recipe  # noqa: E402  # 官方 scripts/training/run_recipe.py

if __name__ == "__main__":
    run_recipe.main()
