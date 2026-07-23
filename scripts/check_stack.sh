#!/usr/bin/env bash
# 软件栈校验 — 在训练容器内、任何训练任务之前运行。
# 校验三件事:
#   1. Megatron-LM 子模块处于 Bridge .dev.commit 固定的 dev commit
#   2. DSv4 CP 代码在位(#5087 引入的 csa_cp_utils 可导入)
#   3. 必需依赖 fast_hadamard_transform 在位
set -euo pipefail

BRIDGE_ROOT=${BRIDGE_ROOT:-/opt/Megatron-Bridge}
cd "$BRIDGE_ROOT"

BRIDGE_COMMIT=$(git rev-parse HEAD)
MCORE_COMMIT=$(git -C 3rdparty/Megatron-LM rev-parse HEAD)
DEV_PIN=$(tr -d '[:space:]' < .dev.commit 2>/dev/null || echo "<missing .dev.commit>")

echo "Megatron-Bridge commit : $BRIDGE_COMMIT"
echo "Megatron-LM     commit : $MCORE_COMMIT"
echo "Bridge .dev.commit pin : $DEV_PIN"

if [ "$MCORE_COMMIT" != "$DEV_PIN" ]; then
    echo "FAIL: Megatron-LM 未处于 dev pin。先执行:"
    echo "      ./scripts/switch_mcore.sh dev && uv sync"
    exit 1
fi

python - <<'PY'
import importlib
import torch
import transformer_engine

print(f"torch              : {torch.__version__}")
print(f"transformer_engine : {transformer_engine.__version__}")

# DSv4 CP 支持的直接证据 (Megatron-LM #5087)
importlib.import_module("megatron.core.transformer.experimental_attention_variant.csa_cp_utils")
print("OK: csa_cp_utils 可导入 -> DSv4 Context Parallel 代码在位")

# DSA attention 硬依赖 (无 PyTorch fallback)
import fast_hadamard_transform  # noqa: F401
print("OK: fast_hadamard_transform 在位")
PY

echo "PASS: 软件栈校验通过。把上述 commit/版本随训练日志一起归档。"
