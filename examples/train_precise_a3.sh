#!/usr/bin/env bash
# train_precise_a3.sh — A3 (NPU) 容器内确定性精度训练脚本
# 用法: bash /workspace/train_precise_a3.sh [RUN_NAME] [MAX_ITER]
#   RUN_NAME — 可选运行名, 默认 precise_时间戳
#   MAX_ITER — 可选迭代步数, 默认 10
#
# 前置: 调用 check_determinism.py 检查确定性配置, FAIL 则中止
# 输出: /workspace/logs/<RUN_NAME>/logs/vision_sft_edge_sft.log
#       /workspace/logs/<RUN_NAME>/check_determinism.json

set -uo pipefail

RUN_NAME="${1:-precise_$(date +%Y%m%d_%H%M%S)}"
MAX_ITER="${2:-10}"

# === 路径 ===
WORKDIR="/workspace/cosmos-framework-npu-port"
DATASET_PATH="/workspace/datasets/sft_dataset_bridge"
CHECKPOINT="/data2/models/Cosmos3-Edge"
VAE="${CHECKPOINT}/vae"
HF_HOME="/data2/hf_cache"
OUTPUT_ROOT="/workspace/logs/${RUN_NAME}"

# === 确定性环境变量 ===
export COSMOS_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_TIMEOUT=7200
export OMP_NUM_THREADS=8
export MASTER_PORT="${MASTER_PORT:-50120}"

# === 清理残留 ===
pkill -9 -f "cosmos_framework.scripts.train" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
sleep 2

# === 创建输出目录 ===
mkdir -p "${OUTPUT_ROOT}/logs"

# === 启动训练 ===
cd "${WORKDIR}"

TORCHRUN_ARGS=(
    --nproc_per_node=8
    --master_port="${MASTER_PORT}"
)

IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
PYTHONPATH=. \
torchrun "${TORCHRUN_ARGS[@]}" \
    -m cosmos_framework.scripts.train \
    --sft-toml="examples/toml/sft_config/vision_sft_edge.toml" \
    -- \
    model.config.device_type=npu \
    data_setting.dataset_path="${DATASET_PATH}" \
    checkpoint.load_path="${CHECKPOINT}" \
    checkpoint.save_path="${OUTPUT_ROOT}/checkpoints" \
    trainer.max_iter="${MAX_ITER}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/vision_sft_edge_sft.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> Done (exit $EXIT_CODE)"
exit $EXIT_CODE
