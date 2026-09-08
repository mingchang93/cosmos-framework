#!/usr/bin/env bash
# convert_checkpoint_to_dcp_npu.sh — Convert HF safetensors checkpoint to DCP format on A3 NPU
#
# Usage: bash examples/convert_checkpoint_to_dcp_npu.sh [HF_MODEL_PATH] [OUTPUT_PATH]
#   HF_MODEL_PATH  default: /data2/models/Cosmos3-Edge
#   OUTPUT_PATH    default: examples/checkpoints/Cosmos3-Edge
#
# Prerequisites:
#   - Container running with cosmos-venv activated
#   - HF model at /data2/models/Cosmos3-Edge (safetensors format)
#   - Enough disk space (~6GB for 2B model DCP output)

set -euo pipefail

HF_MODEL_PATH="${1:-/data2/models/Cosmos3-Edge}"
OUTPUT_PATH="${2:-examples/checkpoints/Cosmos3-Edge}"

echo ">>> Converting HF safetensors to DCP format"
echo ">>> Source: $HF_MODEL_PATH"
echo ">>> Output: $OUTPUT_PATH"

# Use CPU only (no NPU needed for conversion)
export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES=""

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHONPATH=. python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint.path="$HF_MODEL_PATH" \
    -o "$OUTPUT_PATH" \
    2>&1 | tee -a "${OUTPUT_PATH}/convert.log"

echo ">>> Done. DCP checkpoint at: $OUTPUT_PATH/model/"
echo ">>> Files:"
find "$OUTPUT_PATH/model/" -type f -name '*.distcp' -o -name 'config.json' -o -name '.metadata' | sort
