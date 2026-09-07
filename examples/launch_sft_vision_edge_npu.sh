#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_edge on NPU (T2V / I2V / V2V vision-only
# SFT on Nemotron-2B-Dense-VL / Cosmos3-Edge, 8-NPU FSDP). Drives
# cosmos_framework.scripts.train against examples/toml/sft_config/vision_sft_edge.toml.
#
# NPU counterpart of launch_sft_vision_edge.sh. Uses HCCL backend via
# COSMOS_DEVICE=npu and the npu attention backend (SDPA → CANN Flash Attn 2).
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
#                         (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              not needed for nvidia/Cosmos3-Edge (the repo is
#                         ungated); set only if another download requires it
#   OUTPUT_ROOT           default: outputs/train
#   NPROC_PER_NODE        default: 8 (NPU count)
#
# Usage (8-NPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_vision_edge_npu.sh

export COSMOS_DEVICE=npu
# NPU设备可见性: 4卡8die场景须设为0-7, 否则device_count只返回4个物理卡
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=8}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

# NPU: use the npu attention backend (sdpa → CANN Flash Attn 2), disable CUDA
# graphs, and route distributed backend through HCCL.
TAIL_OVERRIDES=(
    "model.config.device_type=npu"
    "trainer.max_iter=1"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"