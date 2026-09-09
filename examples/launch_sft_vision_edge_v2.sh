#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_edge (T2V / I2V / V2V vision-only
# SFT on Nemotron-2B-Dense-VL / Cosmos3-Edge, 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_edge.toml.
#
# Device-aware counterpart of launch_sft_vision_edge.sh (defaults to CUDA;
# override with `--device npu`). Exports the deterministic env vars via
# _sft_device_env.sh and routes model.config.device_type from the resolved DEVICE.
#
# Required env vars (no defaults — export before running):
#   DATASET_PATH          dataset dir (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  base DCP checkpoint dir
#   WAN_VAE_PATH          Wan2.2 VAE checkpoint (.pth)
#   OUTPUT_ROOT           output/log directory
#   HF_TOKEN              not needed for nvidia/Cosmos3-Edge (the repo is
#                         ungated); set only if another download requires it
#
# Optional env vars:
#   MAX_ITER              trainer.max_iter override (default: TOML's 500)
#   SHUFFLE               dataset shuffle (default: True; set False to disable)
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   export DATASET_PATH=... BASE_CHECKPOINT_PATH=... WAN_VAE_PATH=... OUTPUT_ROOT=...
#   bash examples/launch_sft_vision_edge_v2.sh [--device auto|cuda|npu]

DEFAULT_DEVICE=cuda

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"

# Mandatory inputs (no defaults).
: "${DATASET_PATH:?DATASET_PATH is required (no default)}"
: "${BASE_CHECKPOINT_PATH:?BASE_CHECKPOINT_PATH is required (no default)}"
: "${WAN_VAE_PATH:?WAN_VAE_PATH is required (no default)}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required (no default)}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_device_env.sh"

TAIL_OVERRIDES=("model.config.device_type=$DEVICE")
if [[ -n "${MAX_ITER:-}" ]]; then
    TAIL_OVERRIDES+=("trainer.max_iter=$MAX_ITER")
fi
if [[ -n "${SHUFFLE:-}" ]]; then
    TAIL_OVERRIDES+=("dataloader_train.dataloader.datasets.video.dataset.shuffle=$SHUFFLE")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
