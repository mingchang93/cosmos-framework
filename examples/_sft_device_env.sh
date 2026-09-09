# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Deterministic/offline-environment setup shared by the device-aware SFT launchers
# (launch_sft_vision_edge_v2.sh / _npu_v2.sh). Source AFTER setting recipe vars,
# BEFORE _sft_launcher_common.sh.
#
# Parses an optional `--device auto|cuda|npu` from the caller's argv (a sourced
# file inherits the caller's `$@`). `auto` resolves to `npu` when COSMOS_DEVICE=npu
# or ASCEND_RT_VISIBLE_DEVICES is set, else the caller's DEFAULT_DEVICE, else `cuda`.
# Exports DEVICE + COSMOS_DEVICE (read by cosmos_framework.utils.flags and
# OmniMoTModel.set_precision) plus the platform env vars below, gated by device.

# Resolve DEVICE from an optional `--device` flag (default: auto).
DEVICE="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)   DEVICE="${2:-auto}"; shift ;;
        --device=*) DEVICE="${1#*=}" ;;
    esac
    shift
done

if [[ "$DEVICE" == "auto" ]]; then
    if [[ "${COSMOS_DEVICE:-}" == "npu" || -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        DEVICE="npu"
    elif [[ -n "${DEFAULT_DEVICE:-}" ]]; then
        DEVICE="$DEFAULT_DEVICE"
    else
        DEVICE="cuda"
    fi
fi

case "$DEVICE" in
    cuda|npu) ;;
    *) echo "ERROR: unknown --device '$DEVICE' (expected auto|cuda|npu)" >&2; exit 1 ;;
esac

export DEVICE
export COSMOS_DEVICE="$DEVICE"

# Repo root = parent of this examples/ directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Common (always exported) ---
export PYTHONHASHSEED=1234
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DETERMINISTIC=TRUE
export HCCL_DETERMINISTIC=TRUE
export PYTHONPATH="$ROOT:$ROOT/reference/Emu3:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export EXTRA_TRAIN_ARGS="--deterministic"

# --- NPU-only (--device npu) ---
if [[ "$DEVICE" == "npu" ]]; then
    export INF_NAN_MODE_ENABLE=1
    export CLOSE_MATMUL_K_SHIFT=1
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    export ACL_OP_DETERMINISTIC=1
    export ASCEND_LAUNCH_BLOCKING=1
    export TASK_QUEUE_ENABLE=0
    export FLAGS_npu_storage_format=0
    export LCCL_DETERMINISTIC=1
fi

# --- CUDA-only (--device cuda) ---
if [[ "$DEVICE" == "cuda" ]]; then
    export NCCL_DEBUG=INFO
    export NCCL_BLOCKING_WAIT=1
    export NCCL_CROSS_NIC=1
fi

# --- Non-NPU only (--device != npu) ---
if [[ "$DEVICE" != "npu" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
