# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NPU Backend: dtype metadata.
"""

import torch


def get_fwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """Return supported forward dtypes for the NPU backend.

    NPU supports all standard float dtypes via SDPA.
    """
    return [torch.float16, torch.bfloat16, torch.float32]


def get_bwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """Return supported backward dtypes for the NPU backend."""
    return [torch.float16, torch.bfloat16, torch.float32]