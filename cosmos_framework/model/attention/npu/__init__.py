# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NPU Backend — delegates to ``F.scaled_dot_product_attention``, which
``torch_npu`` routes to CANN's Flash Attention 2.
"""

from cosmos_framework.model.attention.utils.safe_ops import log


def npu_supported() -> bool:
    """Returns whether NPU SDPA is available."""
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except ImportError:
        return False


NPU_SUPPORTED = npu_supported()

if NPU_SUPPORTED:
    from cosmos_framework.model.attention.npu.functions import (  # noqa: F401
        npu_attention,
    )
else:
    # When NPU is not available, npu_attention is never called — the frontend
    # gates on npu_attention_check which returns False when NPU_SUPPORTED is
    # False. Provide a stub that raises if reached anyway.
    from torch import Tensor

    from cosmos_framework.model.attention.masks import CausalType

    def npu_attention(  # noqa: D103
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        causal_type: CausalType | None = None,
        scale: float | None = None,
        cumulative_seqlen_Q: Tensor | None = None,
        cumulative_seqlen_KV: Tensor | None = None,
        max_seqlen_Q: int | None = None,
        max_seqlen_KV: int | None = None,
        return_lse: bool = False,
        backend_kwargs: dict | None = None,
        deterministic: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise RuntimeError(
            "Tried to run NPU attention, but torch_npu is not available. "
            "Set COSMOS_DEVICE=npu and install torch_npu."
        )