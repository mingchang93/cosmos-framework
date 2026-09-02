# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NPU Backend: actual attention kernels.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from cosmos_framework.model.attention.masks import CausalType


def npu_attention(
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
    """NPU attention via ``F.scaled_dot_product_attention``.

    ``torch_npu`` intercepts this call and routes it to CANN's Flash Attention 2
    implementation. The signature matches the standard backend contract so the
    frontend can dispatch here as a drop-in replacement.

    Varlen support: when ``cumulative_seqlen_Q`` / ``cumulative_seqlen_KV`` are
    provided, the sequence-packed tensor (B=1) is treated as a regular batch and
    a causal mask is constructed from the cumulative lengths.
    """

    is_varlen = cumulative_seqlen_Q is not None

    if is_varlen:
        output = _varlen_sdpa(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            scale=scale,
        )
    else:
        output = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),  # [B,S,H,D] → [B,H,S,D]
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            attn_mask=None,
            scale=scale,
            dropout_p=0.0,
            is_causal=is_causal,
        ).permute(0, 2, 1, 3)  # [B,H,S,D] → [B,S,H,D]

    if return_lse:
        lse = _compute_lse(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            scale=scale,
        )
        return output, lse

    return output


def _varlen_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    cumulative_seqlen_Q: Tensor,
    cumulative_seqlen_KV: Tensor,
    scale: float | None,
) -> Tensor:
    """SDPA with block-diagonal mask built from cumulative sequence lengths."""
    total_q = query.shape[1]
    total_kv = key.shape[1]
    device = query.device
    dtype = query.dtype

    mask = torch.full((1, 1, total_q, total_kv), float("-inf"), device=device, dtype=dtype)
    for i in range(cumulative_seqlen_Q.shape[0] - 1):
        q_start = cumulative_seqlen_Q[i].item()
        q_end = cumulative_seqlen_Q[i + 1].item()
        kv_start = cumulative_seqlen_KV[i].item()
        kv_end = cumulative_seqlen_KV[i + 1].item()
        mask[:, :, q_start:q_end, kv_start:kv_end] = 0.0

    if is_causal:
        for i in range(cumulative_seqlen_Q.shape[0] - 1):
            q_start = cumulative_seqlen_Q[i].item()
            q_end = cumulative_seqlen_Q[i + 1].item()
            kv_start = cumulative_seqlen_KV[i].item()
            kv_end = cumulative_seqlen_KV[i + 1].item()
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            causal = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            mask[:, :, q_start:q_end, kv_start:kv_end] = torch.maximum(
                mask[:, :, q_start:q_end, kv_start:kv_end], causal
            )

    return F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        attn_mask=mask,
        scale=scale,
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3)


def _compute_lse(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    """Compute logsumexp from QK^T scores for merge_attentions compatibility.

    ponytail: compute via torch.logsumexp on the raw scores. This is O(N²)
    and not efficient, but it's only invoked when ``return_lse=True``, which is
    used by the attention-merging path. Replace with a fused kernel when
    attention merging moves to the NPU backend natively.
    """
    scale = scale if scale is not None else query.shape[-1] ** -0.5

    q = query.permute(0, 2, 1, 3).float()
    k = key.permute(0, 2, 1, 3).float()

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal and cumulative_seqlen_Q is None:
        causal_mask = torch.triu(
            torch.full((scores.shape[-2], scores.shape[-1]), float("-inf"), device=scores.device),
            diagonal=1,
        )
        scores = scores + causal_mask
    elif cumulative_seqlen_Q is not None:
        for i in range(cumulative_seqlen_Q.shape[0] - 1):
            q_start = cumulative_seqlen_Q[i].item()
            q_end = cumulative_seqlen_Q[i + 1].item()
            kv_start = cumulative_seqlen_KV[i].item() if cumulative_seqlen_KV is not None else q_start
            kv_end = cumulative_seqlen_KV[i + 1].item() if cumulative_seqlen_KV is not None else q_end
            if q_start > 0:
                scores[:, :, q_start:q_end, :kv_start] = float("-inf")
            if kv_end < scores.shape[-1]:
                scores[:, :, q_start:q_end, kv_end:] = float("-inf")
            if is_causal:
                causal = torch.triu(
                    torch.full((q_end - q_start, kv_end - kv_start), float("-inf"), device=scores.device),
                    diagonal=1,
                )
                scores[:, :, q_start:q_end, kv_start:kv_end] = torch.maximum(
                    scores[:, :, q_start:q_end, kv_start:kv_end], causal
                )

    lse = torch.logsumexp(scores, dim=-1, keepdim=True).to(query.dtype)
    return lse.permute(0, 2, 1, 3)