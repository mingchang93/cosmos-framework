# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NPU Backend: actual attention kernels.
"""

import torch
import torch_npu
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
    q_tnd = query.reshape(-1, query.shape[2], query.shape[3])
    k_tnd = key.reshape(-1, key.shape[2], key.shape[3])
    v_tnd = value.reshape(-1, value.shape[2], value.shape[3])
    if q_tnd.shape[1] != k_tnd.shape[1]:
        rep = q_tnd.shape[1] // k_tnd.shape[1]
        k_tnd = k_tnd.repeat_interleave(rep, dim=1)
        v_tnd = v_tnd.repeat_interleave(rep, dim=1)
    actual_seq_qlen = cumulative_seqlen_Q[1:].tolist()
    actual_seq_kvlen = cumulative_seqlen_KV[1:].tolist()
    _scale = scale if scale is not None else 1.0

    # Causal varlen: sparse_mode=2 (leftUpCausal) REQUIRES an atten_mask — with
    # no mask CANN treats it as "full computation" and silently drops causality.
    # sparse_mode=2/3/4 use CANN's "attenmask compression" path, which needs a
    # fixed 2048x2048 compressed lower-triangular mask (mask size is constant,
    # not sequence-dependent); the kernel applies it causally within each
    # sequence block, guided by actual_seq_qlen/actual_seq_kvlen.
    sparse_mode = 2 if is_causal else 0
    atten_mask = None
    if is_causal:
        atten_mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=query.device),
            diagonal=1,
        )

    output = torch_npu.npu_fusion_attention(
        q_tnd, k_tnd, v_tnd, q_tnd.shape[1],
        input_layout='TND',
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        scale=_scale, keep_prob=1.0, sparse_mode=sparse_mode,
        atten_mask=atten_mask,
    )
    return output[0].reshape(query.shape[0], query.shape[1], query.shape[2], query.shape[3])

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