# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NPU backend checks
"""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.npu import NPU_SUPPORTED
from cosmos_framework.model.attention.npu.meta import get_bwd_dtypes, get_fwd_dtypes
from cosmos_framework.model.attention.utils import get_arch_tag, log_or_raise_error


def npu_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """Input validation for the NPU backend.

    The NPU backend delegates to ``F.scaled_dot_product_attention``, so it
    accepts virtually all input shapes and dtypes. The only gate is NPU
    availability.
    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not NPU_SUPPORTED:
        target_fn(
            "NPU Attention is not supported in this environment. "
            "Install torch_npu and set COSMOS_DEVICE=npu.",
            exception=RuntimeError,
        )
        return False

    arch_tag = get_arch_tag(device)
    fwd_dtypes = get_fwd_dtypes(arch_tag)
    bwd_dtypes = get_bwd_dtypes(arch_tag)
    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=fwd_dtypes,
        supported_dtypes_backward=bwd_dtypes,
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="NPU (SDPA)",
    ):
        target_fn("NPU (SDPA) does not support the given inputs.", exception=RuntimeError)
        return False

    # Verify causal_type is valid when is_causal
    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    return True