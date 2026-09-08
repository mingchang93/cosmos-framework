# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
)

pytestmark = pytest.mark.L0


def test_rotary_init_weights_reconstructs_nonpersistent_buffers() -> None:
    """init_weights must rebuild BOTH inv_freq and original_inv_freq.

    Regression guard for the NPU loss-gap root cause: `to_empty` materializes
    persistent=False buffers as uninitialized garbage, and they are not in the
    checkpoint, so `init_weights(buffer_device=...)` is their only
    reconstruction point. It must cover every buffer `__init__` registered.
    """
    rotary = MultiModalRotaryEmbedding(Nemotron3DenseVLTextConfig())
    device = torch.device("cpu")

    # Simulate the to_empty() garbage state, then reconstruct.
    rotary.inv_freq.fill_(float("nan"))
    rotary.original_inv_freq.fill_(float("nan"))
    rotary.init_weights(buffer_device=device)

    assert rotary.inv_freq.device == device
    assert rotary.original_inv_freq.device == device
    assert torch.isfinite(rotary.inv_freq).all()
    assert torch.isfinite(rotary.original_inv_freq).all()
    torch.testing.assert_close(rotary.original_inv_freq, rotary.inv_freq, rtol=0, atol=0)
    # Non-persistent: excluded from state_dict, so init_weights is the sole reconstructor.
    assert "inv_freq" not in rotary.state_dict()
    assert "original_inv_freq" not in rotary.state_dict()
