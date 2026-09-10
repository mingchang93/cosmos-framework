#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Check TimestepEmbedder._timestep_frequencies vs a CPU golden through init/move.

The training run shows this buffer as garbage on NPU (min=-0.045, max=0.028,
mean~0) vs correct on CUDA ([0.000107, 1.0]). This steps through the likely
construction/move paths to isolate where the deterministic exp-decay table gets
corrupted. A large max_err on one row pinpoints the mechanism; all-clean rows
mean the corruption happens later (full-model FSDP/checkpoint load).

Usage:
    python3 tools/npu_debug/probe_frequencies.py
"""

import argparse
import math

import torch

from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder


def _device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _golden(dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    return torch.tensor(
        [math.exp(-math.log(max_period) * k / half) for k in range(half)], dtype=torch.float32
    )  # [half]


def _check(label: str, buf: torch.Tensor, golden: torch.Tensor) -> None:
    if buf.device.type == "meta":
        print(f"{label:48s} <meta tensor, no data>")
        return
    buf = buf.detach().cpu().float()
    err = (buf - golden).abs().max().item()
    print(
        f"{label:48s} min={buf.min().item():+.6f} max={buf.max().item():+.6f} "
        f"mean={buf.mean().item():+.6f} max_err={err:.3e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--freq-emb-size", type=int, default=256)
    args = ap.parse_args()

    dev = _device()
    golden = _golden(args.freq_emb_size)
    print(f"golden: min={golden.min().item():.6f} max={golden.max().item():.6f}  device={dev}")

    # 1. _build_timestep_frequencies directly, various device args.
    for d in [None, torch.device("cpu"), dev, torch.device("meta")]:
        freqs = TimestepEmbedder._build_timestep_frequencies(args.freq_emb_size, 10000, device=d)
        _check(f"_build_timestep_frequencies(device={d})", freqs, golden)

    # 2. Through __init__ (CPU), then cast/move to mimic model construction.
    emb = TimestepEmbedder(args.hidden_size, frequency_embedding_size=args.freq_emb_size)
    _check("__init__ on CPU", emb._timestep_frequencies, golden)
    emb = emb.to(device=dev)  # move first (fp32)
    _check(f".to({dev})", emb._timestep_frequencies, golden)
    emb = emb.to(dtype=torch.bfloat16)  # then bf16
    _check(f".to(bf16)", emb._timestep_frequencies, golden)

    # 3. _init_weights with the real-device buffer (mimic the model's init_weights).
    emb2 = TimestepEmbedder(args.hidden_size, frequency_embedding_size=args.freq_emb_size)
    emb2._init_weights(buffer_device=dev)
    _check(f"_init_weights(device={dev})", emb2._timestep_frequencies, golden)

    # 4. _init_weights on meta (FSDP lazy-init hypothesis), then materialize to the device.
    emb3 = TimestepEmbedder(args.hidden_size, frequency_embedding_size=args.freq_emb_size)
    emb3._init_weights(buffer_device=torch.device("meta"))
    try:
        b = emb3._timestep_frequencies.to(device=dev)  # materialize meta -> real
        _check("_init_weights(meta) then .to(dev)", b, golden)
    except Exception as e:  # noqa: BLE001
        print(f"_init_weights(meta) then .to(dev): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
