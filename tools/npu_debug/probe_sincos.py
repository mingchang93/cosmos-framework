#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Probe torch.cos / torch.sin on the current device vs a CPU float64 golden.

Confirms the timestep-embedding root cause: the Ascend transcendental unit
returning wrong values. cos/sin of args in [0, 0.9] (the embedder's real range)
must be >= 0 — a negative min is the bug. Tests plain fp32, the actual
autocast(dtype=float32) path, and fp64 (the candidate fix).

Usage:
    python3 tools/npu_debug/probe_sincos.py
"""

import argparse

import torch


def _device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _report(name: str, dev_val: torch.Tensor, ref_val: torch.Tensor) -> None:
    dev_val = dev_val.cpu().double()
    ref_val = ref_val.cpu().double()
    err = (dev_val - ref_val).abs()
    print(
        f"  {name:24s} max_abs_err={err.max().item():.3e}  mean_abs_err={err.mean().item():.3e} "
        f"min={dev_val.min().item():.6f}  max={dev_val.max().item():.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = _device()
    gen = torch.Generator().manual_seed(args.seed)

    for name, hi in [("narrow [0, 0.9]", 0.9), ("broad [0, 2pi]", 6.283185307179586)]:
        print(f"\n=== {name}  (device={dev}) ===")
        angles = torch.rand(args.n, generator=gen) * hi  # [n], non-negative

        ref_cos = torch.cos(angles.double())
        ref_sin = torch.sin(angles.double())

        a_fp32 = angles.to(dev, torch.float32)
        _report("cos fp32", torch.cos(a_fp32), ref_cos.float())
        _report("sin fp32", torch.sin(a_fp32), ref_sin.float())

        if dev.type in ("npu", "cuda"):
            try:
                with torch.autocast(dev.type, enabled=True, dtype=torch.float32):
                    c = torch.cos(a_fp32)
                    s = torch.sin(a_fp32)
                _report("cos fp32 (autocast)", c, ref_cos.float())
                _report("sin fp32 (autocast)", s, ref_sin.float())
            except Exception as e:  # noqa: BLE001 — report and continue
                print(f"  autocast unavailable: {e}")

        a_fp64 = angles.to(dev, torch.float64)
        _report("cos fp64", torch.cos(a_fp64), ref_cos)
        _report("sin fp64", torch.sin(a_fp64), ref_sin)


if __name__ == "__main__":
    main()
