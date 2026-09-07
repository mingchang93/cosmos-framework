#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Probe ``torch._grouped_mm`` numerics against a float32 CPU reference.

Diagnostic for the NPU-vs-GPU loss gap (Cosmos3-Edge SFT). The gen-tower MoE
expert GEMM is ``torch._grouped_mm`` (see ``moe.py:_run_experts_grouped_mm``),
the largest matmul in the model and a private PyTorch op with no device branch.
This script runs the same grouped matmul three ways on identical inputs and
reports how far each drifts from the float32 ground truth:

  1. reference : per-expert matmul, CPU, float32     (golden)
  2. naive     : per-expert matmul, accelerator, bf16  (standard matmul path)
  3. grouped   : torch._grouped_mm, accelerator, bf16  (the suspect)

Interpretation: if ``grouped`` ~= ``naive`` (both bf16), the drift is just normal
bf16 noise and grouped_mm is NOT the cause. If ``grouped`` drifts from ``naive``
by more than the bf16 floor (~4e-3), torch._grouped_mm is numerically broken on
this device.

Run on both the NPU and GPU boxes and compare the ``grouped vs naive`` row:

    python3 tools/npu_debug/probe_grouped_mm.py            # bf16, model-ish sizes
    python3 tools/npu_debug/probe_grouped_mm.py --dtype float32
"""

import argparse

import torch


def _device() -> torch.device:
    if torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _grouped_mm(x: torch.Tensor, w: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    # Mirrors moe.py:_run_experts_grouped_mm's exact call (offs = cumsum, int32).
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    return torch._grouped_mm(x, w, offs=offsets)


def _naive(x: torch.Tensor, w: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    # Per-expert matmul, matching Qwen3VLMoeTextExpertsNaive (standard matmul).
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)
    outs = []
    start = 0
    for e in range(w.shape[0]):
        end = int(offsets[e])
        if end > start:
            outs.append(x[start:end] @ w[e])
        start = end
    return torch.cat(outs, dim=0) if outs else x.new_empty(0, w.shape[-1])


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    rel = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
    mae = (a - b).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"  {name:24s} max_rel_err={rel:.3e}  mae={mae:.3e}  cosine={cos:.8f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-experts", type=int, default=60)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--intermediate-size", type=int, default=1408)
    ap.add_argument("--tokens-per-expert", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = _device()
    dtype = getattr(torch, args.dtype)
    E, H, I, T = args.num_experts, args.hidden_size, args.intermediate_size, args.tokens_per_expert
    print(f"device={dev}  dtype={dtype}  E={E} H={H} I={I} tokens/expert={T}")

    gen = torch.Generator().manual_seed(args.seed)
    w = torch.randn(E, H, 2 * I, generator=gen) * (H ** -0.5)  # [E,H,2I]
    x = torch.randn(E * T, H, generator=gen)  # [E*T,H], grouped: expert e -> rows [e*T:(e+1)*T]
    num_per = torch.full((E,), T, dtype=torch.int64)  # [E]

    ref = _naive(x, w, num_per)  # float32 CPU golden

    x_d = x.to(dev, dtype)
    w_d = w.to(dev, dtype)
    num_per_d = num_per.to(dev)

    naive_d = _naive(x_d, w_d, num_per_d)
    try:
        grouped_d = _grouped_mm(x_d, w_d, num_per_d)
    except Exception as e:  # noqa: BLE001 — report and exit, this is a diagnostic probe
        print(f"torch._grouped_mm unavailable/failed on {dev}: {e}")
        return

    print("\nmax rel err vs float32 CPU reference:")
    _report("naive (bf16) vs ref", naive_d.cpu(), ref)
    _report("grouped (bf16) vs ref", grouped_d.cpu(), ref)
    print("\nmax rel err, naive vs grouped (isolates grouped_mm itself):")
    _report("grouped vs naive", grouped_d.cpu(), naive_d.cpu())


if __name__ == "__main__":
    main()
