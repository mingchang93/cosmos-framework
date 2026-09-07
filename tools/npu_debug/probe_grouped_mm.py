#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Probe the MoE expert numerics on the current device vs a CPU fp32 reference.

Isolates the remaining loss-gap suspects in the gen-tower MoE experts. Three
tests, each comparing a device-bf16 result against a CPU float32 golden (and,
where relevant, grouped_mm vs the plain per-expert matmul):

  1. matmul      : the gate_up projection (K=hidden_size)
  2. silu        : elementwise x*sigmoid(x)
  3. SwiGLU      : the full expert forward (gate_up matmul -> silu(gate)*up*score
                   -> down_proj matmul), via both grouped_mm and naive matmul

The verdict is cross-platform: run on NPU and CUDA and compare the `mae`/`cosine`
columns. If NPU's drift is far larger than CUDA's on the same row, that op is the
culprit. (`max_rel_err` explodes on near-zero reference values — ignore it; the
`mae`/`cosine` are the real signal.)

Usage:
    python3 tools/npu_debug/probe_grouped_mm.py                 # bf16, default shapes
    python3 tools/npu_debug/probe_grouped_mm.py --dtype float32 # isolate bf16 rounding
"""

import argparse

import torch
import torch.nn.functional as F


def _device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _grouped_mm(x: torch.Tensor, w: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    # Mirrors moe.py:_run_experts_grouped_mm's exact call (offs = cumsum, int32).
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    return torch._grouped_mm(x, w, offs=offsets)


def _naive_mm(x: torch.Tensor, w: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    outs = []
    start = 0
    for e in range(w.shape[0]):
        n = int(num_tokens_per_expert[e])
        if n > 0:
            outs.append(x[start : start + n] @ w[e])
        start += n
    return torch.cat(outs, dim=0) if outs else x.new_empty(0, w.shape[-1])


def _swiglu_grouped(x, gate_up, down, num_tokens_per_expert, scores):
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(x, gate_up, offs=offsets)  # [T,2I]
    gate, up = torch.chunk(h, 2, dim=-1)
    h = F.silu(gate) * up * scores.unsqueeze(-1)  # [T,I]
    return torch._grouped_mm(h, down, offs=offsets)  # [T,H]


def _swiglu_naive(x, gate_up, down, num_tokens_per_expert, scores):
    outs = []
    start = 0
    for e in range(gate_up.shape[0]):
        n = int(num_tokens_per_expert[e])
        if n > 0:
            h = x[start : start + n] @ gate_up[e]  # [n,2I]
            gate, up = torch.chunk(h, 2, dim=-1)
            h = F.silu(gate) * up * scores[start : start + n].unsqueeze(-1)  # [n,I]
            outs.append(h @ down[e])  # [n,H]
        start += n
    return torch.cat(outs, dim=0) if outs else x.new_empty(0, down.shape[-1])


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    rel = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
    mae = (a - b).abs().mean().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"  {name:24s} max_rel_err={rel:.3e}  mae={mae:.3e}  cosine={cos:.8f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-experts", type=int, default=60)
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--moe-intermediate", type=int, default=1408)
    ap.add_argument("--tokens-per-expert", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = _device()
    dtype = getattr(torch, args.dtype)
    E, H, I, T = args.num_experts, args.hidden_size, args.moe_intermediate, args.tokens_per_expert
    print(f"device={dev}  dtype={dtype}  E={E} H={H} I={I} tokens/expert={T}")

    gen = torch.Generator().manual_seed(args.seed)
    gate_up = torch.randn(E, H, 2 * I, generator=gen) * (H ** -0.5)  # [E,H,2I]
    down = torch.randn(E, I, H, generator=gen) * (I ** -0.5)  # [E,I,H]
    x = torch.randn(E * T, H, generator=gen)  # [E*T,H], expert e -> rows [e*T:(e+1)*T]
    num_per = torch.full((E,), T, dtype=torch.int64)  # [E]
    scores = torch.ones(E * T)  # ones -> isolate silu from the score mul

    x_d = x.to(dev, dtype)
    gate_up_d = gate_up.to(dev, dtype)
    down_d = down.to(dev, dtype)
    num_per_d = num_per.to(dev)
    scores_d = scores.to(dev, dtype)

    # Test 1: matmul only (gate_up projection)
    print("\n=== matmul (gate_up, K=hidden_size) ===")
    ref = _naive_mm(x.float(), gate_up.float(), num_per)
    naive = _naive_mm(x_d, gate_up_d, num_per_d)
    try:
        grouped = _grouped_mm(x_d, gate_up_d, num_per_d)
    except Exception as e:  # noqa: BLE001 — report and exit, this is a diagnostic probe
        print(f"torch._grouped_mm unavailable/failed on {dev}: {e}")
        return
    _report("naive vs ref", naive.cpu(), ref)
    _report("grouped vs ref", grouped.cpu(), ref)
    _report("grouped vs naive", grouped.cpu(), naive.cpu())

    # Test 2: elementwise silu
    print("\n=== elementwise silu ===")
    z = torch.randn(E * T, generator=gen) * 3.0  # cover sigmoid's interesting range
    ref_silu = z.float() * torch.sigmoid(z.float())
    out_silu = F.silu(z.to(dev, dtype))
    _report("silu vs ref", out_silu.cpu(), ref_silu)

    # Test 3: full SwiGLU expert (matmul + silu + mul + matmul)
    print("\n=== full SwiGLU expert (grouped vs naive vs ref) ===")
    ref_sw = _swiglu_naive(x.float(), gate_up.float(), down.float(), num_per, scores.float())
    sw_naive = _swiglu_naive(x_d, gate_up_d, down_d, num_per_d, scores_d)
    try:
        sw_grouped = _swiglu_grouped(x_d, gate_up_d, down_d, num_per_d, scores_d)
    except Exception as e:  # noqa: BLE001
        print(f"torch._grouped_mm unavailable/failed on {dev}: {e}")
        return
    _report("swiglu naive vs ref", sw_naive.cpu(), ref_sw)
    _report("swiglu grouped vs ref", sw_grouped.cpu(), ref_sw)
    _report("swiglu grouped vs naive", sw_grouped.cpu(), sw_naive.cpu())


if __name__ == "__main__":
    main()
