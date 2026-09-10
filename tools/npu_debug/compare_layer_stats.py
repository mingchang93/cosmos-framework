#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Diff two ``[layer_stats]`` dumps (NPU vs GPU) to find the first divergent op.

Parses the per-layer gen-sequence stats printed by ``MoTDecoderLayer.forward``
with ``COSMOS3_DEBUG_LAYER_STATS=1``, aligns them by ``L{layer}.{boundary}``, and
shows the std relative difference at each boundary in forward order. The first
boundary whose std differs from GPU by more than ``--threshold`` is the first
op where NPU diverges.

Usage:
    python3 tools/npu_debug/compare_layer_stats.py layer_stats_npu.txt layer_stats_gpu.txt

Interpretation (first boundary to cross threshold):
    in_gen   -> divergence is BEFORE the transformer (vae2llm / timestep embed)
    ln1/ln2  -> RMSNorm
    attn_gen -> attention (npu_fusion_attention vs natten)
    mlp_gen  -> MoE experts (expected clean)
    out_gen  -> residual / compounding
"""

import argparse
import re

_LINE = re.compile(
    r"\[layer_stats\] iter=(-?\d+) rank=(-?\d+) (?:L(\d+)\.)?([\w.]+) mean=(-?[\d.eE+-]+) std=(-?[\d.eE+-]+) min=(-?[\d.eE+-]+) max=(-?[\d.eE+-]+)"
)

_BOUNDARIES = ["in_gen", "ln1_gen", "attn_gen", "attn_res_gen", "ln2_gen", "mlp_gen", "out_gen"]


def _parse(path: str, rank: int | None, iteration: int | None = None) -> dict[tuple[int, str], tuple[float, float, float, float]]:
    stats: dict[tuple[int, str], tuple[float, float, float, float]] = {}
    seen_ranks: set[int] = set()
    for line in open(path):
        m = _LINE.search(line)
        if not m:
            continue
        it = int(m.group(1))
        if iteration is not None and it != iteration:
            continue
        r = int(m.group(2))
        seen_ranks.add(r)
        if rank is not None and r != rank:
            continue
        layer = int(m.group(3)) if m.group(3) is not None else -1  # -1 = global boundary (no L-layer prefix)
        key = (layer, m.group(4))
        if key in stats:
            print(f"[warn] duplicate {key} for rank {r} in {path} (gradient-checkpointing recompute?)")
        stats[key] = (float(m.group(5)), float(m.group(6)), float(m.group(7)), float(m.group(8)))  # (mean, std, min, max)
    if rank is None and len(seen_ranks) > 1:
        print(f"[warn] {path} contains {len(seen_ranks)} ranks {sorted(seen_ranks)}; pass --rank to filter")
    return stats


def _rel_diff(
    mean_n: float, std_n: float, mn_n: float, mx_n: float,
    mean_g: float, std_g: float, mn_g: float, mx_g: float,
    metric: str,
) -> float:
    """Relative difference between two (mean, std, min, max) summaries.

    ``shift`` is first-order sensitive to per-element differences (a per-element
    perturbation moves mean/max/min linearly); ``std`` is second-order (a
    per-element perturbation moves the variance quadratically, hiding small diffs).
    """
    denom = max(abs(std_g), 1e-6)
    if metric == "shift":
        return max(abs(mean_n - mean_g), abs(mx_n - mx_g), abs(mn_n - mn_g)) / denom
    return abs(std_n - std_g) / denom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npu_dump")
    ap.add_argument("gpu_dump")
    ap.add_argument("--rank", type=int, default=None, help="filter to this rank (default: use all lines, warn if multi-rank)")
    ap.add_argument("--iter", type=int, default=None, help="filter to this training iteration (default: all)")
    ap.add_argument("--threshold", type=float, default=0.01, help="relative-diff threshold (default 1%%)")
    ap.add_argument(
        "--metric",
        choices=["std", "shift"],
        default="shift",
        help="std = |Δstd|/std (2nd-order, misses small per-element diffs); "
        "shift = peak(|Δmean|,|Δmax|,|Δmin|)/std (1st-order, catches per-element divergence)",
    )
    ap.add_argument("--grad", action="store_true", help="diff backward-grad boundaries (*.grad); scanned loss->input, so FIRST >threshold = the op that injects the error")
    args = ap.parse_args()

    npu = _parse(args.npu_dump, args.rank, args.iter)
    gpu = _parse(args.gpu_dump, args.rank, args.iter)
    layers = sorted({l for l, _ in set(npu) | set(gpu) if l >= 0})
    globals_ = sorted({b for l, b in set(npu) | set(gpu) if l == -1})

    # Backward grads fire in reverse (loss->input): the first divergent grad is
    # the op that injects the error, so scan reversed. Forward scans input->output.
    boundaries = [f"{b}.grad" for b in _BOUNDARIES] if args.grad else _BOUNDARIES
    ordered_layers = reversed(layers) if args.grad else layers

    first = None
    for layer in ordered_layers:
        row = []
        for b in boundaries:
            key = (layer, b)
            if key not in npu or key not in gpu:
                row.append(f"{b}=?")
                continue
            mean_n, std_n, mn_n, mx_n = npu[key]
            mean_g, std_g, mn_g, mx_g = gpu[key]
            rd = _rel_diff(mean_n, std_n, mn_n, mx_n, mean_g, std_g, mn_g, mx_g, args.metric)
            if first is None and rd > args.threshold:
                first = (layer, b, mean_n, std_n, mean_g, std_g, rd)
            row.append(f"{b}={rd:.3%}")
        print(f"L{layer:2d}: " + "  ".join(row))

    for b in globals_:
        key = (-1, b)
        if key not in npu or key not in gpu:
            print(f"{b}=?")
            continue
        mean_n, std_n, mn_n, mx_n = npu[key]
        mean_g, std_g, mn_g, mx_g = gpu[key]
        rd = _rel_diff(mean_n, std_n, mn_n, mx_n, mean_g, std_g, mn_g, mx_g, args.metric)
        if first is None and rd > args.threshold:
            first = (-1, b, mean_n, std_n, mean_g, std_g, rd)
        print(f"{b}: {args.metric}={rd:.3%}  (npu std={std_n:.6f}  gpu std={std_g:.6f})")

    if first is None:
        print(f"\nno boundary exceeded {args.threshold:.1%}")
        return
    layer, b, mn, sn, mg, sg, rd = first
    label = f"{b}" if layer < 0 else f"L{layer}.{b}"
    print(
        f"\nFIRST boundary > {args.threshold:.1%} ({args.metric}): {label}\n"
        f"  rel = {rd:.3%}\n"
        f"  mean: npu={mn:.6f}  gpu={mg:.6f}"
    )


if __name__ == "__main__":
    main()
