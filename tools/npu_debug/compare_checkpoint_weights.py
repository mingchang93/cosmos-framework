#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Compare two DCP checkpoints tensor-by-tensor, without a model.

Diagnostic for the NPU-vs-GPU loss gap: verify the A3 converted DCP
(``convert_model_to_dcp`` output) is numerically identical to the A800 original
``*.distcp``. Input alignment alone proves nothing about weights, and the
conversion has never been checked (only file size was).

Reports:
  - key-set diff (tensors present in only one checkpoint),
  - per-tensor max relative error / cosine for shared keys,
  - a name-agnostic content signature so a rename-only conversion still shows up
    as "identical values under different keys".

Usage:
    python3 tools/npu_debug/compare_checkpoint_weights.py \
        --a /workspace/tmp/checkpoint_base \
        --b /path/to/original/model

Note: reads DCP via ``torch.distributed.checkpoint`` metadata. Only single-chunk
(unsharded) tensors are compared; multi-chunk (sharded) tensors are reported and
skipped.
"""

import argparse
from collections import Counter

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def _load_tensors(path: str) -> dict[str, torch.Tensor]:
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    tensors: dict[str, torch.Tensor] = {}
    skipped = 0
    for fqn, meta in metadata.state_dict_metadata.items():
        if not isinstance(meta, TensorStorageMetadata):
            continue
        if len(meta.chunks) != 1:
            skipped += 1
            continue
        tensors[fqn] = reader.read_tensor(fqn, meta.chunks[0]).cpu()
    if skipped:
        print(f"  [skip sharded tensors: {skipped}]")
    return tensors


def _content_sig(t: torch.Tensor) -> tuple[tuple[int, ...], str, float]:
    # Name-agnostic signature: shape + dtype + fp64 sum. fp64 sum is stable enough
    # to tell "same values" apart from "different values" for a rename-only check.
    return (tuple(t.shape), str(t.dtype), t.double().sum().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint A (converted DCP)")
    ap.add_argument("--b", required=True, help="checkpoint B (original distcp)")
    ap.add_argument("--max-diff-tensors", type=int, default=20)
    args = ap.parse_args()

    print(f"loading A: {args.a}")
    A = _load_tensors(args.a)
    print(f"  {len(A)} tensors")
    print(f"loading B: {args.b}")
    B = _load_tensors(args.b)
    print(f"  {len(B)} tensors")

    keys_a, keys_b = set(A), set(B)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print(f"\ncommon keys: {len(common)}   only-in-A: {len(only_a)}   only-in-B: {len(only_b)}")
    if only_a:
        print("  only-in-A (first 10):", only_a[:10])
    if only_b:
        print("  only-in-B (first 10):", only_b[:10])

    rows = []
    for k in common:
        a, b = A[k].float(), B[k].float()
        if a.shape != b.shape:
            rows.append((float("inf"), k, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
            continue
        a = a.reshape(-1)
        b = b.reshape(-1)
        rel = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rows.append((rel, k, f"cosine={cos:.8f}"))
    rows.sort(key=lambda r: -r[0])

    print(f"\nworst {min(args.max_diff_tensors, len(rows))} shared tensors by max rel err:")
    for rel, k, note in rows[: args.max_diff_tensors]:
        print(f"  {rel:.3e}  {k}  {note}")

    # Name-agnostic content check: catch rename-only conversions.
    sig_a = Counter(_content_sig(t) for t in A.values())
    sig_b = Counter(_content_sig(t) for t in B.values())
    common_sigs = sum((sig_a & sig_b).values())
    print(f"\ncontent signatures: A={sum(sig_a.values())}  B={sum(sig_b.values())}  matching={common_sigs}")
    if common_sigs < min(len(A), len(B)):
        print("  -> some tensor VALUES differ (not just names); see per-tensor table above.")


if __name__ == "__main__":
    main()
