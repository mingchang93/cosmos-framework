#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Fingerprint a DCP checkpoint, or compare two fingerprints.

For the NPU-vs-GPU loss gap: verify the A3 converted DCP (``convert_model_to_dcp``
output) matches the A800 original ``*.distcp``. The two checkpoints live on
different machines that cannot reach each other, so this tool splits the check
into two halves you run on each box and diff locally:

  # on A3 (NPU) -- path = dir containing .metadata (the model/ subdir):
  python3 tools/npu_debug/checkpoint_fingerprint.py dump /workspace/tmp/checkpoint_base/model --out fp_a3.json
  # on A800 (GPU):
  python3 tools/npu_debug/checkpoint_fingerprint.py dump <path-to-Cosmos3-Edge>/model --out fp_a800.json
  # copy the two small JSON files to one place, then:
  python3 tools/npu_debug/checkpoint_fingerprint.py compare fp_a3.json fp_a800.json

Each fingerprint records, per tensor: fqn, shape, dtype, numel, squared-L2-norm,
sum, and a SHA256 over the float32-cast bytes (position-sensitive, exact). A
mismatch in the hash means the tensors differ; the norm delta tells you whether
that's a permutation/precision difference (~0) or a real value difference (large).
"""

import argparse
import hashlib
import json

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def _fingerprint(t: torch.Tensor) -> dict:
    t32 = t.detach().cpu().float().contiguous()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": t.numel(),
        "norm2": float(t.double().pow(2).sum().item()),  # squared L2 norm (stable, positive)
        "sum": float(t.double().sum().item()),  # weak (cancellation) but cheap context
        "hash": hashlib.sha256(t32.numpy().tobytes()).hexdigest(),
    }


def _load_tensors(path: str) -> dict[str, torch.Tensor]:
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    # Build placeholder tensors from the metadata, then let dcp.load fill them in.
    # dcp.load is the framework's own load path (handles the __{rank}_{i}.distcp
    # file layout and sharded chunks); FileSystemReader has no public read_tensor.
    state_dict: dict[str, torch.Tensor] = {}
    for fqn, meta in metadata.state_dict_metadata.items():
        if not isinstance(meta, TensorStorageMetadata):
            continue
        props = getattr(meta, "properties", None)
        dtype = getattr(props, "dtype", torch.float32) if props is not None else torch.float32
        state_dict[fqn] = torch.empty(meta.size, dtype=dtype)
    dcp.load(state_dict, storage_reader=reader, no_dist=True)
    return {fqn: t.detach().cpu() for fqn, t in state_dict.items()}


def cmd_dump(args) -> None:
    fps = []
    for fqn, t in _load_tensors(args.checkpoint).items():
        fp = _fingerprint(t)
        fp["fqn"] = fqn
        fps.append(fp)
    fps.sort(key=lambda x: x["fqn"])
    with open(args.out, "w") as f:
        json.dump({"source": args.checkpoint, "num_tensors": len(fps), "tensors": fps}, f)
    print(f"wrote {len(fps)} tensors -> {args.out}")


def cmd_compare(args) -> None:
    A = {t["fqn"]: t for t in json.load(open(args.a))["tensors"]}
    B = {t["fqn"]: t for t in json.load(open(args.b))["tensors"]}
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    common = sorted(set(A) & set(B))
    print(f"common={len(common)}   only-in-A={len(only_a)}   only-in-B={len(only_b)}")
    if only_a:
        print("  only-in-A (first 10):", only_a[:10])
    if only_b:
        print("  only-in-B (first 10):", only_b[:10])

    identical = 0
    mismatches = []
    for k in common:
        a, b = A[k], B[k]
        if a["hash"] == b["hash"] and a["shape"] == b["shape"]:
            identical += 1
            continue
        rel = abs(a["norm2"] - b["norm2"]) / max(a["norm2"], b["norm2"], 1e-12)
        mismatches.append((rel, k, a, b))
    mismatches.sort(key=lambda r: -r[0])

    print(f"\nidentical={identical}/{len(common)}   mismatches={len(mismatches)}")
    for rel, k, a, b in mismatches[:20]:
        print(
            f"  {k}: norm_rel={rel:.2e}  dtype {a['dtype']}->{b['dtype']}  "
            f"shape {a['shape']} vs {b['shape']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="fingerprint a checkpoint to JSON")
    d.add_argument("checkpoint")
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_dump)
    c = sub.add_parser("compare", help="compare two fingerprint JSON files")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
