#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Dump the gen-tower MoE flags from a Cosmos3-Edge checkpoint config.

Determines whether the config-gated suspects in the NPU-vs-GPU loss gap are even
active for the generation tower:
  - gen_noisy_gating              (RNG noise in the router forward pass)
  - gen_cosine_router_config      (fp32 F.normalize + cosine routing)
  - gen_moe_shared_expert         (extra always-on SwiGLU branch)
  - gen_aux_loss_free_load_balancing_config

Reads only the checkpoint's config.json (no model weights), so it runs in
seconds. If none of these keys appear, all defaults apply — i.e. noisy gating,
cosine routing, and the shared expert are all OFF.

Usage:
    python3 tools/npu_debug/dump_gen_flags.py /workspace/tmp/checkpoint_base
"""

import argparse
import sys

from cosmos_framework.inference2._model_io import Cosmos3OmniConfig

# Keys to surface. The two "*_config" entries dump their whole sub-dict when
# present; the leaf keys are reported inline wherever they nest.
_FLAG_KEYS = (
    "gen_noisy_gating",
    "gen_moe_shared_expert",
    "gen_moe_shared_expert_intermediate_scale",
    "gen_moe_top_k",
    "gen_cosine_router_config",
    "gen_aux_loss_free_load_balancing_config",
    "use_cosine_similarity",
)


def _walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if k in _FLAG_KEYS:
                yield p, v
            elif isinstance(v, (dict, list)):
                yield from _walk(v, p)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _walk(v, f"{path}[{i}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_path", help="path to the DCP checkpoint directory")
    args = ap.parse_args()

    try:
        cfg = Cosmos3OmniConfig.from_pretrained(args.checkpoint_path)
    except Exception as e:  # noqa: BLE001 — report a clear failure, this is a probe
        print(f"failed to load config from {args.checkpoint_path}: {e}")
        sys.exit(1)

    found = list(_walk(cfg.model))
    if not found:
        print("no gen-tower MoE flags in config -> all defaults apply:")
        print("  gen_noisy_gating=False  cosine_router(OFF)  shared_expert=False  aux-loss-free LB(OFF)")
        return

    for path, value in found:
        print(f"  {path} = {value!r}")


if __name__ == "__main__":
    main()
