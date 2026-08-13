#!/usr/bin/env python3
"""Check that the scene prior influences the DEPLOYED action.

Author: Dawit Chun

Training diagnostics measure the latent where z comes from the posterior. At inference z comes from
the prior, so a model can pass every training check and still behave as plain ACT on the robot. This
feeds two different scene images with an identical state and reports the change in the commanded
chunk.

Also verifies three things that fail silently:
  - the checkpoint loads (an unlocalised DINOv3 path raises a HuggingFace repo-id error)
  - output is deterministic across identical calls (state_dropout must be inert at inference)
  - inference latency against the control period

Usage:
    python3 test_scene_dependence.py <ckpt>/pretrained_model [--policy-dir /path/to/deploy_pkg]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--policy-dir", default=None,
                    help="directory containing act_prior_policy.py, if not on sys.path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--execute-steps", type=int, default=25,
                    help="control steps executed per re-plan, for the latency budget")
    ap.add_argument("--demo-step", type=float, default=0.0071,
                    help="mean per-step action delta in the demonstrations, as a unit of comparison")
    a = ap.parse_args()

    if a.policy_dir:
        sys.path.insert(0, a.policy_dir)
    from act_prior_policy import ActPriorPolicy

    ckpt = Path(a.ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())

    # Camera keys and image shapes come from the checkpoint, not from an assumption about the task.
    feats = cfg.get("input_features", {})
    cams = [k for k in feats if k.startswith("observation.images")]
    if not cams:
        raise SystemExit("no image features in config.json")
    shapes = {k: tuple(feats[k]["shape"]) for k in cams}
    state_dim = feats.get("observation.state", {}).get("shape", [16])[0]

    print(f"cameras     {len(cams)}")
    for k in cams:
        print(f"  {k}  {shapes[k]}")
    print(f"state dim   {state_dim}")
    print(f"prior       backbone={cfg.get('scene_backbone')} features={cfg.get('scene_prior_features')} "
          f"injection={cfg.get('latent_injection')} prior_fit_weight={cfg.get('prior_fit_weight')}")

    pol = ActPriorPolicy(str(ckpt), camera_keys=cams, device="cuda")
    print(f"training flag (must be False): {pol.policy.training}")

    rng = np.random.default_rng(0)

    def imgs(seed_offset=0):
        r = np.random.default_rng(seed_offset)
        out = []
        for k in cams:
            c, h, w = shapes[k]                       # config stores CHW
            out.append(r.integers(0, 255, (h, w, c), dtype=np.uint8))
        return out

    state = np.zeros(state_dim, np.float32)
    task = "task"

    base = imgs(0)
    a1 = pol.predict_chunk(base, state, task)
    a2 = pol.predict_chunk(base, state, task)
    det = float(np.abs(a1 - a2).max())
    print(f"\nchunk {a1.shape}  range [{a1.min():.3f}, {a1.max():.3f}]")
    print(f"determinism  max|diff| over identical calls = {det:.2e}   "
          f"{'OK' if det < 1e-9 else 'FAIL: state_dropout is active at inference'}")

    # Vary ONLY the first camera, which is the one the prior reads by default.
    other = imgs(1)
    swapped = [other[0]] + base[1:]
    a3 = pol.predict_chunk(swapped, state, task)
    shift = float(np.abs(a3 - a1).mean())
    print(f"scene dependence  different scene image shifts the action {shift:.4f} rad "
          f"({shift / a.demo_step:.1f} demonstration steps)")
    if shift < 1e-4:
        print("  FAIL: the prior does not influence the deployed action. This is plain ACT with a "
              "frozen backbone attached.")
    else:
        print("  OK: the prior drives the deployed action.")

    ts = []
    for _ in range(6):
        t = time.time(); pol.predict_chunk(base, state, task); ts.append(time.time() - t)
    med = float(np.median(ts))
    budget = a.execute_steps / a.fps
    print(f"\nlatency  {1000*med:.0f} ms/chunk, one re-plan per {1000*budget:.0f} ms   "
          f"{'OK' if med < budget else 'TOO SLOW for this execute-steps'}")


if __name__ == "__main__":
    main()
