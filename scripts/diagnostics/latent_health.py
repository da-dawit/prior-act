#!/usr/bin/env python3
"""Whether the latent branch is doing anything, for any act_prior checkpoint.

Author: Dawit Chun

Computes post-hoc the numbers `latent_diag_every` logs during training, so an existing checkpoint can
be judged without retraining.

    latent_authority   |a(z=mu_q) - a(z=0)|      the decoder ignores z if this is ~0
    prior_post_gap     |a(z=mu_q) - a(z=mu_p)|   the train/test mismatch, in action units
    mu_prior_std       spread of the prior mean across scenes; ~0 is a learned constant
    mu_q_std           spread of the posterior mean; ~0 is posterior collapse
    kld_loss           against the learned prior

Reference values from a working model: authority 0.1489, gap 0.0032 (2% of authority),
mu_prior_std 0.0961. Two failure signatures seen in practice: authority ~1e-4 (latent dead), and
authority high with the gap nearly equal to it (train/test shortcut, worse than dead).

Camera keys, image shapes and the state dimension are read from the checkpoint's config.json. The
dataset is only needed for real observations; with --synthetic the probes run on noise, which is
enough to detect a dead latent though not to judge scene dependence.

Usage:
    python3 latent_health.py <ckpt>/pretrained_model --dataset /path/to/lerobot_dataset
    python3 latent_health.py <ckpt>/pretrained_model --synthetic
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_real_batch(ds_root, cams, n, rng):
    """n observations from the dataset, one per episode where possible."""
    import cv2
    import pandas as pd

    info = json.loads((Path(ds_root) / "meta" / "info.json").read_text())
    fps = info["fps"]
    ep_files = glob.glob(f"{ds_root}/meta/episodes/**/*.parquet", recursive=True)
    em = pd.read_parquet(ep_files[0])
    df = pd.read_parquet(glob.glob(f"{ds_root}/data/**/*.parquet", recursive=True)[0])
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)

    keys = [c.split("observation.images.")[-1] for c in cams]
    out = []
    for i in range(min(n, len(em))):
        r = em.iloc[i]
        frames = []
        ok = True
        for k in keys:
            ci = int(r[f"videos/observation.images.{k}/chunk_index"])
            fi = int(r[f"videos/observation.images.{k}/file_index"])
            t0 = float(r[f"videos/observation.images.{k}/from_timestamp"])
            path = f"{ds_root}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            cap = cv2.VideoCapture(path)
            # mid-episode: the arm is engaged with the scene rather than at the home pose
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)) + 200)
            got, im = cap.read()
            cap.release()
            if not got:
                ok = False
                break
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if not ok:
            continue
        j = int(r["dataset_from_index"]) + 200
        if j + 100 >= len(states):
            continue
        out.append((frames, states[j], actions[j:j + 100]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to a checkpoint's pretrained_model directory")
    ap.add_argument("--dataset", default=None, help="LeRobot dataset root, for real observations")
    ap.add_argument("--synthetic", action="store_true", help="use noise instead of a dataset")
    ap.add_argument("--n", type=int, default=24, help="number of probe observations")
    ap.add_argument("--policy-dir", default=None,
                    help="directory containing act_prior_policy.py, if not importable")
    a = ap.parse_args()
    if not a.dataset and not a.synthetic:
        ap.error("give --dataset <root> or --synthetic")
    if a.policy_dir:
        sys.path.insert(0, a.policy_dir)

    ck = Path(a.ckpt)
    cfg = json.loads((ck / "config.json").read_text())
    feats = cfg.get("input_features", {})
    cams = [k for k in feats if k.startswith("observation.images")]
    shapes = {k: tuple(feats[k]["shape"]) for k in cams}
    state_dim = feats.get("observation.state", {}).get("shape", [16])[0]

    from lerobot.policies.factory import get_policy_class
    pol = get_policy_class(cfg["type"]).from_pretrained(str(ck)).cuda()
    m = pol.model
    print(f"{ck.parent.name}  {sum(p.numel() for p in pol.parameters())/1e6:.1f}M params  "
          f"cameras {len(cams)}")
    print(f"injection={cfg.get('latent_injection')}  features={cfg.get('scene_prior_features')}  "
          f"prior_fit_weight={cfg.get('prior_fit_weight')}  kl_free_bits={cfg.get('kl_free_bits')}")

    rng = np.random.default_rng(0)
    if a.synthetic:
        batch = []
        for _ in range(a.n):
            frames = [rng.integers(0, 255, (h, w, c), dtype=np.uint8)
                      for (c, h, w) in (shapes[k] for k in cams)]
            batch.append((frames, np.zeros(state_dim, np.float32),
                          np.zeros((100, state_dim), np.float32)))
    else:
        batch = load_real_batch(a.dataset, cams, a.n, rng)
        if not batch:
            raise SystemExit("no usable observations read from the dataset")
    print(f"\n{len(batch)} probes ({'synthetic' if a.synthetic else 'real frames'})\n")

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    pre, _ = make_pre_post_processors(PreTrainedConfig.from_pretrained(str(ck)),
                                      pretrained_path=str(ck))

    # Probes run in EVAL mode. nn.MultiheadAttention keeps dropout as a float attribute rather than
    # an nn.Dropout submodule, so it stays active in training mode no matter what is done to the
    # module tree; two training-mode forwards then differ by more than the effect being measured.
    pol.eval()
    acc = {}
    with torch.no_grad():
        for frames, state, action in batch:
            obs = {"observation.state": torch.as_tensor(state).reshape(-1), "task": "probe"}
            for k, im in zip(cams, frames):
                t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()
                obs[k] = t / 255.0 if float(t.max()) > 1.5 else t
            obs["action"] = torch.as_tensor(action)
            obs["action_is_pad"] = torch.zeros(len(action), dtype=torch.bool)
            b = pre(obs)
            b = {k: (v.cuda().unsqueeze(0) if torch.is_tensor(v) and v.dim() > 0 else v)
                 for k, v in b.items()}
            b["observation.images"] = [b[k] for k in cams]

            mu_q, mu_p = m.encode_latents(b) if hasattr(m, "encode_latents") else (None, None)
            if mu_q is None:
                raise SystemExit(
                    "this checkpoint's model does not expose encode_latents(); use the training-time "
                    "diagnostics via --policy.latent_diag_every instead")
            a_mu = m(b, latent_override=mu_q)[0]
            a_zero = m(b, latent_override=torch.zeros_like(mu_q))[0]
            acc.setdefault("latent_authority", []).append((a_mu - a_zero).abs().mean().item())
            if mu_p is not None:
                a_pri = m(b, latent_override=mu_p)[0]
                acc.setdefault("prior_post_gap", []).append((a_mu - a_pri).abs().mean().item())
                acc.setdefault("_mu_p", []).append(mu_p.squeeze(0).cpu().numpy())
            acc.setdefault("_mu_q", []).append(mu_q.squeeze(0).cpu().numpy())

    auth = float(np.mean(acc["latent_authority"]))
    gap = float(np.mean(acc.get("prior_post_gap", [float("nan")])))
    mu_q_std = float(np.stack(acc["_mu_q"]).std(0).mean())
    mu_p_std = float(np.stack(acc["_mu_p"]).std(0).mean()) if "_mu_p" in acc else float("nan")

    print(f"  latent_authority   {auth:.6f}")
    print(f"  prior_post_gap     {gap:.6f}")
    print(f"  mu_q_std           {mu_q_std:.6f}")
    print(f"  mu_prior_std       {mu_p_std:.6f}")
    print("\nVERDICT")
    if auth < 1e-3:
        print("  the decoder ignores z. The scene backbone contributes nothing to the action.")
    elif gap > 0.5 * auth:
        print(f"  authority {auth:.4f} but the prior only reproduces it to {gap:.4f} "
              f"({100*gap/auth:.0f}% is train/test mismatch) -- ACTIVELY HARMFUL.")
    else:
        print(f"  latent works: authority {auth:.4f}, prior/posterior gap {gap:.4f} "
              f"({100*gap/max(auth,1e-9):.0f}% of it).")


if __name__ == "__main__":
    main()
