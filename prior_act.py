#!/usr/bin/env python3
"""Prior-ACT command line.

Author: Dawit Chun

One entry point for the whole pipeline. Nothing is specific to a task, an object or a machine:
paths are arguments, and camera keys, image shapes and the state dimension are read from the
dataset's meta/info.json or the checkpoint's config.json.

    python3 prior_act.py train     --dataset DIR --repo-id ID --dino DIR [--out DIR]
    python3 prior_act.py resume    --config CKPT/pretrained_model/train_config.json [--gpus 2]
    python3 prior_act.py localize  --dino DIR CKPT [CKPT ...]
    python3 prior_act.py check     CKPT/pretrained_model [--dataset DIR]
    python3 prior_act.py infer     CKPT/pretrained_model --infer-py PATH [--router-ip IP]

Every subcommand accepts --dry-run, which prints the command it would run and exits. Use it first.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd, dry):
    print("\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if dry:
        return 0
    return subprocess.call(cmd)


def _need_dir(p, what):
    if not Path(p).is_dir():
        sys.exit(f"{what} is not a directory: {p}")
    return str(Path(p).resolve())


def _need_file(p, what):
    if not Path(p).is_file():
        sys.exit(f"{what} is not a file: {p}")
    return str(Path(p).resolve())


# ── train ─────────────────────────────────────────────────────────────────────────────────────────

def cmd_train(a):
    ds = _need_dir(a.dataset, "--dataset")
    dino = _need_dir(a.dino, "--dino")
    info = Path(ds) / "meta" / "info.json"
    if info.is_file():
        d = json.loads(info.read_text())
        cams = [k for k in d.get("features", {}) if k.startswith("observation.images")]
        print(f"dataset: {len(cams)} camera(s), fps {d.get('fps')}")
        for k in cams:
            print(f"  {k}  {d['features'][k]['shape']}")
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={a.repo_id}",
        f"--dataset.root={ds}",
        f"--dataset.eval_split={a.eval_split}",
        f"--eval_steps={a.eval_steps}",
        f"--max_eval_samples={a.max_eval_samples}",
        f"--dataset.image_transforms.enable={str(a.augment).lower()}",
        "--policy.type=act_prior",
        "--policy.push_to_hub=false",          # lerobot refuses to start otherwise
        "--policy.use_amp=true",               # not the default; fp32 OOMs at batch 192 on 24 GB
        f"--policy.chunk_size={a.chunk_size}",
        f"--policy.n_action_steps={a.chunk_size}",
        f"--policy.optimizer_lr={a.lr}",
        f"--policy.optimizer_lr_backbone={a.lr}",
        f"--policy.camera_identity_embedding={str(a.camera_id_embed).lower()}",
        f"--policy.state_dropout={a.state_dropout}",
        f"--policy.scene_backbone={a.backbone}",
        f"--policy.scene_backbone_path={dino}",
        f"--policy.scene_prior_features={a.prior_features}",
        f"--policy.kl_weight={a.kl_weight}",
        f"--policy.kl_free_bits={a.kl_free_bits}",
        f"--policy.prior_fit_weight={a.prior_fit_weight}",
        f"--policy.latent_injection={a.latent_injection}",
        "--policy.deterministic_latent=true",
        f"--policy.latent_diag_every={a.latent_diag_every}",
        f"--steps={a.steps}",
        f"--batch_size={a.batch_size}",        # PER PROCESS
        f"--num_workers={a.num_workers}",
        f"--seed={a.seed}",
        f"--save_freq={a.save_freq}",
        f"--log_freq={a.log_freq}",
        f"--wandb.enable={str(a.wandb).lower()}",
        f"--wandb.project={a.wandb_project}",
        f"--output_dir={a.out}",
        f"--job_name={a.job_name}",
    ]
    if a.prior_cameras:
        cmd.append(f"--policy.scene_prior_cameras={a.prior_cameras}")
    return _run(cmd, a.dry_run)


# ── resume across GPUs ────────────────────────────────────────────────────────────────────────────

def cmd_resume(a):
    cfg = _need_file(a.config, "--config")
    # batch_size is PER PROCESS: lerobot logs "Effective batch size: batch_size x num_processes".
    # Halving it when doubling the process count keeps the effective batch, and therefore the
    # effective learning rate, the same as before the resume.
    cmd = ["accelerate", "launch",
           f"--num_processes={a.gpus}", "--num_machines=1", "--mixed_precision=no",
           "-m", "lerobot.scripts.lerobot_train",
           f"--config_path={cfg}", "--resume=true",
           f"--batch_size={a.batch_size}", f"--num_workers={a.num_workers}",
           f"--save_freq={a.save_freq}"]
    print(f"effective batch = {a.batch_size} x {a.gpus} = {a.batch_size * a.gpus}")
    return _run(cmd, a.dry_run)


# ── localize the DINOv3 path ──────────────────────────────────────────────────────────────────────

def cmd_localize(a):
    dino = _need_dir(a.dino, "--dino")
    n = 0
    for c in a.checkpoints:
        p = Path(c)
        cfgs = [p] if p.name == "config.json" else sorted(p.glob("**/config.json"))
        for cfg in cfgs:
            d = json.loads(cfg.read_text())
            if "scene_backbone_path" not in d:
                continue
            cur = d.get("scene_backbone_path") or ""
            if cur == dino:
                continue
            side = cfg.with_suffix(".json.orig_backbone_path")
            if not side.exists():
                side.write_text(cur + "\n")     # reversible: the original is kept beside it
            d["scene_backbone_path"] = dino
            if not a.dry_run:
                cfg.write_text(json.dumps(d, indent=2))
            print(f"  {cfg}\n    {cur or '(empty)'}\n    -> {dino}")
            n += 1
    print(f"{n} config(s) {'would be ' if a.dry_run else ''}rewritten")
    return 0


# ── checks ────────────────────────────────────────────────────────────────────────────────────────

def cmd_check(a):
    here = Path(__file__).parent / "scripts" / "diagnostics"
    cmd = [sys.executable, str(here / "test_scene_dependence.py"), a.ckpt]
    if a.policy_dir:
        cmd += ["--policy-dir", a.policy_dir]
    rc = _run(cmd, a.dry_run)
    if rc or a.dry_run:
        return rc
    cmd = [sys.executable, str(here / "latent_health.py"), a.ckpt]
    cmd += ["--dataset", a.dataset] if a.dataset else ["--synthetic"]
    if a.policy_dir:
        cmd += ["--policy-dir", a.policy_dir]
    return _run(cmd, a.dry_run)


# ── run on the robot ──────────────────────────────────────────────────────────────────────────────

def cmd_infer(a):
    infer_py = _need_file(a.infer_py, "--infer-py")
    _need_dir(a.ckpt, "checkpoint")
    cmd = [sys.executable, infer_py,
           "--policy", "act_prior", "--policy-path", str(Path(a.ckpt).resolve()),
           "--residual", "", "--router-ip", a.router_ip,
           "--trim-head", str(a.trim_head), "--seam-blend", str(a.seam_blend),
           "--execute-steps", str(a.execute_steps),
           "--savgol", str(a.savgol), "--savgol-order", str(a.savgol_order),
           "--ensemble", str(a.ensemble), "--ensemble-decay", str(a.ensemble_decay),
           "--grip-lead", str(a.grip_lead), "--grip-hold", str(a.grip_hold),
           "--max-vel", str(a.max_vel), "--max-acc", str(a.max_acc),
           "--speed", str(a.speed), "--max-steps", str(a.max_steps), "--home-first"]
    if a.live:
        cmd.append("--live")
    else:
        print("DRY RUN on the robot: infers and prints, publishes nothing. Add --live to move.")
    # ensemble is capped by execute_steps: a chunk is chunk_size waypoints and the averaging loop
    # breaks once execute_steps * i reaches it, so deeper ensembling silently does nothing.
    cap = max(1, a.chunk_size // max(a.execute_steps, 1))
    if a.ensemble > cap:
        print(f"note: --ensemble {a.ensemble} averages only {cap} plans at "
              f"--execute-steps {a.execute_steps} (chunk {a.chunk_size}). Use {cap}.")
    return _run(cmd, a.dry_run)


def main():
    ap = argparse.ArgumentParser(description="Prior-ACT pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dry-run", action="store_true", help="print the command and exit")

    t = sub.add_parser("train", help="train Prior-ACT")
    t.add_argument("--dataset", required=True); t.add_argument("--repo-id", required=True)
    t.add_argument("--dino", required=True, help="local DINOv3 checkpoint directory")
    t.add_argument("--out", default="out/prior_act"); t.add_argument("--job-name", default="prior_act")
    t.add_argument("--steps", type=int, default=20000); t.add_argument("--batch-size", type=int, default=192)
    t.add_argument("--num-workers", type=int, default=32); t.add_argument("--seed", type=int, default=1000)
    t.add_argument("--save-freq", type=int, default=500); t.add_argument("--log-freq", type=int, default=200)
    t.add_argument("--chunk-size", type=int, default=100); t.add_argument("--lr", default="1e-5")
    t.add_argument("--eval-split", type=float, default=0.111); t.add_argument("--eval-steps", type=int, default=500)
    t.add_argument("--max-eval-samples", type=int, default=512)
    t.add_argument("--backbone", default="dinov3"); t.add_argument("--prior-features", default="cls_grid")
    t.add_argument("--prior-fit-weight", type=float, default=10.0)
    t.add_argument("--kl-weight", type=float, default=10.0); t.add_argument("--kl-free-bits", type=float, default=0.05)
    t.add_argument("--latent-injection", default="decoder_seed")
    t.add_argument("--latent-diag-every", type=int, default=250)
    t.add_argument("--state-dropout", type=float, default=0.5)
    t.add_argument("--camera-id-embed", type=lambda s: s.lower() != "false", default=True)
    t.add_argument("--augment", type=lambda s: s.lower() != "false", default=True)
    t.add_argument("--prior-cameras", default="", help="comma-separated image keys for the prior")
    t.add_argument("--wandb", type=lambda s: s.lower() != "false", default=True)
    t.add_argument("--wandb-project", default="prior_act")
    common(t); t.set_defaults(func=cmd_train)

    r = sub.add_parser("resume", help="resume a run, optionally across GPUs")
    r.add_argument("--config", required=True); r.add_argument("--gpus", type=int, default=2)
    r.add_argument("--batch-size", type=int, default=96, help="PER PROCESS")
    r.add_argument("--num-workers", type=int, default=24); r.add_argument("--save-freq", type=int, default=500)
    common(r); r.set_defaults(func=cmd_resume)

    l = sub.add_parser("localize", help="point a checkpoint's DINOv3 path at this machine")
    l.add_argument("--dino", required=True); l.add_argument("checkpoints", nargs="+")
    common(l); l.set_defaults(func=cmd_localize)

    c = sub.add_parser("check", help="verify a checkpoint loads and the prior is alive")
    c.add_argument("ckpt"); c.add_argument("--dataset", default=None)
    c.add_argument("--policy-dir", default=None)
    common(c); c.set_defaults(func=cmd_check)

    i = sub.add_parser("infer", help="run on the robot")
    i.add_argument("ckpt"); i.add_argument("--infer-py", required=True)
    i.add_argument("--router-ip", default="127.0.0.1")
    i.add_argument("--live", action="store_true", help="publish; without it nothing moves")
    i.add_argument("--execute-steps", type=int, default=25); i.add_argument("--chunk-size", type=int, default=100)
    i.add_argument("--ensemble", type=int, default=4); i.add_argument("--ensemble-decay", type=float, default=0.3)
    i.add_argument("--savgol", type=int, default=17); i.add_argument("--savgol-order", type=int, default=3)
    i.add_argument("--seam-blend", type=int, default=12); i.add_argument("--trim-head", type=int, default=0)
    i.add_argument("--grip-lead", type=int, default=14); i.add_argument("--grip-hold", type=int, default=4)
    i.add_argument("--max-vel", type=float, default=0.8); i.add_argument("--max-acc", type=float, default=6.0)
    i.add_argument("--speed", type=float, default=1.0); i.add_argument("--max-steps", type=int, default=1200)
    common(i); i.set_defaults(func=cmd_infer)

    a = ap.parse_args()
    sys.exit(a.func(a))


if __name__ == "__main__":
    main()
