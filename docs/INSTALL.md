# Installation

Author: Dawit Chun

## Requirements

- Python 3.10+
- PyTorch with CUDA
- A LeRobot checkout (the policy installs into it)
- `transformers` (loads DINOv3), `safetensors`, `numpy`, `scipy`, `opencv-python`, `pandas`
- 24 GB VRAM at batch 192 with AMP. Less with a smaller batch.

## 1. LeRobot

```bash
git clone https://github.com/huggingface/lerobot
cd lerobot && pip install -e .
```

## 2. DINOv3 weights

DINOv3 is licence-gated on HuggingFace. Accept the terms, then download once to a local directory:

```bash
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir /path/to/dinov3-vitl16-pretrain-lvd1689m
```

The policy is passed a local path, never a hub id. `torch.hub` does not work here: it downloads from
`dl.fbaipublicfiles.com`, which never sees the HF token and returns 403 even after the licence is
accepted.

ViT-S and ViT-B also work. The prior head reads the backbone width from its config rather than
assuming 384, so no code change is needed.

## 3. The policy

```bash
cp lerobot_policy/modeling_act_prior.py      <lerobot>/src/lerobot/policies/act_prior/
cp lerobot_policy/configuration_act_prior.py <lerobot>/src/lerobot/policies/act_prior/
```

If `act_prior` is not registered in the installed LeRobot version, add it to the policy factory alongside
`act`.

Verify:

```bash
python3 -c "
from lerobot.policies.act_prior.configuration_act_prior import ACTPriorConfig
c = ACTPriorConfig()
for k in ('prior_fit_weight','scene_prior_features','latent_injection','kl_free_bits'):
    print(k, getattr(c, k))
"
```

Expected: `0.0 cls token 0.0` — these are the backward-compatible defaults. The training script sets
the working values.

## 4. Dataset

Any LeRobot v3 dataset. The policy reads camera keys, state dimension and action dimension from
`meta/info.json`; nothing is hardcoded to a task.

Requirements:

- at least one camera (the prior reads camera 0 unless `scene_prior_cameras` is set)
- `observation.state` and `action` of matching dimension
- enough episodes for an episode-level holdout (`eval_split=0.111` keeps 10 of 90)

## 5. Moving checkpoints between machines

`config.json` records the DINOv3 path from the machine that trained it. On a different machine
loading fails with:

```text
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/workspace/pretrained/...'
```

Rewrite it:

```bash
LOCAL_DINO=/path/to/dinov3 scripts/localize_ckpt.sh <ckpt>/pretrained_model/config.json
```

The original value is kept in a `.orig_backbone_path` sidecar, so the edit is reversible.
