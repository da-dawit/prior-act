# Training Prior-ACT

Author: Dawit Chun

## The command that produced the reported result

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<user/dataset> \
  --dataset.root=/path/to/dataset \
  --dataset.eval_split=0.111 --eval_steps=500 --max_eval_samples=512 \
  --dataset.image_transforms.enable=true \
  --policy.type=act_prior \
  --policy.push_to_hub=false --policy.use_amp=true \
  --policy.chunk_size=100 --policy.n_action_steps=100 \
  --policy.optimizer_lr=1e-5 --policy.optimizer_lr_backbone=1e-5 \
  --policy.camera_identity_embedding=true --policy.state_dropout=0.5 \
  --policy.scene_backbone=dinov3 \
  --policy.scene_backbone_path=/path/to/dinov3-vitl16-pretrain-lvd1689m \
  --policy.kl_weight=10.0 --policy.kl_free_bits=0.05 \
  --policy.prior_fit_weight=10.0 --policy.scene_prior_features=cls_grid \
  --policy.latent_injection=decoder_seed --policy.deterministic_latent=true \
  --policy.latent_diag_every=250 \
  --steps=20000 --batch_size=192 --num_workers=32 --seed=1000 \
  --save_freq=500 --log_freq=200 \
  --output_dir=out/prior_fit_grid --job_name=prior_fit_grid
```

`python3 prior_act.py train` wraps this.

---

## The arguments that matter, and why

### The three that make the prior work

| argument | value | why |
|---|---|---|
| `--policy.prior_fit_weight` | `10.0` | Adds `KL(sg(q) ‖ p)`, which trains the prior and **cannot be silenced by the free-bits floor**. Without it the prior receives literally zero gradient whenever every latent dim sits below the floor — which is the normal case. Matched to `kl_weight` so the two directions pull equally. This is the argument that makes the prior train. |
| `--policy.scene_prior_features` | `cls_grid` | The prior reads CLS *and* the patch grid pooled to 2×3, so it can see *where* objects are. CLS alone is a global semantic summary and is most of why `mu_prior_std` would not move. Default `cls` reproduces the old behaviour. |
| `--policy.latent_injection` | `decoder_seed` | `z` seeds all 100 action queries. With `token` it is 1 of 478 encoder tokens and the decoder ignores it for free — measured authority 1e-4. |

### `--policy.kl_free_bits = 0.05`, not 0.5

The floor must sit **below** the measured per-dimension KL or the term becomes a constant with zero
gradient. Check it against the run:

```
per-dim KL = kld_loss / latent_dim
```

If `kld_loss / 32 < kl_free_bits`, the KL term is inert. The indicator is a headline loss pinned at
`latent_dim × kl_free_bits × kl_weight`. Here it was `160.000`.

With `prior_fit_weight` set this is much less critical (the prior gets gradient regardless), but a
floor above the data still wastes the variational bound.

### `--policy.state_dropout = 0.5`

Drops the whole proprioceptive vector 50% of the time during training. Measured motivation: the
action head weighted its own joint angles **4.3× more** than the camera image, and
returned a 0.94-cosine-identical chunk for two different object layouts. This is causal confusion; state dropout is the counter.

It costs held-out loss by construction. Held-out eval is computed on episodes where the
proprioceptive shortcut is still available and still pays, so a policy denied it scores worse *by
construction* while being better at the thing that actually fails. It should not be tuned by eval_loss.

### `--dataset.image_transforms.enable = true`

Colour/brightness augmentation. Took eval 0.204 → 0.190 on an earlier arm. Free.

### `--policy.latent_diag_every = 250`

Logs `latent_authority`, `prior_post_gap`, `mu_prior_std`, `mu_q_std`, `sample_jitter` to wandb.
Costs two extra forwards on the steps it fires (~0.4% overhead). Without it a dead latent is invisible: `l1_loss` and `kld_loss` both fall and both look healthy while the prior does
nothing.

---

## Traps

**AMP is not the default.** `--policy.use_amp=true` is required; at batch 192 the model OOMs on a
24 GB card in fp32. It is also most of the speed.

**`--policy.push_to_hub=false`** or lerobot refuses to start, asking for a hub repo id.

**DINOv3 is licence-gated on HuggingFace.** Download it once and pass a local path. If a checkpoint
is moved between machines its `config.json` still records the *training* machine's path and loading
fails with a confusing HuggingFace error:

```
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/workspace/pretrained/...'
```

Fix with `python3 prior_act.py localize --dino DIR <ckpt>`.

**Hold out by episode, never by frame.** Frames within an episode are 30 Hz samples of one
trajectory; a random frame split puts near-duplicates on both sides and reports a score memorisation
alone can reach. `--dataset.eval_split=0.111` splits by episode (10 of 90 here).

---

## Multi-GPU

`batch_size` is **per process** — lerobot logs `Effective batch size: batch_size × num_processes`.
To use two GPUs as a pure speedup, halve it:

```bash
accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=no \
  -m lerobot.scripts.lerobot_train --config_path=<ckpt>/pretrained_model/train_config.json \
  --resume=true --batch_size=96 --num_workers=24 --save_freq=500
```

Leaving `batch_size=192` on 2 processes silently doubles the effective batch to 384, which changes
the effective learning rate and makes everything after the resume incomparable with everything
before it.

Measured: 1.3 step/s on one H100 → **1.9–3.0 step/s** on two. DDP emits a benign warning that it
found no unused parameters; the frozen DINOv3 is correctly excluded rather than stalling the
reduction.

`num_workers`: 24 per process on a 96-core box. At 32×2 the two processes contended and `data_s`
climbed from 0.28 to 1.15 s per step.

---

## How long, and when to stop

On this dataset (90 episodes, 96k frames) held-out loss falls steeply to ~step 4000 and then plateaus
around **0.195 ± 0.005**. A linear fit over steps 3000–9500 gives a slope of −0.00093 per 1000 steps
at **p = 0.21** — not distinguishable from flat. Vanilla ACT under the same settings *turns upward*
after step 5500 (0.2238 → 0.2333), i.e. it overfits where Prior-ACT does not.

Practical advice: **save every 500 steps.** The best eval (0.1872) landed at step 4500 on an earlier
arm where `save_freq=1000`, so the best loadable model was 0.1925.
