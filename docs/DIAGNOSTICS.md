# Diagnostics

Author: Dawit Chun

A scene prior can be inert while every training loss looks healthy. These tools distinguish a
working latent from a decorative one.

## Why loss curves are not enough

On an earlier run `l1_loss` and `kld_loss` both fell smoothly for 33k steps and the latent was dead:
zeroing it moved the action by 0.000012 rad. On a later run the held-out loss was *better* than
vanilla ACT while 99% of the latent's influence was training-only information. Selecting on eval loss
alone would have picked that model.

## latent_health.py

```bash
python3 scripts/diagnostics/latent_health.py <ckpt>/pretrained_model
```

Reports, over a batch of real observations:

| | |
|---|---|
| `latent_authority` | `\|a(z=mu_q) − a(z=0)\|`. Decoder ignores `z` if ≈ 0. |
| `prior_post_gap` | `\|a(z=mu_q) − a(z=mu_p)\|`. Train/test mismatch in action units. |
| `mu_prior_std` | spread of the prior mean across scenes. ≈ 0 means a learned constant. |
| `mu_q_std` | spread of the posterior mean. ≈ 0 means posterior collapse. |
| `sample_jitter` | action change from resampling `z`. |

Reference values from a working model (checkpoint 2000):

```text
latent_authority   0.1489
prior_post_gap     0.0032      2% of authority
mu_prior_std       0.0961
mu_q_std           0.1451
```

Two failure signatures, both seen on this data:

```text
authority 1.0e-4                       latent dead; the backbone is decoration
authority 0.1642, gap 0.1620 (99%)     train/test shortcut; WORSE than dead
```

### Probes must run in eval mode

`nn.MultiheadAttention` stores dropout as a float attribute, not an `nn.Dropout` submodule, so it
stays active in training mode regardless of what is done to the module tree. Two training-mode
forwards differ by ~0.007 in normalised action units from dropout alone, which is larger than the
effect being measured. An earlier version of these diagnostics reported authority, prior gap and
sampling jitter as three nearly identical numbers because all three were measuring dropout.

The probes therefore run in eval mode and anchor on `a(z = mu_q)` rather than on a sampled `z`.

## test_scene_dependence.py

```bash
python3 scripts/diagnostics/test_scene_dependence.py <ckpt>/pretrained_model
```

Training diagnostics are computed where `z` comes from the posterior. At inference it comes from the
prior. This feeds two different scene images with an identical state and reports the change in the
commanded chunk — the only check that the prior influences deployed behaviour.

Reference: 0.0058–0.0081 rad, roughly one demonstration step. Near zero means plain ACT with extra
parameters.

It also verifies:

- the checkpoint loads (catches an unlocalised DINOv3 path)
- output is deterministic across identical calls (catches `state_dropout` leaking into inference)
- inference latency

## Other tools

| | |
|---|---|
| `eval_openloop.py` | action error against demonstrations over a chunk |
| `eval_heldout.py` | the same on held-out episodes only |
| `sweep_ckpts.py` | error against training step, for a convergence curve |
| `test_camdep.py` | camera dependence: swap two cameras and measure the action change |
| `attn_viz.py` | decoder cross-attention and DINOv3 CLS attention overlays |
| `time_dino.py` | backbone cost per forward |
