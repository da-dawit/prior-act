# Tuning the scene prior

Author: Dawit Chun

How to make the prior influence the policy more or less, and how to tell which way to move.

## Read the diagnostics first

Do not tune blind. Every decision below is made from three numbers, logged during training with
`--policy.latent_diag_every=250` and computable post-hoc on any checkpoint:

```bash
python3 prior_act.py check <ckpt>/pretrained_model
```

| number | meaning |
|---|---|
| `latent_authority` | how far the action moves when `z` goes from the posterior mean to zero. How much the decoder uses the latent at all. |
| `prior_post_gap` | how far the action moves when `z` goes from the posterior mean to the prior mean. The train/test mismatch, in action units. |
| `mu_prior_std` | spread of the prior's output across a batch of different scenes. Whether the prior varies with the image. |
| `mu_q_std` | spread of the posterior mean. Near zero means the posterior has collapsed. |

Target state:

```text
latent_authority   well above 0        the decoder uses z
prior_post_gap     small fraction of authority (under ~10%)
mu_prior_std       a useful fraction of mu_q_std (measured: 0.096 vs 0.145, 66%)
```

## Diagnosing from the numbers

| symptom | meaning | fix |
|---|---|---|
| `latent_authority` ≈ 0 | the decoder ignores `z`; the backbone is dead weight | `latent_injection=decoder_seed`; raise `latent_dim` |
| authority high, `prior_post_gap` ≈ authority | the decoder leans on training-only information | raise `prior_fit_weight` |
| `mu_prior_std` ≈ 0 | the prior emits a constant regardless of scene | raise `prior_fit_weight`; `scene_prior_features=cls_grid` |
| `mu_q_std` ≈ 0 | posterior collapse | lower `kl_weight`, or raise `kl_free_bits` |
| loss pinned at `latent_dim × kl_free_bits × kl_weight` | free bits has made the KL a constant with no gradient | lower `kl_free_bits` |

That last row is worth checking explicitly. If the headline loss sits at a suspiciously round
constant, compute `latent_dim × kl_free_bits × kl_weight`. If it matches, every latent dimension is
below the floor and the KL term contributes no gradient at all.

## To make the prior influence MORE

In order of effect.

### 1. `prior_fit_weight` (default 10.0)

The dominant knob. It weights `KL(sg(q) ‖ p)`, which trains the prior to predict the posterior and
carries no free-bits floor, so it cannot be silenced.

- **Raise** (20–50) if `prior_post_gap` is a large fraction of `latent_authority`. This forces the
  prior to track the posterior, so the latent's influence survives to inference.
- Set to **0** to disable and reproduce the standard symmetric-KL behaviour.
- Raising it very high makes the prior chase per-sample posterior noise. If `mu_prior_std` climbs
  above `mu_q_std`, it has gone too far.

### 2. `latent_injection` (default `decoder_seed`)

- `decoder_seed` — `z` seeds all action queries. The decoder cannot avoid it.
- `token` — `z` is one encoder token among hundreds. The decoder can ignore it for free, and on this
  data it did: authority 1e-4.

Use `decoder_seed` whenever the latent should matter.

### 3. `scene_prior_features` (default `cls_grid`)

- `cls` — the CLS token only. A global semantic summary; discards most spatial information.
- `cls_grid` — CLS plus the patch grid average-pooled to 2×3 cells. Gives the prior coarse
  *position* information.

Use `cls_grid` when the latent should encode where things are. The backbone cost is unchanged; the
patch tokens are computed either way.

### 4. `kl_free_bits` (default 0.05)

Per-dimension floor below which the KL contributes no gradient. It must sit **below** the measured
per-dimension KL (`kld_loss / latent_dim`) or the term is inert.

- Lower it (0.01, or 0) to let the KL pull the posterior toward the prior harder.
- Raise it to protect the posterior from collapse when `mu_q_std` is falling.

### 5. `latent_dim` (default 32)

More capacity in the channel. Raise to 64 if authority is healthy and the prior tracks well but there is reason to think
suspect the latent is saturated. Costs a wider `prior_mlp` head only.

### 6. `scene_prior_cameras`

By default the prior reads camera 0. Pass a comma-separated list of dataset image keys to fuse
several static views. `scene_fusion=concat` widens the prior input; `mean` (default) keeps it fixed
so older checkpoints still load.

## To make the prior influence LESS

Reverse the above: `prior_fit_weight=0`, `latent_injection=token`, `scene_prior_features=cls`. To
remove it entirely, train `--policy.type=act`.

## Interactions to be aware of

**`kl_weight` and `prior_fit_weight` pull in opposite directions.** `kl_weight` moves the posterior
toward the prior; `prior_fit_weight` moves the prior toward the posterior. Keeping them equal (both
10.0) is a reasonable starting point. If the posterior collapses, the first is too strong relative to
the second.

**`kl_free_bits` and `prior_fit_weight` are not substitutes.** The floor protects the posterior; the
`prior_fit` term trains the prior. Before `prior_fit_weight` existed, a floor high enough to protect
the posterior also silenced the prior's only gradient, and there was no setting that did both.

**`deterministic_latent`** uses `z = mu` rather than sampling at inference. Keep it `true` for
reproducible robot behaviour; two identical observations should give an identical chunk.

## Verifying a change reached the robot

Training diagnostics are computed in training mode, where `z` comes from the posterior. At inference
`z` comes from the prior. Confirm the prior actually drives the deployed action:

```bash
python3 prior_act.py check <ckpt>/pretrained_model
```

Feeds two different scene images with an identical state and reports the change in the commanded
chunk. Here it is 0.0058–0.0081 rad, roughly one demonstration step. A value near zero means
the prior is not influencing deployment regardless of what the training diagnostics said.
