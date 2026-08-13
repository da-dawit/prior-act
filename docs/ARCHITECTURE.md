# Architecture

Author: Dawit Chun

## The model

Vanilla ACT is a CVAE transformer. During training a posterior `q(z | actions, state)` sees the
future action chunk; at inference that encoder is gone and `z = 0`. The latent is a training-time
regulariser and nothing else.

Prior-ACT adds a **learned prior conditioned on the scene**:

    q(z | a, s)      posterior   -- training only, sees the future actions
    p(z | image)     PRIOR       -- a frozen DINOv3 ViT-L/16 + a small MLP, available at inference

At inference `z` comes from `p`, so the policy can condition on where the objects are before it
commits to a trajectory. 355M total parameters, **53.5M trainable** — DINOv3 is frozen throughout.

For that to be worth anything, three things must all hold:

1. the decoder must actually **use** `z` (`latent_authority` > 0),
2. the prior must **predict** the posterior (`prior_post_gap` ≈ 0), otherwise the decoder is leaning
   on information it will not have at test time,
3. the prior must **vary with the scene** (`mu_prior_std` > 0), or it is a learned constant.

Two runs failed at different points, and in both cases the loss curve looked fine.

---

## Defect 1 — the decoder ignored the latent (`latent_injection="token"`)

With `latent_injection="token"`, `z` enters as **one token among 478** encoder tokens. The decoder
can ignore it for free by never attending to it, and it did:

    latent_authority   1.0e-4      zeroing z moved the action by 0.000012 rad
    kld_loss           1.9e-4
    mu_q_std           1.0e-3      the posterior had collapsed too

**Fix:** `latent_injection="decoder_seed"` — `z` seeds all 100 action queries instead of competing
for attention against 476 image patches. This is the only path by which `z` reaches the output
without winning an attention contest.

---

## Defect 2 — free bits silenced the prior's only gradient

This is why the second run performed worse than the first.

The KL uses per-dimension free bits:

```python
kld_opt = torch.clamp(kld_per_dim.mean(0), min=free_bits).sum()
loss    = l1_loss + kl_weight * kld_opt
```

Free bits is meant to stop the prior chasing the posterior to zero. But `clamp(x, min=f)` has
**zero gradient wherever `x < f`**. Measured on that run's checkpoint 1000:

    kld_loss   8.77 nats over 32 dims  =  0.274 nats/dim
    kl_free_bits                          0.5

**Every dimension sat below the floor.** So `kld_opt` was the constant `32 × 0.5 = 16.0`, the loss
term was a constant `160.000`, and `d(loss)/d(prior_mlp) = 0` exactly. The headline loss read
`160.xxx` for the entire run — that constant *was* the symptom, which reads as a converged KL term.

The prior never left initialisation:

    mu_prior_std      0.0315      near-identical output for every scene
    mu_q_std          0.4569      posterior spread 14x larger
    latent_authority  0.1642      but decoder_seed gave z real authority...
    prior_post_gap    0.1620      ...over information it only has at TRAINING time

99% of the latent's influence was a train/test shortcut. **Worse than the dead latent of run 1.**

### Why lowering the floor is not the fix

The symmetric KL has to serve two masters: keep the posterior from collapsing (wants a floor) and
teach the prior to predict it (wants gradient). A floor high enough for the first kills the second —
that is exactly what happened. A floor low enough for the second re-opens the collapse that made the
latent dead at 1e-4 in run 1. Retuning `kl_free_bits` just moves along that trade-off.

### The fix: a second KL term the floor cannot switch off

```python
# trains the PRIOR only -- q is detached, so the prior cannot make its own job easier
# by dragging the posterior toward a constant. NO free-bits floor.
prior_fit = KL( sg(q) || p ).sum(-1).mean()
loss = l1_loss + kl_weight * kld_opt + prior_fit_weight * prior_fit
```

The floored `KL(q‖p)` keeps doing what it did — it is the variational bound and it regularises the
posterior — and `prior_fit` adds a path to the prior that the floor cannot silence. Because `q` is
detached, the prior can never collapse the posterior to make itself right.

**Result, checkpoint 2000:**

| | before (run 2) | after |
|---|---|---|
| `latent_authority` | 0.1642 | 0.1489 |
| `prior_post_gap` | 0.1620 — **99%** of authority | **0.0032 — 2%** |
| `mu_prior_std` | 0.0315 | **0.0961** |
| `mu_q_std` | 0.4569 | 0.1451 |

Authority stayed high while the gap collapsed. The latent carries real information *and* the prior can reproduce it from the image alone, so the
influence survives to inference.

The KL is now 0.256 nats total = 0.008/dim, i.e. **below the 0.05 floor again** . Under the old design that would have silenced the term completely. It does not matter here because `prior_fit` carries no floor, which is why the terms are split rather than retuned.

---

## Defect 3 — the prior could not see *where* anything was

The prior read DINOv3's **CLS token** only. CLS is a global semantic summary; an object's *position*
is spatial information it largely discards. And the latent's whole job is to say where things are.

**Fix:** `scene_prior_features="cls_grid"` — CLS concatenated with the patch grid average-pooled to
2×3 cells (near/far × left/centre/right). Seven vectors instead of one. The patch tokens were already
being computed and thrown away, so the backbone cost is unchanged; only `prior_mlp`'s input widens
(355.0M → 356.6M parameters).

---

## A fourth defect, not fixed

`camera_identity_embedding` gives each camera a learned additive vector, because ACT is otherwise
permutation-invariant over cameras (swapping wrist_left and wrist_right changes the output by exactly
0.0000 rad — upstream ACT gives every camera the identical spatial position embedding).

It is implemented and enabled here, and it is **practically inert**: after 7000 steps the per-camera
norm is ~0.012 over 512 dims, and swapping the wrists shifts the action by **6.8e-6 rad** — a
thousandth of one demonstration step. Amplifying the embedding ×100 does change the output, so the
code path is live; it simply never trains.

The reason is structural: it is zero-initialised and **nothing in training requires it**. Cameras are
never swapped in the data, so the network fits everything without distinguishing them and the
embedding gets almost no gradient. It is reported rather than claimed as a fix.
