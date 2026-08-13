# Results

Author: Dawit Chun

All numbers measured 2026-08-13 on the AI Worker bimanual pick-and-place dataset: 90 episodes,
96,003 frames, 30 Hz, 3 cameras (scene 376×672, wrists 424×240), 16-D state/action.

**Held out by episode**: `eval_split=0.111` → 10 unseen episodes, identical split, seed and budget
across both arms. One demonstration step is **0.0071 rad** — the natural unit for these numbers.

---

## Held-out action L1

| step | vanilla ACT | **Prior-ACT** |
|---|---|---|
| 500 | 0.3622 | **0.3238** |
| 1000 | 0.3392 | **0.3001** |
| 1500 | 0.2733 | **0.2432** |
| 2000 | 0.2459 | **0.2134** |
| 2500 | 0.2433 | **0.2050** |
| 3000 | 0.2289 | **0.1990** |
| 3500 | 0.2275 | 0.2016 |
| 4000 | 0.2313 | **0.1925** |
| 4500 | 0.2308 | **0.1885** |
| 5000 | 0.2261 | 0.1967 |
| 5500 | **0.2238** ← ACT best | 0.1928 |
| 6000 | 0.2263 | 0.1946 |
| 6500 | 0.2292 | 0.2067 |
| 7000 | 0.2333 | 0.1961 |
| 7500 | — | 0.1923 |
| 8000 | — | **0.1872** ← best |
| 8500 | — | 0.1924 |
| 9000 | — | 0.1967 |
| 9500 | — | 0.1877 |
| 10000 | — | 0.1883 |
| 10500 | — | 0.1925 |
| 11000 | — | 0.1906 |

**Prior-ACT is lower at every single matched step.**

| comparison | Prior-ACT | ACT | gap |
|---|---|---|---|
| best vs best | 0.1872 | 0.2238 | **−16.4%** |
| at matched step 7000 | 0.1961 | 0.2333 | **−15.9%** |
| plateau mean vs ACT best | ~0.195 | 0.2238 | **−12.9%** |

It also reaches ACT's *best-ever* value (0.2238) by **step 2000**, against ACT's 5500 — roughly
**2.7× fewer steps** to the same held-out error.

### Overfitting behaviour differs

Vanilla ACT **turns upward** after step 5500: 0.2238 → 0.2263 → 0.2292 → 0.2333, three consecutive
rises. Training was stopped there.

Prior-ACT does not. A linear fit over its steps 3000–9500 gives slope **−0.00093 per 1000 steps,
t = −1.32, p = 0.210** — not distinguishable from flat, and certainly not rising. Comparing
pre-resume (3000–7000, mean 0.1965) against post-resume (7500–9500, mean 0.1913) gives Welch
**p = 0.061** — suggestive of continued slow improvement, not established.

**Prior-ACT plateaus where ACT overfits.** Whether it is still slowly improving at
10k is unresolved with 14 evaluation points.

---

## Latent health

The numbers that say the architecture is doing what it claims. See `docs/DIAGNOSTICS.md` for how to
read them.

| | run 1 (`token`) | run 2 (`decoder_seed`, floor too high) | **this** |
|---|---|---|---|
| `latent_authority` | 1.0e-4 | 0.1642 | **0.1489** |
| `prior_post_gap` | — | 0.1620 (**99%** of authority) | **0.0032 (2%)** |
| `mu_prior_std` | — | 0.0315 | **0.0961** |
| `mu_q_std` | 1.0e-3 | 0.4569 | 0.1451 |
| `kld_loss` | 1.9e-4 | 8.77 nats | 0.256 nats |
| verdict | latent dead | **actively harmful** | **latent works** |

Progression at checkpoints 1000 → 2000: authority 0.1421 → 0.1489 (rising) while the gap fell
0.0053 → 0.0032. The latent gains influence *and* the prior tracks it better — the good direction,
and the specific risk `decoder_seed` carries if it went the other way.

### It survives to deployment

The above is measured in training mode. What matters is inference, where `z` comes from the prior:

```
different scene image -> commanded action shifts 0.0058-0.0081 rad  (~1 demonstration step)
inference cost                                    19 ms/chunk on a 4090
parameters                    356.6M total, 53.5M trainable (DINOv3 frozen)
```

---

## Real-robot behaviour (qualitative)

Run on the FFW_SG2 with `checkpoints/008000` and `011000`. **These are observations, not a measured
success rate**, and should be read as such:

* approaches the grasp point reliably; grasps succeed
* occasional misses on the approach
* with `--grip-hold 10` it held the grip too long and pressed into the basket → fixed with
  `--grip-hold 4`
* with the shipped `--max-vel 0.6` the approach was slow to turn → `0.8` with `--max-acc 6.0`
* turns quickly on close objects

Controller settings and the measurements behind them are in `docs/INFERENCE.md`.

---

## Caveats

**1. n = 1 seed per arm.** A seed replication was started and then cancelled. The ~15% gap has not
been shown to exceed seed noise. This is the first thing a reviewer will ask.

**2. The ablation is unrun.** Three things changed together — `prior_fit_weight=10`,
`scene_prior_features=cls_grid`, `kl_free_bits` 0.5→0.05. Which one produced the gain is unknown.
The theory in `docs/ARCHITECTURE.md` says `prior_fit_weight` is load-bearing (without it the prior
provably receives zero gradient), but that is an argument, not a measurement.

**3. Held-out L1 is not task success.** Every quantitative number here is action-matching on unseen
episodes. A previous run demonstrated the two can disagree: run 2 had a *better* eval curve than
vanilla ACT (0.2262 @2000) while its latent was measurably harmful. **Eval loss alone would have
selected the broken model.** It is the pairing of a better curve *with* a healthy `prior_post_gap`
that makes this result trustworthy.

**4. `state_dropout=0.5` costs eval loss by design.** Both arms carry it, so the comparison is fair,
but neither number is comparable to a run without it.

**5. Generalisation is not addressed.** The base policy's held-out error is 3.9× its
training-episode error (0.0362 vs 0.0094 rad). That gap is data coverage. Nothing here closes it.

**6. `camera_identity_embedding` is inert.** Enabled on both arms, but it trains to ~6.8e-6 rad of
effect. It is not part of why this works. See `docs/ARCHITECTURE.md`.
