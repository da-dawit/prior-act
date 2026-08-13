# Running Prior-ACT on the robot

Author: Dawit Chun

Every knob below has a **measured** effect on this setup. The numbers come from running the policy
on real held-out demonstration frames through the real controller, not from intuition.

## The command

```bash
python3 infer.py --policy act_prior \
  --policy-path /path/to/checkpoints/008000/pretrained_model \
  --residual "" --router-ip 10.42.0.22 \
  --trim-head 0 --seam-blend 12 --execute-steps 25 \
  --savgol 17 --savgol-order 3 --ensemble 4 --ensemble-decay 0.3 \
  --grip-lead 14 --grip-hold 4 \
  --max-vel 0.8 --max-acc 6.0 \
  --speed 1.0 --max-steps 1200 --live --home-first
```

`python3 prior_act.py infer <ckpt> --infer-py PATH` wraps it.

**Drop `--live` for a dry run** : it observes, infers, clamps and prints the plan without publishing. Do this first on any new checkpoint.

---

## Smoothing: what each knob actually buys

Measured on held-out episodes, real frames, jerk = mean |third difference| of the commanded
trajectory (the quantity perceived as shaking):

```
setting                             jerk  vs demo  track err   change
deployed baseline                0.001170     0.6x     0.0488   --
ensemble 8 (at execute-steps 25) 0.001170     0.6x     0.0488   +0.0%  EXACTLY NOTHING
seam-blend 12                    0.001105     0.6x     0.0495   -5.5% jerk, +1.5% tracking
savgol 17                        0.001052     0.5x     0.0488  -10.1% jerk, no tracking cost
execute-steps 12 + ensemble 8    0.001542     0.8x     0.0475  +31.8% jerk -- WORSE
no smoothing at all              0.011353     5.9x     0.0492   +870%
```

### `--ensemble` is capped by `--execute-steps`

Both `infer.py` and the controller average chunks with:

```python
for i, ch in enumerate(chunks):
    sh = execute_steps * i
    if sh >= len(ch): break
```

A chunk is 100 waypoints. At `--execute-steps 25` only `i = 0,1,2,3` are in range, so **`--ensemble 8`
averages 4 and silently discards the rest.** `--ensemble 4` is already the maximum useful value.

Re-planning more often to allow deeper averaging **makes it worse**, not better: `--execute-steps 12
--ensemble 8` measured +31.8% jerk, because twice as many re-plans means twice as many seams and the
extra averaging does not pay for them.

### `--savgol 17` over `--savgol 11`

Savitzky-Golay fits a degree-3 polynomial over the window; a moving average fits a *constant*, so it
pulls a turning trajectory toward the chord and blunts the turn. On a synthetic turn: true peak
1.0000, SavGol 0.9994, boxcar 0.9858.

A wider window still smooths more. If the approach feels sluggish or "slowly turns", `savgol
17` is part of the cost — drop to 11 and see. This trades smoothness against responsiveness.

### The commanded trajectory is already smoother than the demonstrations

Demonstration jerk is 0.001917; the deployed setting is 0.001170 — **0.6× the human's own**. If the
arm still visibly shakes, the shake is probably *not in the command* but downstream: servo tracking,
gains on the follower, or velocity clamping. More smoothing cannot remove a wobble the command does
not contain. Check the commanded-versus-observed joint trace before smoothing further.

---

## Speed: `--max-vel` and `--max-acc` must move together

The defaults throttle the demonstrated motion. Measured over 90 demonstrations, per control step,
the **fastest joint** in that step:

```
p50 0.230   p90 0.552   p99 0.874   max 9.99 rad/s

steps where the LEADING joint is clipped:
  max_vel 0.6 (default) ->  6.61%
  max_vel 0.8           ->  ~2%
  max_vel 1.0           ->  0.44%
```

At the default the entire upper decile of the motion is being cut .

**Acceleration is the one that bites.** `MAX_ACC` ships at **2.0 rad/s²** while the demonstrations'
leading joint does p50 **2.76**, p75 **4.14**, p90 **6.90**. So the brake was already weaker than the
data. Raising `max_vel` alone lets the arm reach a speed it cannot shed:

```
max_vel 0.6, max_acc 2.0  ->  stops in 300 ms, 90 mrad of overshoot
max_vel 1.0, max_acc 4.0  ->  stops in 250 ms, 125 mrad   <- this swings
max_vel 0.8, max_acc 6.0  ->  stops in 133 ms,  53 mrad   <- recommended
max_vel 1.0, max_acc 8.0  ->  stops in 125 ms,  63 mrad
```

If the arm swings, raise `--max-acc` before lowering `--max-vel`. `0.8 / 6.0` is both faster *and*
better braked than the shipped `0.6 / 2.0`.

`--speed` scales *time*, never angles, and `infer.py` says so directly: "speed alone does NOT slow
the approach — max_vel does."

> **Safety.** `--max-acc 6.0` is 3× the shipped limit. It is inside what the demonstrations do,
> but it is a real limit increase on real hardware. Keep the e-stop accessible on the first episode.

---

## Gripper timing

```
--grip-lead 14    look this many waypoints AHEAD to decide to CLOSE. Higher closes earlier.
--grip-hold 4     consecutive ticks the plan must agree to OPEN before it opens. Lower releases sooner.
```

The default `--grip-hold 10` exists for a measured reason: each gripper closes exactly **once** per
demonstration and holds a median **188 steps (6.3 s)**, never reopening mid-carry, so a high
threshold rejects brief flickers to "open" during transport.

It also stacks on top of a 25-waypoint agreement window, so release lags roughly a second behind the
policy's intent. On the robot this appears as the gripper holding past the intended release point. Dropping
to 4 releases sooner at the cost of some flicker protection.

If the object is dropped mid-carry, raise this toward 6-7.

The gripper is decoded as a **class**, not smoothed — averaging a gripper gives a value between open
and closed that means neither. Levels come from the data (`gripper_levels.json`): commanded
0.320/0.861 and 0.288/1.052. Commanding past the object's width is how a firm grip is produced; a
position-controlled gripper stalls against the object.

---

## Checkpoint hygiene

Dry-run a new checkpoint first (omit `--live`) and check three things:

```bash
python3 prior_act.py check <ckpt>/pretrained_model
```

1. **it loads** — if `config.json` still holds the training machine's DINOv3 path loading raises a
   HuggingFace "Repo id must be in the form namespace/repo_name" error. Run
   `prior_act.py localize`.
2. **it is deterministic** — two identical calls must return an identical chunk. If not,
   `state_dropout` is leaking into inference and the policy is dropping its own joint angles at inference.
3. **the prior is live** — changing the scene image must change the action. Measured here:
   **0.0058–0.0081 rad**, about one demonstration step. If it is ~0, the model is plain ACT with 303M parameters of overhead.

Inference cost: **19 ms per chunk** on a 4090, against a 33 ms budget at 30 Hz and roughly 1.2
re-plans/s at `--execute-steps 25`. DINOv3 does not constrain the control rate.
