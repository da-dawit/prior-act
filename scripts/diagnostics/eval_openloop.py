"""Open-loop accuracy of Prior-ACT against the demonstrations themselves.

One frame is an anecdote. This walks several episodes at several phases, feeds the REAL
observation, and compares the predicted chunk to what the operator actually commanded next.
Arm and gripper are scored separately because they fail differently: the arm is a regression,
the gripper is effectively a two-class decision and an averaged error hides a wrong class.

LeRobot v3 layout: ONE parquet for all 90 episodes, videos split into chunk files with a
per-episode (file_index, from_timestamp). Seeking by global frame index would read the wrong
episode entirely, which is why the resolver below exists.
"""
import sys, glob, json
sys.path.insert(0, "/tmp/igen_sim/deploy_pkg")
import numpy as np, cv2, pandas as pd

DS = "/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/robotis/robot_aiworker/act_prior_aiw/033000"
CAMS = ["observation.images.scene", "observation.images.wrist_left", "observation.images.wrist_right"]
KEYS = ("scene", "wrist_left", "wrist_right")

info = json.load(open(f"{DS}/meta/info.json"))
FPS = info["fps"]
ep_meta = pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet", recursive=True)[0])
df = pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet", recursive=True)[0])
ST = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
AC = np.stack(df["action"].to_numpy()).astype(np.float64)
print(f"fps {FPS} | {len(ep_meta)} episodes | {len(df)} frames | action dim {AC.shape[1]}")

def grab(ep, local_f):
    """Frame `local_f` of episode `ep`, for each camera."""
    row = ep_meta.iloc[ep]
    out = []
    for k in KEYS:
        ci = int(row[f"videos/observation.images.{k}/chunk_index"])
        fi = int(row[f"videos/observation.images.{k}/file_index"])
        t0 = float(row[f"videos/observation.images.{k}/from_timestamp"])
        path = f"{DS}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4"
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)) + local_f)
        ok, im = cap.read(); cap.release()
        if not ok: return None
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out

from act_prior_policy import ActPriorPolicy
pol = ActPriorPolicy(CKPT, camera_keys=CAMS, device="cuda")
if hasattr(pol.policy.config, "deterministic_latent"):
    pol.policy.config.deterministic_latent = True      #reproducible; costs 0.0001 rad

EPS = [0, 7, 23, 41, 66, 88]
PHASES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
A, H = 14, 100
rows = []
for e in EPS:
    row = ep_meta.iloc[e]
    i0, i1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    T = i1 - i0
    for ph in PHASES:
        f = int(ph * T)
        if f + H >= T: continue
        imgs = grab(e, f)
        if imgs is None: continue
        pol.policy.reset()
        c = np.asarray(pol.predict_chunk(images=imgs, state=ST[i0 + f], instruction=""), np.float64)
        h = min(len(c), H)
        gt = AC[i0 + f: i0 + f + h]
        rows.append((e, ph,
                     np.abs(c[:h, :A] - gt[:, :A]).mean(),
                     np.abs(c[h-1, :A] - gt[h-1, :A]).mean(),
                     np.abs(c[:h, A:] - gt[:, A:]).mean(),
                     c[:h, A:].mean(), gt[:, A:].mean()))

r = np.array([x[2:] for x in rows])
print(f"\n{'ep':>3} {'phase':>6} {'arm err':>9} {'arm@end':>9} {'grip err':>9} {'grip pred':>10} {'grip gt':>8}")
for x in rows:
    print(f"{x[0]:>3} {x[1]:>6.2f} {x[2]:>9.4f} {x[3]:>9.4f} {x[4]:>9.4f} {x[5]:>10.3f} {x[6]:>8.3f}")
print(f"\nMEAN over {len(rows)} probes: arm {r[:,0].mean():.4f} rad  arm@end {r[:,1].mean():.4f} rad  "
      f"grip {r[:,2].mean():.4f}")
print(f"reference: one demo step = 0.0071 rad; the 100-step chunk spans {H/FPS:.1f} s")
