"""Swapping the wrist views changed the output by exactly 0.0000. Does Prior-ACT read them at all?

If it does not, every claim about wrist mounting/rotation is moot and the policy stands or falls on
the SCENE camera alone -- which is the one view our simulator has to render convincingly.
"""
import sys, glob, json
sys.path.insert(0, "/tmp/igen_sim/deploy_pkg")
import numpy as np, cv2, pandas as pd
DS = "/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
CAMS = ["observation.images.scene", "observation.images.wrist_left", "observation.images.wrist_right"]
KEYS = ("scene", "wrist_left", "wrist_right")
info = json.load(open(f"{DS}/meta/info.json")); FPS = info["fps"]
ep_meta = pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet", recursive=True)[0])
df = pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet", recursive=True)[0])
ST = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
AC = np.stack(df["action"].to_numpy()).astype(np.float64)
def grab(ep, lf):
    row = ep_meta.iloc[ep]; out=[]
    for k in KEYS:
        ci=int(row[f"videos/observation.images.{k}/chunk_index"]); fi=int(row[f"videos/observation.images.{k}/file_index"])
        t0=float(row[f"videos/observation.images.{k}/from_timestamp"])
        cap=cv2.VideoCapture(f"{DS}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t0*FPS))+lf); ok,im=cap.read(); cap.release()
        if not ok: return None
        out.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    return out
from act_prior_policy import ActPriorPolicy
pol=ActPriorPolicy("/home/robotis/robot_aiworker/act_prior_aiw/033000",camera_keys=CAMS,device="cuda")
pol.policy.config.deterministic_latent=True
A=14
ABL={"intact":lambda v:v,
     "wrists BLACK":lambda v:[v[0],np.zeros_like(v[1]),np.zeros_like(v[2])],
     "scene BLACK":lambda v:[np.zeros_like(v[0]),v[1],v[2]],
     "wrists from a DIFFERENT episode":None,
     "state ZEROED":lambda v:v}
res={k:[] for k in ABL}
for e in (0,23,66):
    row=ep_meta.iloc[e]; i0=int(row["dataset_from_index"]); T=int(row["dataset_to_index"])-i0
    other=grab((e+40)%90,50)
    for ph in (0.15,0.35,0.55,0.75):
        f=int(ph*T)
        if f+100>=T: continue
        base=grab(e,f)
        if base is None or other is None: continue
        pol.policy.reset()
        ref=np.asarray(pol.predict_chunk(images=base,state=ST[i0+f],instruction=""),np.float64)
        for name in ABL:
            if name=="wrists from a DIFFERENT episode": v=[base[0],other[1],other[2]]
            elif name=="state ZEROED": v=base
            else: v=ABL[name]([x.copy() for x in base])
            st = np.zeros_like(ST[i0+f]) if name=="state ZEROED" else ST[i0+f]
            pol.policy.reset()
            c=np.asarray(pol.predict_chunk(images=v,state=st,instruction=""),np.float64)
            h=min(len(c),len(ref))
            res[name].append(np.abs(c[:h,:A]-ref[:h,:A]).mean())
print(f"\n{'ablation':<34} {'shift vs intact':>16}")
for k,v in res.items():
    print(f"{k:<34} {np.mean(v):>16.4f} rad")
print("\n0.0000 => that input is IGNORED.  reference: real-vs-sim scene gap was 0.0500 rad")
