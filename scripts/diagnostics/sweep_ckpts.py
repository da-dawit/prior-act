"""Did Prior-ACT finish training, and is more training worth it?

Open-loop chunk error against the demonstrations, for every saved checkpoint. This is a better
convergence signal than the training loss: it is in radians, comparable to the 0.0071 rad the
demos move per step, and it is the quantity the robot actually cares about.

CAVEAT, and it is a big one: `train_config.json` has `dataset.episodes = None`, so ALL 90 episodes
went into training. There is no held-out set, so every number here is a TRAINING fit. It says
whether the model has converged; it says NOTHING about generalisation.
"""
import sys, glob, json, os
sys.path.insert(0, "/tmp/igen_sim/deploy_pkg")
import numpy as np, cv2, pandas as pd

DS="/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
ROOT="/home/robotis/robot_aiworker/act_prior_aiw"
CAMS=["observation.images.scene","observation.images.wrist_left","observation.images.wrist_right"]
KEYS=("scene","wrist_left","wrist_right")
info=json.load(open(f"{DS}/meta/info.json")); FPS=info["fps"]
em=pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet",recursive=True)[0])
df=pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet",recursive=True)[0])
ST=np.stack(df["observation.state"].to_numpy()).astype(np.float32)
AC=np.stack(df["action"].to_numpy()).astype(np.float64)

CACHE={}
def grab(ep,lf):
    k=(ep,lf)
    if k in CACHE: return CACHE[k]
    r=em.iloc[ep]; out=[]
    for c in KEYS:
        ci=int(r[f"videos/observation.images.{c}/chunk_index"]);fi=int(r[f"videos/observation.images.{c}/file_index"])
        t0=float(r[f"videos/observation.images.{c}/from_timestamp"])
        cap=cv2.VideoCapture(f"{DS}/videos/observation.images.{c}/chunk-{ci:03d}/file-{fi:03d}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t0*FPS))+lf);ok,im=cap.read();cap.release()
        if not ok: return None
        out.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    CACHE[k]=out; return out

EPS=[0,7,23,41,66,88]; PHASES=[0.10,0.25,0.40,0.55,0.70,0.85]
PROBES=[]
for e in EPS:
    r=em.iloc[e]; i0=int(r["dataset_from_index"]); T=int(r["dataset_to_index"])-i0
    for ph in PHASES:
        f=int(ph*T)
        if f+100<T: PROBES.append((e,i0,f))
print(f"{len(PROBES)} probes\n")

from act_prior_policy import ActPriorPolicy
A,H=14,100
print(f"{'step':>7} {'arm err':>9} {'arm@end':>9} {'grip err':>9} {'worst ep':>9}")
rows=[]
for ck in sorted(os.listdir(ROOT)):
    p=os.path.join(ROOT,ck)
    #010000 was written without its weights -- skip incomplete checkpoints rather than
    #dying halfway through the sweep.
    if not os.path.isfile(os.path.join(p,"pretrained_model","model.safetensors")): continue
    pol=ActPriorPolicy(p,camera_keys=CAMS,device="cuda")
    if hasattr(pol.policy.config,"deterministic_latent"): pol.policy.config.deterministic_latent=True
    per_ep={}
    arm=[];end=[];gr=[]
    for e,i0,f in PROBES:
        imgs=grab(e,f)
        if imgs is None: continue
        pol.policy.reset()
        c=np.asarray(pol.predict_chunk(images=imgs,state=ST[i0+f],instruction=""),np.float64)
        h=min(len(c),H); gt=AC[i0+f:i0+f+h]
        a=np.abs(c[:h,:A]-gt[:,:A]).mean(); arm.append(a); per_ep.setdefault(e,[]).append(a)
        end.append(np.abs(c[h-1,:A]-gt[h-1,:A]).mean())
        gr.append(np.abs(c[:h,A:]-gt[:,A:]).mean())
    worst=max(per_ep.items(), key=lambda kv: np.mean(kv[1]))
    print(f"{int(ck):>7} {np.mean(arm):>9.4f} {np.mean(end):>9.4f} {np.mean(gr):>9.4f} "
          f"{'ep'+str(worst[0]):>9} {np.mean(worst[1]):.4f}")
    rows.append((int(ck),float(np.mean(arm)),float(np.mean(end)),float(np.mean(gr))))
    del pol
    import torch; torch.cuda.empty_cache()
json.dump(rows, open("/tmp/igen_sim/pa/ckpt_sweep.json","w"), indent=1)
print(f"\nreference: one demo step = 0.0071 rad")
print("NOTE: all 90 episodes were in training (dataset.episodes=None) -- these are TRAINING fits.")
