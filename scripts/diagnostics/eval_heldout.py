"""Open-loop error on the HELD-OUT episodes, in radians.

lerobot's eval_split holds out the LAST ceil(n*split) episodes per task, so with 90 episodes and
eval_split=0.111 that is episodes 80-89. Those ten are the only honest test set we have.

READ THE TWO ROWS DIFFERENTLY. act_probe never saw these episodes. Prior-ACT 033000 trained on all
90 (`dataset.episodes = None`), so for IT these are training data and its number is an upper bound
on how good it really is -- that is exactly the confound the A/B exists to remove.
"""
import sys, glob, json
sys.path.insert(0,"/tmp/igen_sim/deploy_pkg")
import numpy as np, cv2, pandas as pd
DS="/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
CAMS=["observation.images.scene","observation.images.wrist_left","observation.images.wrist_right"]
KEYS=("scene","wrist_left","wrist_right")
info=json.load(open(f"{DS}/meta/info.json")); FPS=info["fps"]
em=pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet",recursive=True)[0])
df=pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet",recursive=True)[0])
ST=np.stack(df["observation.state"].to_numpy()).astype(np.float32)
AC=np.stack(df["action"].to_numpy()).astype(np.float64)
CACHE={}
def grab(ep,lf):
    if (ep,lf) in CACHE: return CACHE[(ep,lf)]
    r=em.iloc[ep]; out=[]
    for k in KEYS:
        ci=int(r[f"videos/observation.images.{k}/chunk_index"]);fi=int(r[f"videos/observation.images.{k}/file_index"])
        t0=float(r[f"videos/observation.images.{k}/from_timestamp"])
        cap=cv2.VideoCapture(f"{DS}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t0*FPS))+lf);ok,im=cap.read();cap.release()
        if not ok: return None
        out.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    CACHE[(ep,lf)]=out; return out

HELD=list(range(80,90))
PHASES=[0.10,0.25,0.40,0.55,0.70,0.85]
A=14
from act_prior_policy import ActPriorPolicy
for label,ck,seen in (("act_probe  (4k steps, NEVER saw these)", "/home/robotis/robot_aiworker/act_ab/act_probe/checkpoints/last/pretrained_model", False),
                      ("prior-ACT 033000 (trained ON these)",   "/home/robotis/robot_aiworker/act_prior_aiw/033000", True)):
    try:
        pol=ActPriorPolicy(ck,camera_keys=CAMS,device="cuda")
    except Exception as e:
        print(f"{label}: could not load -- {e}"); continue
    if hasattr(pol.policy.config,"deterministic_latent"): pol.policy.config.deterministic_latent=True
    H=pol.policy.config.chunk_size
    arm=[];gr=[]
    for e in HELD:
        r=em.iloc[e]; i0=int(r["dataset_from_index"]); T=int(r["dataset_to_index"])-i0
        for ph in PHASES:
            f=int(ph*T)
            if f+H>=T: continue
            imgs=grab(e,f)
            if imgs is None: continue
            pol.policy.reset()
            c=np.asarray(pol.predict_chunk(images=imgs,state=ST[i0+f],instruction=""),np.float64)
            h=min(len(c),H); gt=AC[i0+f:i0+f+h]
            arm.append(np.abs(c[:h,:A]-gt[:,:A]).mean()); gr.append(np.abs(c[:h,A:]-gt[:,A:]).mean())
    print(f"\n{label}")
    print(f"   arm {np.mean(arm):.4f} rad   gripper {np.mean(gr):.4f}   ({len(arm)} probes on eps 80-89)")
    del pol
    import torch; torch.cuda.empty_cache()
print(f"\nreference: one demonstration step = 0.0071 rad")
