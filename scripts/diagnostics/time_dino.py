"""What does the inert DINOv3 branch cost, in time and memory?

Section 4c showed it changes the action by 1.2e-5 rad. If it is also a large fraction of the
forward pass, dropping it is free speed -- which matters for training throughput, not just
inference, because the ViT-L runs once per sample per step.
"""
import sys, time, glob, json
sys.path.insert(0,"/tmp/igen_sim/deploy_pkg")
import numpy as np, cv2, pandas as pd, torch
DS="/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
CAMS=["observation.images.scene","observation.images.wrist_left","observation.images.wrist_right"]
KEYS=("scene","wrist_left","wrist_right")
info=json.load(open(f"{DS}/meta/info.json")); FPS=info["fps"]
em=pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet",recursive=True)[0])
df=pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet",recursive=True)[0])
ST=np.stack(df["observation.state"].to_numpy()).astype(np.float32)
r=em.iloc[0]; i0=int(r["dataset_from_index"]); f=300
imgs=[]
for k in KEYS:
    ci=int(r[f"videos/observation.images.{k}/chunk_index"]);fi=int(r[f"videos/observation.images.{k}/file_index"])
    t0=float(r[f"videos/observation.images.{k}/from_timestamp"])
    cap=cv2.VideoCapture(f"{DS}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t0*FPS))+f);_,im=cap.read();cap.release()
    imgs.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
from act_prior_policy import ActPriorPolicy
pol=ActPriorPolicy("/home/robotis/robot_aiworker/act_prior_aiw/033000",camera_keys=CAMS,device="cuda")
pol.policy.config.deterministic_latent=True
m=pol.policy.model
n_all=sum(p.numel() for p in pol.policy.parameters())
n_dino=sum(p.numel() for p in m.dino.parameters())
print(f"params: total {n_all/1e6:.1f}M   dino {n_dino/1e6:.1f}M ({100*n_dino/n_all:.0f}%)   rest {(n_all-n_dino)/1e6:.1f}M")

def bench(label, patch):
    if patch:
        _enc=m._encode_scene_prior
        z=torch.zeros(1,m.config.latent_dim,device=pol.dev)
        m._encode_scene_prior=lambda i:(z,z)
    for _ in range(3):
        pol.policy.reset(); pol.predict_chunk(images=imgs,state=ST[i0+f],instruction="")
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(15):
        pol.policy.reset(); pol.predict_chunk(images=imgs,state=ST[i0+f],instruction="")
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/15
    if patch: m._encode_scene_prior=_enc
    print(f"{label:<28} {dt*1000:7.1f} ms/chunk")
    return dt
a=bench("with DINOv3 (as shipped)", False)
b=bench("DINOv3 branch bypassed", True)
print(f"\nDINOv3 costs {(a-b)*1000:.1f} ms/chunk = {100*(a-b)/a:.0f}% of the forward pass")
print(f"for an action worth {0.000012:.6f} rad (one demo step = 0.0071)")
