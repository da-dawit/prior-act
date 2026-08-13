"""Verdict on whether the latent branch is doing anything, for any act_prior checkpoint.

Reads the three numbers `latent_diag_every` logs during training, but post-hoc, so an existing
checkpoint can be judged without retraining. Run it on the old checkpoint and the new one and
the comparison is the whole argument.

  latent_authority  |a(z=mu_q) - a(z=0)|      decoder ignores z if ~0
  prior_post_gap    |a(z=mu_q) - a(z=mu_p)|   the train/test mismatch, in action units
  mu_q_std          spread of the posterior mean across probes; ~0 = collapsed
  kld               against the learned prior, and against N(0,I) for scale
"""
import sys, glob, json
sys.path.insert(0,"/tmp/igen_sim/deploy_pkg")
import numpy as np, cv2, pandas as pd, torch
DS="/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
CAMS=["observation.images.scene","observation.images.wrist_left","observation.images.wrist_right"]
KEYS=("scene","wrist_left","wrist_right")
CKPT=sys.argv[1] if len(sys.argv)>1 else "/home/robotis/robot_aiworker/act_prior_aiw/033000"
info=json.load(open(f"{DS}/meta/info.json")); FPS=info["fps"]
em=pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet",recursive=True)[0])
df=pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet",recursive=True)[0])
ST=np.stack(df["observation.state"].to_numpy()).astype(np.float32)
AC=np.stack(df["action"].to_numpy()).astype(np.float32)
def grab(ep,lf):
    r=em.iloc[ep]; out=[]
    for k in KEYS:
        ci=int(r[f"videos/observation.images.{k}/chunk_index"]);fi=int(r[f"videos/observation.images.{k}/file_index"])
        t0=float(r[f"videos/observation.images.{k}/from_timestamp"])
        cap=cv2.VideoCapture(f"{DS}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t0*FPS))+lf);ok,im=cap.read();cap.release()
        if not ok: return None
        out.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    return out
from act_prior_policy import ActPriorPolicy
pol=ActPriorPolicy(CKPT,camera_keys=CAMS,device="cuda")
P=pol.policy; m=P.model; H=P.config.chunk_size
P.config.latent_diag_every=1
P.train()                                   #the diagnostics gate on the POLICY's training flag, not the model's
for mod in m.modules():
    if isinstance(mod,torch.nn.Dropout): mod.p=0.0
print(f"injection={getattr(P.config,'latent_injection','token')}  "
      f"kl_weight={P.config.kl_weight}  kl_free_bits={getattr(P.config,'kl_free_bits',0.0)}\n")

#BATCH THE PROBES. In train mode ResNet18's BatchNorm uses BATCH statistics, so a batch of 1
#normalises every channel to zero mean per sample and destroys the image features -- which made an
#earlier version of this script report the latent as 500x more influential than it is. Training
#runs at batch 192; measure under the same conditions or do not measure at all.
BATCH = 8
items=[]
for e in (0,7,23,41,66,88):
    r=em.iloc[e]; i0=int(r["dataset_from_index"]); T=int(r["dataset_to_index"])-i0
    for ph in (0.15,0.40,0.65,0.85):
        f=int(ph*T)
        if f+H<T: items.append((e,i0,f))
acc={}
n=0
for s0 in range(0, len(items)-BATCH+1, BATCH):
    grp=items[s0:s0+BATCH]
    obs_list=[]
    ok=True
    for e,i0,f in grp:
        imgs=grab(e,f)
        if imgs is None: ok=False; break
        o={"observation.state":torch.as_tensor(ST[i0+f]).reshape(-1)}
        for k,im in zip(CAMS,imgs):
            o[k]=torch.from_numpy(np.ascontiguousarray(im)).permute(2,0,1).float()/255.
        o["task"]=""; o["action"]=torch.as_tensor(AC[i0+f:i0+f+H])
        obs_list.append(pol.pre(o))
    if not ok: continue
    #Per-key expected rank WITHOUT the batch dim: images (C,H,W)=3, state (D,)=1, action (T,D)=2.
    #Getting this wrong is silent -- an action chunk that is already 2-D concatenates along TIME
    #and produces a (800,16) tensor the VAE encoder happily reshapes into nonsense.
    RANK={**{k:3 for k in CAMS}, "observation.state":1, "action":2}
    b={}
    for k,r in RANK.items():
        vs=[(o[k] if o[k].ndim==r+1 else o[k].unsqueeze(0)) for o in obs_list]
        b[k]=torch.cat(vs,0)
        assert b[k].shape[0]==len(grp), f"{k} batched to {b[k].shape}"
    b["task"]=[""]*len(grp)
    b["observation.images"]=[b[k] for k in CAMS]
    b["action_is_pad"]=torch.zeros(b["action"].shape[:2],dtype=torch.bool,device=pol.dev)
    with torch.no_grad():
        _,ld=P.forward(b)
    for k,v in ld.items(): acc.setdefault(k,[]).append(v)
    n+=len(grp)

print(f"{n} probes\n")
for k in ("l1_loss","kld_loss","latent_authority","prior_post_gap","sample_jitter","mu_q_std","mu_prior_std","mu_q_norm"):
    if k in acc: print(f"  {k:<18} {np.mean(acc[k]):.6f}")

a=np.mean(acc.get("latent_authority",[0])); g=np.mean(acc.get("prior_post_gap",[0]))
s=np.mean(acc.get("mu_q_std",[0])); kl=np.mean(acc.get("kld_loss",[0]))
print("\nVERDICT")
if a < 1e-3:
    print(f"  latent is DEAD: authority {a:.2e} in normalized action units.")
    print(f"  The prior branch is decoration; kld {kl:.2e} is the prior catching a collapsed")
    print(f"  posterior (mu_q std {s:.2e}), not a converged objective.")
elif g > 0.5*a:
    print(f"  latent has authority {a:.4f} but the prior only predicts the posterior to {g:.4f}")
    print(f"  ({100*g/a:.0f}% of the authority is train/test mismatch) -- ACTIVELY HARMFUL.")
else:
    print(f"  latent WORKS: authority {a:.4f}, prior/posterior gap {g:.4f} ({100*g/a:.0f}% of it).")
