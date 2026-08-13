"""Where does the policy actually look?  ("the highlighting tool")

Two different attentions, and they answer different questions:

  1. DINOv3 CLS attention -- what the FROZEN scene prior attends to. This is the one Dawit's
     professor is sceptical of: a self-supervised ViT has no reason to attend to the object you
     care about, and DINOv2 in particular is known to dump huge attention onto uninformative
     background patches (the "register" artifacts, Darcet et al. 2023). Worth LOOKING at rather
     than assuming either way.

  2. ACT decoder cross-attention -- where the 100 action queries read from the image tokens when
     they produce the chunk. This is the one that decides the motion, and it is trained, not
     self-supervised. If this lands on the bottle, the policy is grounded regardless of what
     DINOv3 does.

Output: one PNG per probe, original | DINOv3 CLS | decoder cross-attn, per camera.
"""
import sys, glob, json, argparse
sys.path.insert(0, "/tmp/igen_sim/deploy_pkg"); sys.path.insert(0, "/tmp/igen_sim")
import numpy as np, cv2, pandas as pd, torch, torch.nn as nn

DS = "/home/robotis/robot_aiworker/datasets/aiw_pp_2bttles_lerobot"
CAMS = ["observation.images.scene", "observation.images.wrist_left", "observation.images.wrist_right"]
KEYS = ("scene", "wrist_left", "wrist_right")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/robotis/robot_aiworker/act_prior_aiw/033000")
ap.add_argument("--out", default="/tmp/igen_sim/pa/attn")
ap.add_argument("--probes", default="0:0.15,0:0.40,23:0.55,66:0.75")
a = ap.parse_args()
import os; os.makedirs(a.out, exist_ok=True)

info = json.load(open(f"{DS}/meta/info.json")); FPS = info["fps"]
ep_meta = pd.read_parquet(glob.glob(f"{DS}/meta/episodes/**/*.parquet", recursive=True)[0])
df = pd.read_parquet(glob.glob(f"{DS}/data/**/*.parquet", recursive=True)[0])
ST = np.stack(df["observation.state"].to_numpy()).astype(np.float32)

def grab(ep, lf):
    row = ep_meta.iloc[ep]; out = []
    for k in KEYS:
        ci = int(row[f"videos/observation.images.{k}/chunk_index"]); fi = int(row[f"videos/observation.images.{k}/file_index"])
        t0 = float(row[f"videos/observation.images.{k}/from_timestamp"])
        cap = cv2.VideoCapture(f"{DS}/videos/observation.images.{k}/chunk-{ci:03d}/file-{fi:03d}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * FPS)) + lf); ok, im = cap.read(); cap.release()
        if not ok: return None
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out

from act_prior_policy import ActPriorPolicy
pol = ActPriorPolicy(a.ckpt, camera_keys=CAMS, device="cuda")
if hasattr(pol.policy.config, "deterministic_latent"):
    pol.policy.config.deterministic_latent = True
model = pol.policy.model

#--- capture 1: decoder cross-attention -----------------------------------------------------------
#nn.MultiheadAttention discards the weights unless asked. Wrap forward so it always returns them,
#averaged over heads, and stash the result. Restored implicitly by only ever wrapping once.
CAP = {}
mha = model.decoder.layers[0].multihead_attn
_orig = mha.forward
def _fwd(*args, **kw):
    kw["need_weights"] = True; kw["average_attn_weights"] = True
    out, w = _orig(*args, **kw)
    CAP["cross"] = w.detach()          #(B, n_queries, n_kv)
    return out, w
mha.forward = _fwd

#--- capture 2: DINOv3 CLS attention --------------------------------------------------------------
def dino_cls_attn(img_uint8):
    """(H,W) map of how much the CLS token attends to each patch, last block, heads averaged."""
    import torchvision.transforms as T
    x = torch.from_numpy(img_uint8).permute(2, 0, 1).float().div(255.)[None].to(pol.dev)
    x = T.Resize((224, 224), antialias=True)(x)
    x = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
    hf = model.dino.m
    #SDPA silently refuses to hand back attention weights. Switch this copy to the eager kernel;
    #it is slower but this is an offline diagnostic, and the alternative is no map at all.
    hf.config._attn_implementation = "eager"
    for mod in hf.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "eager"
        if hasattr(mod, "attention_interface"):
            mod.attention_interface = None
    with torch.no_grad():
        o = hf(pixel_values=x, output_attentions=True)
    if not o.attentions:
        raise SystemExit("DINOv3 returned no attentions even under eager -- check transformers version")
    at = o.attentions[-1][0].mean(0)                 #(T, T) heads averaged
    cls = at[0]                                      #CLS row over every token
    n = int(cls.shape[0])
    #DINOv3 prepends CLS *and register tokens*; the patch grid is the trailing square. Solve for it
    #rather than assuming a register count, which differs between v2 and v3 and between sizes.
    g = int((n) ** 0.5)
    while g > 0 and g * g > n - 1:
        g -= 1
    patches = cls[n - g * g:]
    print(f"    DINOv3: {n} tokens, {n - g*g} prefix (CLS+registers), {g}x{g} patch grid")
    return patches.reshape(g, g).float().cpu().numpy()

def heat(base_rgb, m):
    m = m - m.min(); m = m / (m.max() + 1e-9)
    m = cv2.resize(m.astype(np.float32), (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return (0.55 * base_rgb.astype(np.float32) + 0.45 * hm.astype(np.float32)).astype(np.uint8)

def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

from preprocess import preprocess_views_actprior
for spec in a.probes.split(","):
    e, ph = spec.split(":"); e = int(e); ph = float(ph)
    row = ep_meta.iloc[e]; i0 = int(row["dataset_from_index"]); T_ = int(row["dataset_to_index"]) - i0
    f = int(ph * T_)
    base = grab(e, f)
    if base is None: print(f"skip {spec}"); continue
    pol.policy.reset()
    _ = pol.predict_chunk(images=base, state=ST[i0 + f], instruction="")
    cross = CAP["cross"][0]                              #(100 queries, n_kv)

    #token layout: [latent, state, scene..., wrist_l..., wrist_r...] -- recover each grid by
    #running the backbone, rather than assuming stride 32, so a config change cannot silently
    #misalign the overlay onto the wrong camera.
    grids = []
    with torch.no_grad():
        for im in base:
            t = torch.from_numpy(np.ascontiguousarray(im)).permute(2,0,1).float().div(255.)[None].to(pol.dev)
            fm = model.backbone(t)["feature_map"]
            grids.append((fm.shape[-2], fm.shape[-1]))
    n_pre = cross.shape[1] - sum(h*w for h, w in grids)
    print(f"ep{e} phase{ph}: {cross.shape[1]} kv tokens = {n_pre} non-image + grids {grids}")

    panels = []
    off = n_pre
    for ci, (k, im) in enumerate(zip(KEYS, base)):
        h, w = grids[ci]
        q = cross[:, off:off + h*w].mean(0).reshape(h, w).float().cpu().numpy()   #avg over queries
        off += h*w
        col = [label(im, f"{k}  (raw)"), label(heat(im, q), f"{k}  ACT decoder cross-attn")]
        if ci == 0:
            col.append(label(heat(im, dino_cls_attn(im)), "scene  DINOv3 CLS attn (frozen prior)"))
        else:
            col.append(np.zeros_like(im))
        panels.append(np.concatenate(col, axis=0))
    H = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, H-p.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(20,20,20)) for p in panels]
    outp = f"{a.out}/ep{e}_ph{int(ph*100):02d}.png"
    cv2.imwrite(outp, cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))
    print("  ->", outp)
