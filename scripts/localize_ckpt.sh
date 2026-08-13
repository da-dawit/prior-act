#!/usr/bin/env bash
#Rewrite a pulled checkpoint's DINOv3 path from the rented box's layout to this machine's.
#
#WHY. act_prior records `scene_backbone_path` in config.json, and on the box that is
#/workspace/pretrained/... . Locally that path does not exist, so from_pretrained hands the string
#to HuggingFace, which reads it as a repo id and dies with "Repo id must be in the form
#namespace/repo_name" -- an error that says nothing about the real cause. Rewriting it here means
#every pulled checkpoint is loadable without touching the training config on the box.
#The original string is kept beside it, so the edit is always reversible and never silent.
set -euo pipefail
LOCAL_DINO="${LOCAL_DINO:-/home/robotis/robot_aiworker/TurboVLA/pretrained/dinov3-vitl16-pretrain-lvd1689m}"
[ -d "$LOCAL_DINO" ] || { echo "no DINOv3 at $LOCAL_DINO"; exit 1; }
n=0
for cfg in "$@"; do
  grep -q '"scene_backbone_path"' "$cfg" || continue
  cur=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('scene_backbone_path') or '')" "$cfg")
  [ "$cur" = "$LOCAL_DINO" ] && continue
  [ -f "$cfg.orig_backbone_path" ] || printf '%s\n' "$cur" > "$cfg.orig_backbone_path"
  python3 - "$cfg" "$LOCAL_DINO" <<'PY'
import json,sys
p,new=sys.argv[1],sys.argv[2]
d=json.load(open(p)); d["scene_backbone_path"]=new
json.dump(d,open(p,"w"),indent=2)
PY
  echo "  localized $cfg"; n=$((n+1))
done
echo "[localize] rewrote $n config(s)"
