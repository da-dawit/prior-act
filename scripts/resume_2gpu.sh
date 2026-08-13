#!/usr/bin/env bash
# Resume a run across multiple GPUs.
#
# Author: Dawit Chun
#
# batch_size is PER PROCESS: lerobot logs "Effective batch size: batch_size x num_processes".
# Halve it when doubling the process count, or the effective batch doubles, the effective learning
# rate changes, and results after the resume are not comparable with results before it.
set -euo pipefail
CFG="${1:?usage: resume_2gpu.sh <ckpt>/pretrained_model/train_config.json}"
NPROC="${NPROC:-2}"
BATCH="${BATCH:-96}"          # x NPROC = the original per-run batch
WORKERS="${WORKERS:-24}"      # per process; 32x2 contended on a 96-core machine
SAVE_FREQ="${SAVE_FREQ:-500}"

[ -f "$CFG" ] || { echo "no config at $CFG"; exit 1; }
echo "resuming $CFG on $NPROC processes, effective batch $((BATCH * NPROC))"
exec accelerate launch --num_processes="$NPROC" --num_machines=1 --mixed_precision=no \
  -m lerobot.scripts.lerobot_train \
  --config_path="$CFG" --resume=true \
  --batch_size="$BATCH" --num_workers="$WORKERS" --save_freq="$SAVE_FREQ"
