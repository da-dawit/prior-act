#!/usr/bin/env bash
# Train Prior-ACT.
#
# Author: Dawit Chun
#
# Everything task-specific is an environment variable with a default. Nothing below assumes a
# particular task, object or number of cameras. To train on a different dataset, set DATASET_ROOT
# and REPO_ID; the policy reads the camera keys, state dimension and action dimension from the
# dataset's meta/info.json.
#
#   DATASET_ROOT=/data/my_task REPO_ID=me/my_task bash train_prior.sh
#
# The three arguments that make the scene prior work are PRIOR_FIT_WEIGHT, SCENE_PRIOR_FEATURES and
# LATENT_INJECTION. See docs/TUNING.md before changing them.

set -euo pipefail

# ── dataset ───────────────────────────────────────────────────────────────────────────────────────
REPO_ID="${REPO_ID:?set REPO_ID, e.g. myuser/my_task}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the LeRobot dataset directory}"
EVAL_SPLIT="${EVAL_SPLIT:-0.111}"      # fraction of EPISODES held out, never frames
EVAL_STEPS="${EVAL_STEPS:-500}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"

# ── scene prior ───────────────────────────────────────────────────────────────────────────────────
DINO_PATH="${DINO_PATH:?set DINO_PATH to a local DINOv3 checkpoint directory}"
SCENE_BACKBONE="${SCENE_BACKBONE:-dinov3}"
SCENE_PRIOR_FEATURES="${SCENE_PRIOR_FEATURES:-cls_grid}"
PRIOR_FIT_WEIGHT="${PRIOR_FIT_WEIGHT:-10.0}"
KL_WEIGHT="${KL_WEIGHT:-10.0}"
KL_FREE_BITS="${KL_FREE_BITS:-0.05}"
LATENT_INJECTION="${LATENT_INJECTION:-decoder_seed}"
LATENT_DIAG_EVERY="${LATENT_DIAG_EVERY:-250}"

# Optional: which cameras the prior reads. Unset means camera 0 only. Give a comma-separated list of
# dataset image keys for a multi-camera prior, e.g.
#   SCENE_PRIOR_CAMERAS=observation.images.scene,observation.images.overhead
SCENE_PRIOR_CAMERAS="${SCENE_PRIOR_CAMERAS:-}"

# ── policy ────────────────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE="${CHUNK_SIZE:-100}"
LR="${LR:-1e-5}"
STATE_DROPOUT="${STATE_DROPOUT:-0.5}"
CAMERA_ID_EMBED="${CAMERA_ID_EMBED:-true}"
IMAGE_TRANSFORMS="${IMAGE_TRANSFORMS:-true}"

# ── schedule ──────────────────────────────────────────────────────────────────────────────────────
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-192}"        # PER PROCESS. effective = BATCH_SIZE x num_processes
NUM_WORKERS="${NUM_WORKERS:-32}"
SEED="${SEED:-1000}"
SAVE_FREQ="${SAVE_FREQ:-500}"          # the optimum often falls between checkpoints; keep this small
LOG_FREQ="${LOG_FREQ:-200}"
OUT_DIR="${OUT_DIR:-out/prior_act}"
JOB_NAME="${JOB_NAME:-prior_act}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-prior_act}"

[ -d "$DATASET_ROOT" ] || { echo "DATASET_ROOT does not exist: $DATASET_ROOT"; exit 1; }
[ -d "$DINO_PATH" ]    || { echo "DINO_PATH does not exist: $DINO_PATH"; exit 1; }

ARGS=(
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$EVAL_SPLIT"
  --eval_steps="$EVAL_STEPS"
  --max_eval_samples="$MAX_EVAL_SAMPLES"
  --dataset.image_transforms.enable="$IMAGE_TRANSFORMS"
  --policy.type=act_prior
  --policy.push_to_hub=false            # lerobot refuses to start otherwise, asking for a hub id
  --policy.use_amp=true                 # NOT the default; fp32 at batch 192 OOMs on 24 GB
  --policy.chunk_size="$CHUNK_SIZE"
  --policy.n_action_steps="$CHUNK_SIZE"
  --policy.optimizer_lr="$LR"
  --policy.optimizer_lr_backbone="$LR"
  --policy.camera_identity_embedding="$CAMERA_ID_EMBED"
  --policy.state_dropout="$STATE_DROPOUT"
  --policy.scene_backbone="$SCENE_BACKBONE"
  --policy.scene_backbone_path="$DINO_PATH"
  --policy.scene_prior_features="$SCENE_PRIOR_FEATURES"
  --policy.kl_weight="$KL_WEIGHT"
  --policy.kl_free_bits="$KL_FREE_BITS"
  --policy.prior_fit_weight="$PRIOR_FIT_WEIGHT"
  --policy.latent_injection="$LATENT_INJECTION"
  --policy.deterministic_latent=true
  --policy.latent_diag_every="$LATENT_DIAG_EVERY"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --seed="$SEED"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --wandb.enable="$WANDB"
  --wandb.project="$WANDB_PROJECT"
  --output_dir="$OUT_DIR"
  --job_name="$JOB_NAME"
)
[ -n "$SCENE_PRIOR_CAMERAS" ] && ARGS+=(--policy.scene_prior_cameras="$SCENE_PRIOR_CAMERAS")

echo "prior_fit_weight=$PRIOR_FIT_WEIGHT  features=$SCENE_PRIOR_FEATURES  injection=$LATENT_INJECTION"
echo "free_bits=$KL_FREE_BITS  kl_weight=$KL_WEIGHT  -> effective batch $BATCH_SIZE x N_PROC"
echo

exec python -m lerobot.scripts.lerobot_train "${ARGS[@]}"
