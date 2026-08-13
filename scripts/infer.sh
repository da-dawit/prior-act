#!/usr/bin/env bash
# Run Prior-ACT on the robot through aiworker_deploy/infer.py.
#
# Author: Dawit Chun
#
# Every value below was measured on held-out demonstration frames through the real controller.
# See docs/INFERENCE.md for the numbers behind each one. Override any of them by environment
# variable; nothing here is specific to a particular task or object.
#
#   bash infer.sh <ckpt>/pretrained_model
#   MAX_VEL=0.6 bash infer.sh <ckpt>/pretrained_model      # slower, if the arm swings
set -euo pipefail
CKPT="${1:?usage: infer.sh <checkpoint>/pretrained_model}"

INFER_PY="${INFER_PY:-/home/robotis/robot_aiworker/aiworker_deploy/infer.py}"
ROUTER_IP="${ROUTER_IP:-10.42.0.22}"

# smoothing. ENSEMBLE above 4 is a no-op at EXECUTE_STEPS 25: a chunk is 100 waypoints and the
# averaging loop breaks at offset 100, so only 4 plans ever overlap.
EXECUTE_STEPS="${EXECUTE_STEPS:-25}"
ENSEMBLE="${ENSEMBLE:-4}"
ENSEMBLE_DECAY="${ENSEMBLE_DECAY:-0.3}"
SAVGOL="${SAVGOL:-17}"            # -10.1% jerk vs 11, no tracking cost. Lower it if turns feel blunt.
SAVGOL_ORDER="${SAVGOL_ORDER:-3}"
SEAM_BLEND="${SEAM_BLEND:-12}"    # -5.5% jerk, +1.5% tracking error
TRIM_HEAD="${TRIM_HEAD:-0}"

# speed. MAX_VEL and MAX_ACC must move together: raising the speed limit without raising the brake
# lets the arm reach a velocity it cannot shed, which shows up as overshoot and swing.
MAX_VEL="${MAX_VEL:-0.8}"         # 0.6 clips the leading joint on 6.6% of demonstration steps
MAX_ACC="${MAX_ACC:-6.0}"         # demonstrations reach p50 2.76, p90 6.90 rad/s^2
SPEED="${SPEED:-1.0}"             # time scale only; does not affect the approach rate

# gripper. GRIP_HOLD is consecutive ticks the plan must agree to OPEN before it opens. The default
# of 10 stacks on a 25-waypoint agreement window and delays release by roughly a second.
GRIP_LEAD="${GRIP_LEAD:-14}"
GRIP_HOLD="${GRIP_HOLD:-4}"       # raise toward 10 if the object is dropped mid-carry

MAX_STEPS="${MAX_STEPS:-1200}"
LIVE="${LIVE:-1}"                 # LIVE=0 for a dry run: infers and prints, publishes nothing

ARGS=(--policy act_prior --policy-path "$CKPT" --residual ""
      --router-ip "$ROUTER_IP"
      --trim-head "$TRIM_HEAD" --seam-blend "$SEAM_BLEND" --execute-steps "$EXECUTE_STEPS"
      --savgol "$SAVGOL" --savgol-order "$SAVGOL_ORDER"
      --ensemble "$ENSEMBLE" --ensemble-decay "$ENSEMBLE_DECAY"
      --grip-lead "$GRIP_LEAD" --grip-hold "$GRIP_HOLD"
      --max-vel "$MAX_VEL" --max-acc "$MAX_ACC"
      --speed "$SPEED" --max-steps "$MAX_STEPS" --home-first)
[ "$LIVE" = "1" ] && ARGS+=(--live)

echo "checkpoint $CKPT"
echo "vel<=$MAX_VEL acc<=$MAX_ACC  savgol $SAVGOL  seam $SEAM_BLEND  ensemble $ENSEMBLE  grip-hold $GRIP_HOLD"
[ "$LIVE" = "1" ] || echo "DRY RUN: nothing will be published"
exec python3 "$INFER_PY" "${ARGS[@]}"
