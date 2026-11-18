#!/usr/bin/env bash
set -euo pipefail

# Held-out-split validated concurrent WM/policy candidate.
SEED="${SEED:-0}"
LOG_DIR="${LOG_DIR:-logs/concurrent_ramp_s${SEED}}"
DEVICE="${DEVICE:-cuda:0}"
POLICY_UPDATES="${POLICY_UPDATES:-2}"
POLICY_UPDATES_WARMUP_ITERS="${POLICY_UPDATES_WARMUP_ITERS:-0}"
POLICY_SECOND_UPDATE_SCALE="${POLICY_SECOND_UPDATE_SCALE:-1.0}"
POLICY_REUSE_BATCH="${POLICY_REUSE_BATCH:-0}"
INTERLEAVE_POLICY_WM="${INTERLEAVE_POLICY_WM:-0}"
POLICY_ACCUMULATE_UPDATES="${POLICY_ACCUMULATE_UPDATES:-0}"

POLICY_REUSE_ARGS=()
if [[ "${POLICY_REUSE_BATCH}" == "1" ]]; then
  POLICY_REUSE_ARGS+=(--policy_reuse_batch)
fi
if [[ "${INTERLEAVE_POLICY_WM}" == "1" ]]; then
  POLICY_REUSE_ARGS+=(--interleave_policy_wm)
fi
if [[ "${POLICY_ACCUMULATE_UPDATES}" == "1" ]]; then
  POLICY_REUSE_ARGS+=(--policy_accumulate_updates)
fi

python supermimic/train.py \
  --num_envs 64 \
  --iterations "${ITERATIONS:-2000}" \
  --buffer_steps 64 \
  --utd 0.4 \
  --horizon 64 \
  --policy_bptt_truncation 4 \
  --goal_frames 10 \
  --hist 10 \
  --wm_window 4 \
  --wm_aux_window 8 \
  --wm_aux_weight 0.05 \
  --wm_aux_ramp_iters 300 \
  --wm_batches 2 \
  --wm_batch 32 \
  --pi_batch 32 \
  --policy_updates "${POLICY_UPDATES}" \
  --policy_updates_warmup_iters "${POLICY_UPDATES_WARMUP_ITERS}" \
  --policy_second_update_scale "${POLICY_SECOND_UPDATE_SCALE}" \
  "${POLICY_REUSE_ARGS[@]}" \
  --no_critic \
  --warmup_steps 8 \
  --wm_only_iters 5 \
  --ckpt_every 300 \
  --log_every 100 \
  --wm_norm_checkpoint logs/shared_norm_avg.pt \
  --wm_teacher_forcing \
  --log_dir "${LOG_DIR}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --amp bf16
