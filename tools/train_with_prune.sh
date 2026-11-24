#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Run a training command and delete its old checkpoints while it trains.
#
#   tools/train_with_prune.sh --checkpoint-dir DIR [options] [--] TRAIN_COMMAND [ARGS...]
#
# Starts TRAIN_COMMAND. On the first node of the job (SLURM_NODEID unset or 0) it also runs
# tools/prune_checkpoints.py on DIR every --interval seconds while the command is alive, and once more
# after it exits, so the last save is pruned too. Other nodes only run the command. The script exits
# with the status of TRAIN_COMMAND; a failing or refusing pruner never stops training. TERM and INT are
# passed on to TRAIN_COMMAND as TERM.
#
# Options (default in brackets):
#   --checkpoint-dir DIR  the run's checkpoints/ directory, or the run directory that holds it
#   --keep-last N         keep the newest N saves [2]
#   --keep-every K        also keep every K-th iteration; 0 keeps none [0]
#   --milestones LIST     also keep these iterations, comma-separated [500,1000,2000]
#   --save-iter S         the run's checkpoint interval; a --keep-every that divides it would keep every
#                         save, so it is refused [100]
#   --interval SECONDS    time between two pruning passes [600]
# The pruner runs with $PRUNE_PYTHON [python3].

set -u

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
checkpoint_dir=""
keep_last=2
keep_every=0
milestones=500,1000,2000
save_iter=100
interval=600

die() {
  echo "train_with_prune: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --checkpoint-dir | --keep-last | --keep-every | --milestones | --save-iter | --interval)
      (($# >= 2)) || die "$1 needs a value"
      case "$1" in
        --checkpoint-dir) checkpoint_dir=$2 ;;
        --keep-last) keep_last=$2 ;;
        --keep-every) keep_every=$2 ;;
        --milestones) milestones=$2 ;;
        --save-iter) save_iter=$2 ;;
        --interval) interval=$2 ;;
      esac
      shift 2
      ;;
    --)
      shift
      break
      ;;
    --*) die "unknown option $1" ;;
    *) break ;;
  esac
done

[[ -n "$checkpoint_dir" ]] || die "--checkpoint-dir is required"
(($#)) || die "no training command given"
[[ "$keep_last" =~ ^[0-9]+$ ]] && ((keep_last >= 1)) || die "--keep-last must be 1 or more, got '$keep_last'"
[[ "$keep_every" =~ ^[0-9]+$ ]] || die "--keep-every must be 0 or more, got '$keep_every'"
[[ "$save_iter" =~ ^[0-9]+$ ]] && ((save_iter >= 1)) || die "--save-iter must be 1 or more, got '$save_iter'"
[[ "$interval" =~ ^[0-9]+$ ]] && ((interval >= 1)) || die "--interval must be 1 or more, got '$interval'"
[[ "$milestones" =~ ^([0-9]+(,[0-9]+)*)?$ ]] || die "--milestones must be comma-separated iterations, got '$milestones'"
if ((keep_every > 0 && save_iter % keep_every == 0)); then
  die "--keep-every $keep_every would keep every save of a run that saves every $save_iter iterations"
fi

prune() {
  "${PRUNE_PYTHON:-python3}" "$here/prune_checkpoints.py" --checkpoint-dir "$checkpoint_dir" \
    --keep-last "$keep_last" --keep-every "$keep_every" --milestones "$milestones" ||
    echo "train_with_prune: the pruner exited with status $?; training goes on" >&2
}

"$@" &
train_pid=$!
trap 'kill -s TERM "$train_pid" 2>/dev/null' TERM INT

loop_pid=""
if [[ "${SLURM_NODEID:-0}" == 0 ]]; then
  echo "train_with_prune: pruning $checkpoint_dir every $interval s:" \
    "keep-last $keep_last, keep-every $keep_every, milestones ${milestones:-none}, save-iter $save_iter"
  (
    last=$SECONDS
    while kill -0 "$train_pid" 2>/dev/null; do
      sleep 1
      if ((SECONDS - last >= interval)) && kill -0 "$train_pid" 2>/dev/null; then
        echo "train_with_prune: pruning pass while training runs"
        prune
        last=$SECONDS
      fi
    done
  ) &
  loop_pid=$!
fi

# A trapped signal interrupts wait with a status above 128; keep waiting while the command lives.
while true; do
  wait "$train_pid"
  status=$?
  kill -0 "$train_pid" 2>/dev/null || break
done

if [[ -n "$loop_pid" ]]; then
  wait "$loop_pid" 2>/dev/null
  echo "train_with_prune: final pruning pass, training exited with status $status"
  prune
fi
exit "$status"
