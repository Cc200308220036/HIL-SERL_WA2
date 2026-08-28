#!/usr/bin/env bash
# Generic WA2 actor launcher. Usage:
#   bash run_actor.sh bottle_pick --dry-run
#   WA2_TASK=bottle_pick bash run_actor.sh --dry-run
# Formal: bash run_actor.sh bottle_pick --checkpoint_path=... --debug
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES="$(cd "$ROOT/../.." && pwd)"
CATKIN_SRC="$(cd "$ROOT/../../../.." && pwd)"

DRY_RUN=0
TASK=""
FORWARD=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  elif [[ "$arg" != -* && -z "$TASK" ]]; then
    TASK="$arg"
  else
    FORWARD+=("$arg")
  fi
done

if [[ -z "$TASK" ]]; then
  TASK="${WA2_TASK:-}"
fi
if [[ -z "$TASK" ]]; then
  echo "MISSING_TASK: pass <task_id> or set WA2_TASK" >&2
  exit 1
fi
if [[ ! "$TASK" =~ ^[a-z0-9_]+$ ]]; then
  echo "invalid task_id: $TASK" >&2
  exit 1
fi
if [[ -n "${WA2_TASK:-}" && "$WA2_TASK" != "$TASK" ]]; then
  echo "TASK_MISMATCH: WA2_TASK=${WA2_TASK} vs arg=${TASK}" >&2
  exit 1
fi

EXP_NAME="wa2_${TASK}"
export PYTHONPATH="${CATKIN_SRC}:${EXAMPLES}:${CATKIN_SRC}/hil-serl-main/serl_launcher:${PYTHONPATH:-}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.1}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  python - "$TASK" <<'PY'
import os, sys
from hilserl_wa2.experiments.task_config import load_task
task = load_task(sys.argv[1])
print(f"exp_name={task.exp_name}")
print(f"task_id={task.task_id}")
for key, path in task.resolved_paths().items():
    print(f"{key}_path={path}")
print(f"config_bundle_hash={task.config_bundle_hash()}")
print("role=actor")
PY
  exit 0
fi

cd "$EXAMPLES"
exec python train_rlpd.py --exp_name="$EXP_NAME" --actor "${FORWARD[@]}"
