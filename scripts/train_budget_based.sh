#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ADAPTIVE_METHOD="curriculum-horizon"
OUTPUT_ROOT="experiments/lorenz/models/budget_based_dt_08_cv"
DT="0.08"
MIN_T=1
MAX_T=10
EPOCHS_PER_T=20
N_SEEDS=20
SYSTEM="lorenz"
FIXED_DIR=""
EARLY_STOPPING=true

usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --adaptive-method METHOD  adaptive-horizon, weighted-loss, or curriculum-horizon"
  echo "  --output-root DIR         Parent directory for budget_dt_*_T* runs"
  echo "  --dt VALUE                Simulation time step (default: $DT)"
  echo "  --min-T VALUE             First budget horizon (default: $MIN_T)"
  echo "  --max-T VALUE             Last budget horizon (default: $MAX_T)"
  echo "  --epochs-per-T VALUE      Epoch budget per horizon (default: $EPOCHS_PER_T)"
  echo "  --n-seeds VALUE           Total desired seed count (default: $N_SEEDS)"
  echo "  --system NAME             Dynamical system (default: $SYSTEM)"
  echo "  --fixed-dir DIR           Fixed models used for adaptive-horizon wall-time budgets"
  echo "  --early-stopping          Enable early stopping (default)"
  echo "  --no-early-stopping       Disable early stopping"
  echo "  -h, --help                Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adaptive-method) ADAPTIVE_METHOD="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dt) DT="$2"; shift 2 ;;
    --min-T) MIN_T="$2"; shift 2 ;;
    --max-T) MAX_T="$2"; shift 2 ;;
    --epochs-per-T) EPOCHS_PER_T="$2"; shift 2 ;;
    --n-seeds) N_SEEDS="$2"; shift 2 ;;
    --system) SYSTEM="$2"; shift 2 ;;
    --fixed-dir) FIXED_DIR="$2"; shift 2 ;;
    --early-stopping) EARLY_STOPPING=true; shift ;;
    --no-early-stopping) EARLY_STOPPING=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "$PROJECT_DIR"

for ((T = MIN_T; T <= MAX_T; T++)); do
  args=(
    --budget-based
    --adaptive
    --adaptive-method "$ADAPTIVE_METHOD"
    --dt "$DT"
    --max-T "$T"
    --epochs-per-T "$EPOCHS_PER_T"
    --n-seeds "$N_SEEDS"
    --system "$SYSTEM"
    --output-dir "$OUTPUT_ROOT/budget_dt_${DT#*.}_T${T}"
  )

  if [[ "$EARLY_STOPPING" == true ]]; then
    args+=(--early-stopping)
  fi
  if [[ -n "$FIXED_DIR" ]]; then
    args+=(--fixed-dir "$FIXED_DIR")
  fi

  .venv/bin/python -m adaptive_horizon.training.train "${args[@]}"
done
