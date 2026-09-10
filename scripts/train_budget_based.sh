#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

METHOD=""
OUTPUT_DIR=""
DT="0.08"
MIN_T=1
MAX_T=10
EPOCHS_PER_T=20
N_SEEDS=20
SYSTEM="lorenz"
FIXED_DIR=""

usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --method METHOD           fixed, lyapunov-based, weighted-loss, linear-scheduler, early-stopping, or cross-validation"
  echo "  --output-dir DIR          Parent directory for budget_dt_*_T* runs"
  echo "  --dt VALUE                Simulation time step (default: $DT)"
  echo "  --min-T VALUE             First budget horizon (default: $MIN_T)"
  echo "  --max-T VALUE             Last budget horizon (default: $MAX_T)"
  echo "  --epochs-per-T VALUE      Epoch budget per horizon (default: $EPOCHS_PER_T)"
  echo "  --n-seeds VALUE           Total desired seed count (default: $N_SEEDS)"
  echo "  --system NAME             Dynamical system (default: $SYSTEM)"
  echo "  --fixed-dir DIR           Fixed models used for lyapunov-based wall-time budgets"
  echo "  -h, --help                Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dt) DT="$2"; shift 2 ;;
    --min-T) MIN_T="$2"; shift 2 ;;
    --max-T) MAX_T="$2"; shift 2 ;;
    --epochs-per-T) EPOCHS_PER_T="$2"; shift 2 ;;
    --n-seeds) N_SEEDS="$2"; shift 2 ;;
    --system) SYSTEM="$2"; shift 2 ;;
    --fixed-dir) FIXED_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$METHOD" ]]; then
  echo "--method is required" >&2
  usage >&2
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "--output-dir is required" >&2
  usage >&2
  exit 1
fi

case "$METHOD" in
  fixed)
    method_args=(--fixed)
    ;;
  early-stopping)
    method_args=(--method early-stopping)
    ;;
  cross-validation)
    method_args=(--method cross-validation)
    ;;
  linear-scheduler)
    method_args=(--method linear-scheduler)
    ;;
  weighted-loss)
    method_args=(--method weighted-loss)
    ;;
  lyapunov-based)
    if [[ -z "$FIXED_DIR" ]]; then
      echo "--fixed-dir is required for lyapunov-based" >&2
      exit 1
    fi
    method_args=(--method lyapunov-based)
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    usage >&2
    exit 1
    ;;
esac

cd "$PROJECT_DIR"

for ((T = MIN_T; T <= MAX_T; T++)); do
  if [[ "$METHOD" == "fixed" ]]; then
    run_output_dir="$OUTPUT_DIR"
  else
    run_output_dir="$OUTPUT_DIR/budget_dt_${DT#*.}_T${T}"
  fi

  args=(
    --budget-based
    --dt "$DT"
    --max-T "$T"
    --epochs-per-T "$EPOCHS_PER_T"
    --n-seeds "$N_SEEDS"
    --system "$SYSTEM"
    --output-dir "$run_output_dir"
  )
  args+=("${method_args[@]}")

  if [[ -n "$FIXED_DIR" ]]; then
    args+=(--fixed-dir "$FIXED_DIR")
  fi

  .venv/bin/python -m adaptive_horizon.training.train "${args[@]}"
done
