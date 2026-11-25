#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

METHOD=""
OUTPUT_DIR=""
DT=""
DATA_PATH=""
TOLERANCE=""
GROWTH=""
MIN_T=1
MAX_T=10
EPOCHS_PER_T=20
N_SEEDS=20
SYSTEM="lorenz"
RANDOM_HORIZONS=0
SPREAD=""

usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --method METHOD           fixed, lyapunov-mean, weighted-loss, linear-scheduler, cross-validation, or error-truncated (tolerance from the FSLE knee)"
  echo "  --output-dir DIR          Parent directory for budget_dt_*_T* runs"
  echo "  --dt VALUE                Simulation time step (default: 0.08; santafe-laser: 1)"
  echo "  --data-path FILE          Santa Fe intensity recording (required for --system santafe-laser)"
  echo "  --tolerance VALUE         error-truncated: fixed RMSE tolerance instead of the FSLE knee"
  echo "  --growth VALUE            error-truncated: scale-free growth factor instead of the FSLE knee"
  echo "  --min-T VALUE             First budget horizon (default: $MIN_T)"
  echo "  --max-T VALUE             Last budget horizon (default: $MAX_T)"
  echo "  --epochs-per-T VALUE      Epoch budget per horizon (default: $EPOCHS_PER_T)"
  echo "  --n-seeds VALUE           Total desired seed count (default: $N_SEEDS)"
  echo "  --system NAME             Dynamical system (default: $SYSTEM)"
  echo "  --random                  lyapunov-mean only: random horizons (FTLE ablation)"
  echo "  --spread VALUE            lyapunov-mean only: T ranges over base_T +- VALUE (default: config.toml)"
  echo "  -h, --help                Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dt) DT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --tolerance) TOLERANCE="$2"; shift 2 ;;
    --growth) GROWTH="$2"; shift 2 ;;
    --min-T) MIN_T="$2"; shift 2 ;;
    --max-T) MAX_T="$2"; shift 2 ;;
    --epochs-per-T) EPOCHS_PER_T="$2"; shift 2 ;;
    --n-seeds) N_SEEDS="$2"; shift 2 ;;
    --system) SYSTEM="$2"; shift 2 ;;
    --random) RANDOM_HORIZONS=1; shift ;;
    --spread) SPREAD="$2"; shift 2 ;;
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

if [[ "$SYSTEM" == "santafe-laser" ]]; then
  DT="${DT:-1}"
  if [[ -z "$DATA_PATH" ]]; then
    echo "--system santafe-laser requires --data-path" >&2
    exit 1
  fi
else
  DT="${DT:-0.08}"
  if [[ -n "$DATA_PATH" ]]; then
    echo "--data-path is only supported for --system santafe-laser" >&2
    exit 1
  fi
fi
if [[ -n "$TOLERANCE" && -n "$GROWTH" ]]; then
  echo "--tolerance and --growth are mutually exclusive" >&2
  exit 1
fi
if [[ ( -n "$TOLERANCE" || -n "$GROWTH" ) && "$METHOD" != "error-truncated" ]]; then
  echo "--tolerance/--growth require --method error-truncated" >&2
  exit 1
fi

case "$METHOD" in
  fixed)
    method_args=(--fixed)
    ;;
  cross-validation)
    method_args=(--method cross-validation)
    ;;
  linear-scheduler)
    method_args=(--method linear-scheduler)
    ;;
  error-truncated)
    if [[ -n "$TOLERANCE" ]]; then
      method_args=(--method error-truncated --tolerance "$TOLERANCE")
    elif [[ -n "$GROWTH" ]]; then
      method_args=(--method error-truncated --growth "$GROWTH")
    else
      method_args=(--method error-truncated --tolerance-from-fsle)
    fi
    ;;
  lyapunov-mean|weighted-loss)
    method_args=(--method "$METHOD")
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "$RANDOM_HORIZONS" == 1 ]]; then
  if [[ "$METHOD" != "lyapunov-mean" ]]; then
    echo "--random requires --method lyapunov-mean" >&2
    exit 1
  fi
  method_args+=(--random)
fi

if [[ -n "$SPREAD" ]]; then
  if [[ "$METHOD" != "lyapunov-mean" ]]; then
    echo "--spread requires --method lyapunov-mean" >&2
    exit 1
  fi
  method_args+=(--spread "$SPREAD")
fi

if [[ -n "$DATA_PATH" ]]; then
  method_args+=(--data-path "$DATA_PATH")
fi

if [[ "$DT" == 0.* ]]; then
  DT_TAG="${DT#0.}"
elif [[ "$DT" == *.* ]]; then
  DT_TAG="${DT/./_}"
else
  DT_TAG="${DT}_0"
fi

cd "$PROJECT_DIR"

for ((T = MIN_T; T <= MAX_T; T++)); do
  if [[ "$METHOD" == "fixed" ]]; then
    run_output_dir="$OUTPUT_DIR"
  else
    run_output_dir="$OUTPUT_DIR/budget_dt_${DT_TAG}_T${T}"
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

  .venv/bin/python -m adaptive_horizon.training.train "${args[@]}"
done
