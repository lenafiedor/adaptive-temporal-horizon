#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR=""
OUTPUT_DIR=""
FIXED_DIR=""
MAX_EVAL_T="10"
METRIC="median"
SYSTEM="lorenz"
CACHE_DIRS=()

usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --model-dir DIR       Root containing budget_dt_*_T*/adaptive directories"
  echo "  --output-dir DIR      Directory for CV JSON files and plots"
  echo "  --fixed-dir DIR       Fixed-model directory (default: inferred)"
  echo "  --max-eval-T VALUE    Maximum validation horizon (default: $MAX_EVAL_T)"
  echo "  --metric VALUE        mean or median (default: $METRIC)"
  echo "  --system NAME         lorenz, lorenz96, or rossler (default: $SYSTEM)"
  echo "  --cache-dir DIR       Cache directory; may be repeated"
  echo "  -h, --help            Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --fixed-dir) FIXED_DIR="$2"; shift 2 ;;
    --max-eval-T) MAX_EVAL_T="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    --system) SYSTEM="$2"; shift 2 ;;
    --cache-dir) CACHE_DIRS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "--model-dir is required" >&2
  usage >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${MODEL_DIR/\/models\//\/evaluation\/}"
fi

cd "$PROJECT_DIR"

args=(
  "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-eval-T "$MAX_EVAL_T"
  --metric "$METRIC"
  --system "$SYSTEM"
)

if [[ -n "$FIXED_DIR" ]]; then
  args+=(--fixed-dir "$FIXED_DIR")
fi
if [[ ${#CACHE_DIRS[@]} -gt 0 ]]; then
  for cache_dir in "${CACHE_DIRS[@]}"; do
    args+=(--cache-dir "$cache_dir")
  done
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/adaptive_horizon_mplconfig_budget_cv}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

exec .venv/bin/python -u -m adaptive_horizon.evaluation.cross_validation_catalog "${args[@]}"
