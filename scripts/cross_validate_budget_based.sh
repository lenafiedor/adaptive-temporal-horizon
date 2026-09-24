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
CACHED=""

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
  echo "  --cached PATH         Cached result file or directory"
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
    --cached) CACHED="$2"; shift 2 ;;
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

if [[ -z "$FIXED_DIR" ]]; then
  sibling="${MODEL_DIR}_fixed"
  if [[ -d "$sibling/fixed" ]]; then
    FIXED_DIR="$sibling/fixed"
  elif [[ -d "$sibling" ]]; then
    FIXED_DIR="$sibling"
  else
    model_name="${MODEL_DIR##*/}"
    if [[ "$model_name" =~ ^(.+dt_[0-9]+)(_.+)?$ ]]; then
      sibling="${MODEL_DIR%/*}/${BASH_REMATCH[1]}_fixed"
      if [[ -d "$sibling/fixed" ]]; then
        FIXED_DIR="$sibling/fixed"
      elif [[ -d "$sibling" ]]; then
        FIXED_DIR="$sibling"
      fi
    fi
  fi
fi

if [[ -z "$FIXED_DIR" || ! -d "$FIXED_DIR" ]]; then
  echo "Could not infer fixed model directory; pass --fixed-dir explicitly." >&2
  exit 1
fi

runs=()
for run_dir in "$MODEL_DIR"/*; do
  [[ -d "$run_dir/adaptive" ]] || continue
  run_name="${run_dir##*/}"
  if [[ "$run_name" =~ _T([0-9]+)$ ]]; then
    runs+=("${BASH_REMATCH[1]}"$'\t'"$run_dir")
  fi
done

if [[ ${#runs[@]} -eq 0 ]]; then
  echo "No budget run directories found in $MODEL_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/adaptive_horizon_mplconfig_budget_cv}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

while IFS=$'\t' read -r max_train_T run_dir; do
  args=(
    --model-dir "$run_dir"
    --fixed-dir "$FIXED_DIR"
    --output-dir "$OUTPUT_DIR"
    --max-train-T "$max_train_T"
    --max-eval-T "$MAX_EVAL_T"
    --metric "$METRIC"
    --system "$SYSTEM"
  )

  cached_file=""
  if [[ -f "$CACHED" ]]; then
    if [[ "${CACHED##*/}" == *_T"$max_train_T"_*.json ]]; then
      cached_file="$CACHED"
    fi
  else
    cache_dir="$CACHED"
    [[ -n "$cache_dir" ]] || cache_dir="$OUTPUT_DIR"
    for candidate in "$cache_dir"/budget_mse_results_*_T"$max_train_T"_*.json; do
      if [[ -f "$candidate" ]]; then
        cached_file="$candidate"
      fi
    done
  fi
  if [[ -n "$cached_file" ]]; then
    args+=(--cached "$cached_file")
  fi

  .venv/bin/python -u -m adaptive_horizon.evaluation.cross_validation "${args[@]}"
done < <(printf '%s\n' "${runs[@]}" | sort -n -k1,1)
