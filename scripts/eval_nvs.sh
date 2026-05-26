#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Evaluate rendered novel-view synthesis images.

Required:
  --pred-dir PATH
  --gt-dir PATH
  --output-dir PATH

Optional:
  --runtime-json PATH
  --render-dir PATH
  --gpu-log PATH
  --scene NAME
  --method NAME
EOF
}

PRED_DIR=""
GT_DIR=""
OUTPUT_DIR=""
RUNTIME_JSON=""
RENDER_DIR=""
GPU_LOG=""
SCENE=""
METHOD=""
REQUIRE_LPIPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred-dir) PRED_DIR="$2"; shift 2 ;;
    --gt-dir) GT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --runtime-json) RUNTIME_JSON="$2"; shift 2 ;;
    --render-dir) RENDER_DIR="$2"; shift 2 ;;
    --gpu-log) GPU_LOG="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --require-lpips) REQUIRE_LPIPS="--require-lpips"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PRED_DIR" || -z "$GT_DIR" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

python -m videogaus.eval.image_metrics --pred-dir "$PRED_DIR" --gt-dir "$GT_DIR" --output-dir "$OUTPUT_DIR" $REQUIRE_LPIPS

runtime_args=(--output "${OUTPUT_DIR}/runtime_metrics.json")
[[ -n "$RUNTIME_JSON" ]] && runtime_args+=(--training-metrics "$RUNTIME_JSON")
[[ -n "$RENDER_DIR" ]] && runtime_args+=(--render-dir "$RENDER_DIR")
[[ -n "$GPU_LOG" ]] && runtime_args+=(--gpu-log "$GPU_LOG")
[[ -n "$SCENE" ]] && runtime_args+=(--scene "$SCENE")
[[ -n "$METHOD" ]] && runtime_args+=(--method "$METHOD")
python -m videogaus.eval.runtime_metrics "${runtime_args[@]}"
