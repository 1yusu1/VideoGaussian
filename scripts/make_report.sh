#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Generate a Markdown report from VideoGaussian metrics.

Example:
  bash scripts/make_report.sh --scene garden --metrics-root outputs/garden/metrics --output-dir reports
  bash scripts/make_report.sh --scene liminal_pool --metrics-root /data1/panshihan/videogaussian_runs --output-dir reports

The metrics root may contain either eval_nvs image_metrics.json files or gsplat
result directories with stats/val_step*.json and runtime_metrics.json.
EOF
}

SCENE=""
METRICS_ROOT=""
OUTPUT_DIR="${PROJECT_DIR}/reports"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --metrics-root) METRICS_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$METRICS_ROOT" ]]; then
  usage >&2
  exit 2
fi

args=(--metrics-root "$METRICS_ROOT" --output-dir "$OUTPUT_DIR" --report)
[[ -n "$SCENE" ]] && args+=(--scene "$SCENE")
python -m videogaus.eval.summarize "${args[@]}"
