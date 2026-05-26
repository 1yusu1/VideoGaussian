#!/usr/bin/env bash
set -euo pipefail

# Template for running gsplat on a remote Linux server.
# Run this from the gsplat examples directory, or set GSPLAT_EXAMPLES_DIR.

GSPLAT_EXAMPLES_DIR="${GSPLAT_EXAMPLES_DIR:-/path/to/gsplat-1.5.3/examples}"
DATA_DIR="${DATA_DIR:-/path/to/colmap_dataset}"
BASE_RESULT_DIR="${BASE_RESULT_DIR:-/path/to/output}"
EXP_NAME="${EXP_NAME:-videogaussian_full_4gpu}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-30000}"
EVAL_STEPS="${EVAL_STEPS:-30000}"
DATA_FACTOR="${DATA_FACTOR:-1}"
TEST_MODE="${TEST_MODE:-false}"

if [ "$TEST_MODE" = "true" ]; then
  MAX_STEPS=100
  EVAL_STEPS="100"
  EXP_NAME="${EXP_NAME}_test"
fi

export CUDA_VISIBLE_DEVICES
RESULT_DIR="${BASE_RESULT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULT_DIR}/training_log.txt"
mkdir -p "$RESULT_DIR"

cd "$GSPLAT_EXAMPLES_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used --format=csv -l 1 > "${RESULT_DIR}/vram_peak.log" &
  MONITOR_PID=$!
else
  MONITOR_PID=""
fi

echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "Data Dir: $DATA_DIR" | tee -a "$LOG_FILE"
echo "Result Dir: $RESULT_DIR" | tee -a "$LOG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Max Steps: $MAX_STEPS" | tee -a "$LOG_FILE"

python simple_trainer.py default \
  --data-dir "$DATA_DIR" \
  --result-dir "$RESULT_DIR" \
  --max-steps "$MAX_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --data-factor "$DATA_FACTOR" \
  --save-ply \
  --disable-video \
  2>&1 | tee -a "$LOG_FILE"

if [ -n "$MONITOR_PID" ]; then
  kill "$MONITOR_PID" || true
fi

echo "End Time: $(date)" | tee -a "$LOG_FILE"
if [ -f "${RESULT_DIR}/vram_peak.log" ]; then
  awk -F, 'NR>1 {if($1>max) max=$1} END {print "Peak VRAM: " max " MiB"}' \
    "${RESULT_DIR}/vram_peak.log" | tee -a "$LOG_FILE"
fi
