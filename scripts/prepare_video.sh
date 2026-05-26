#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Prepare video frames and train/test split.

Example:
  bash scripts/prepare_video.sh --video assets/scene.mp4 --frames-dir outputs/scene/frames --fps 6 --test-every 8
EOF
}

VIDEO=""
FRAMES_DIR=""
FPS=""
MAX_FRAMES=""
RESIZE=""
START_TIME=""
END_TIME=""
TEST_EVERY=""
TEST_RATIO=""
SEED="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --resize) RESIZE="$2"; shift 2 ;;
    --start-time) START_TIME="$2"; shift 2 ;;
    --end-time) END_TIME="$2"; shift 2 ;;
    --test-every) TEST_EVERY="$2"; shift 2 ;;
    --test-ratio) TEST_RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$VIDEO" || -z "$FRAMES_DIR" ]]; then
  usage >&2
  exit 2
fi

extract_args=(--video "$VIDEO" --output-dir "$FRAMES_DIR")
[[ -n "$FPS" ]] && extract_args+=(--fps "$FPS")
[[ -n "$MAX_FRAMES" ]] && extract_args+=(--max-frames "$MAX_FRAMES")
[[ -n "$RESIZE" ]] && extract_args+=(--resize "$RESIZE")
[[ -n "$START_TIME" ]] && extract_args+=(--start-time "$START_TIME")
[[ -n "$END_TIME" ]] && extract_args+=(--end-time "$END_TIME")
python -m videogaus.data.extract_frames "${extract_args[@]}"

split_args=(--frames-dir "$FRAMES_DIR" --manifest "${FRAMES_DIR}/frames.json" --seed "$SEED")
[[ -n "$TEST_EVERY" ]] && split_args+=(--test-every "$TEST_EVERY")
[[ -n "$TEST_RATIO" ]] && split_args+=(--test-ratio "$TEST_RATIO")
python -m videogaus.data.split_train_test "${split_args[@]}"
