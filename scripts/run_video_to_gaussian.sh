#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the VideoGaussian pipeline from one video to one merged gsplat PLY.

Required:
  --video PATH                 Input video path.
  --depth-anything-repo PATH   Depth-Anything-3 repository on the server.
  --gsplat-examples-dir PATH   gsplat-1.5.3/examples directory on the server.

Common options:
  --config PATH                Load paths and parameters from a YAML config.
  --scene NAME                 Scene/run name. Defaults to video filename stem.
  --work-dir PATH              Root for all generated outputs. Defaults to ./runs.
  --model-dir PATH_OR_HF_ID    DA3 model path or Hugging Face id.
  --fps N                      DA3 video sampling fps. Defaults to 12.
  --export-format FORMAT       DA3 export format. Defaults to mini_npz-glb-depth_vis-colmap.
  --process-res N              DA3 processing resolution. Defaults to 504.
  --process-res-method METHOD  DA3 resize/crop method. Defaults to upper_bound_resize.
  --ref-view-strategy NAME     DA3 reference view strategy. Defaults to saddle_balanced.
  --conf-thresh-percentile N   DA3 confidence percentile for GLB/COLMAP exports. Defaults to 40.
  --num-max-points N           DA3 GLB max points. Defaults to 1000000.
  --use-ray-pose               Use DA3 ray-based pose estimation.
  --gpus LIST                  CUDA_VISIBLE_DEVICES value. Defaults to existing env or 0.
  --max-steps N                gsplat max steps. Defaults to 30000.
  --eval-steps N               gsplat eval steps. Defaults to max steps.
  --data-factor N              gsplat data factor. Defaults to 1.
  --save-ply-during-train      Let gsplat export PLY during training. Can OOM on large scenes.
  --enable-viewer              Keep gsplat's training viewer running after training.
  --conda-env NAME             Optional conda env to activate before running.
  --test-mode                  Use 100 gsplat steps for a smoke test.
  --skip-da3                   Reuse an existing --da3-output-dir.
  --skip-train                 Reuse an existing --result-dir and only merge.
  --no-merge                   Skip checkpoint merging.

Advanced paths:
  --da3-output-dir PATH        Override DA3 output directory.
  --dataset-dir PATH           Override prepared gsplat dataset directory.
  --result-dir PATH            Override gsplat result directory.
  --final-ply PATH             Override merged PLY output path.

Example:
  bash scripts/run_video_to_gaussian.sh \
    --config configs/pipeline.example.yaml \
    --test-mode
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE=""
args=("$@")
idx=0
while [[ $idx -lt ${#args[@]} ]]; do
  case "${args[$idx]}" in
    --config)
      next_idx=$((idx + 1))
      if [[ $next_idx -ge ${#args[@]} ]]; then
        echo "Missing value for --config" >&2
        exit 2
      fi
      CONFIG_FILE="${args[$next_idx]}"
      idx=$((idx + 2))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

VIDEO=""
DEPTH_ANYTHING_REPO=""
GSPLAT_EXAMPLES_DIR=""
SCENE_NAME=""
WORK_DIR="${WORK_DIR:-./runs}"
MODEL_DIR="${MODEL_DIR:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"
FPS="${FPS:-12}"
EXPORT_FORMAT="${EXPORT_FORMAT:-mini_npz-glb-depth_vis-colmap}"
PROCESS_RES="${PROCESS_RES:-504}"
PROCESS_RES_METHOD="${PROCESS_RES_METHOD:-upper_bound_resize}"
REF_VIEW_STRATEGY="${REF_VIEW_STRATEGY:-saddle_balanced}"
CONF_THRESH_PERCENTILE="${CONF_THRESH_PERCENTILE:-40}"
NUM_MAX_POINTS="${NUM_MAX_POINTS:-1000000}"
USE_RAY_POSE="false"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_STEPS="${MAX_STEPS:-30000}"
EVAL_STEPS="${EVAL_STEPS:-}"
DATA_FACTOR="${DATA_FACTOR:-1}"
SAVE_PLY_DURING_TRAIN="${SAVE_PLY_DURING_TRAIN:-false}"
ENABLE_VIEWER="${ENABLE_VIEWER:-false}"
CONDA_ENV="${CONDA_ENV:-}"
DA3_OUTPUT_DIR=""
DATASET_DIR=""
RESULT_DIR=""
FINAL_PLY=""
TEST_MODE="false"
SKIP_DA3="false"
SKIP_TRAIN="false"
MERGE="true"

if [[ -n "$CONFIG_FILE" ]]; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 2
  fi
  config_env="$(mktemp)"
  python "$PROJECT_DIR/scripts/config_to_env.py" "$CONFIG_FILE" > "$config_env"
  # shellcheck disable=SC1090
  source "$config_env"
  rm -f "$config_env"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --video) VIDEO="$2"; shift 2 ;;
    --depth-anything-repo) DEPTH_ANYTHING_REPO="$2"; shift 2 ;;
    --gsplat-examples-dir) GSPLAT_EXAMPLES_DIR="$2"; shift 2 ;;
    --scene) SCENE_NAME="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --export-format) EXPORT_FORMAT="$2"; shift 2 ;;
    --process-res) PROCESS_RES="$2"; shift 2 ;;
    --process-res-method) PROCESS_RES_METHOD="$2"; shift 2 ;;
    --ref-view-strategy) REF_VIEW_STRATEGY="$2"; shift 2 ;;
    --conf-thresh-percentile) CONF_THRESH_PERCENTILE="$2"; shift 2 ;;
    --num-max-points) NUM_MAX_POINTS="$2"; shift 2 ;;
    --use-ray-pose) USE_RAY_POSE="true"; shift ;;
    --gpus) CUDA_DEVICES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS="$2"; shift 2 ;;
    --data-factor) DATA_FACTOR="$2"; shift 2 ;;
    --save-ply-during-train) SAVE_PLY_DURING_TRAIN="true"; shift ;;
    --enable-viewer) ENABLE_VIEWER="true"; shift ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --da3-output-dir) DA3_OUTPUT_DIR="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --final-ply) FINAL_PLY="$2"; shift 2 ;;
    --test-mode) TEST_MODE="true"; shift ;;
    --skip-da3) SKIP_DA3="true"; shift ;;
    --skip-train) SKIP_TRAIN="true"; shift ;;
    --no-merge) MERGE="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "Missing required --video" >&2
  usage
  exit 2
fi
if [[ -z "$DEPTH_ANYTHING_REPO" ]]; then
  echo "Missing required --depth-anything-repo" >&2
  usage
  exit 2
fi
if [[ -z "$GSPLAT_EXAMPLES_DIR" ]]; then
  echo "Missing required --gsplat-examples-dir" >&2
  usage
  exit 2
fi

if [[ -z "$SCENE_NAME" ]]; then
  video_base="$(basename "$VIDEO")"
  SCENE_NAME="${video_base%.*}"
fi

if [[ "$TEST_MODE" == "true" ]]; then
  MAX_STEPS="100"
  EVAL_STEPS="100"
  SCENE_NAME="${SCENE_NAME}_test"
fi

if [[ -z "$EVAL_STEPS" ]]; then
  EVAL_STEPS="$MAX_STEPS"
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${WORK_DIR%/}/${SCENE_NAME}_${RUN_TAG}"
DA3_OUTPUT_DIR="${DA3_OUTPUT_DIR:-${RUN_DIR}/da3_output}"
DATASET_DIR="${DATASET_DIR:-${RUN_DIR}/gsplat_dataset}"
RESULT_DIR="${RESULT_DIR:-${RUN_DIR}/gsplat_result}"
FINAL_PLY="${FINAL_PLY:-${RUN_DIR}/${SCENE_NAME}_final.ply}"
LOG_FILE="${RUN_DIR}/pipeline.log"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "VideoGaussian pipeline"
echo "Run dir: $RUN_DIR"
echo "Video: $VIDEO"
echo "Depth Anything 3 repo: $DEPTH_ANYTHING_REPO"
echo "gsplat examples dir: $GSPLAT_EXAMPLES_DIR"
echo "DA3 output dir: $DA3_OUTPUT_DIR"
echo "gsplat dataset dir: $DATASET_DIR"
echo "gsplat result dir: $RESULT_DIR"
echo "Final PLY: $FINAL_PLY"
echo "GPUs: $CUDA_DEVICES"
echo "Max steps: $MAX_STEPS"
echo "DA3 export format: $EXPORT_FORMAT"
echo "DA3 conf thresh percentile: $CONF_THRESH_PERCENTILE"
echo "DA3 process res: $PROCESS_RES"
echo "DA3 ref view strategy: $REF_VIEW_STRATEGY"
echo "DA3 use ray pose: $USE_RAY_POSE"
echo "gsplat viewer enabled: $ENABLE_VIEWER"
echo "Start: $(date)"

if [[ -n "$CONDA_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "Conda env requested but conda activation script was not found: $CONDA_ENV" >&2
    exit 1
  fi
fi

if [[ "$SKIP_DA3" != "true" ]]; then
  echo
  echo "[1/4] Running Depth Anything 3..."
  pushd "$DEPTH_ANYTHING_REPO" >/dev/null
  da3_args=(
    video "$VIDEO"
    --model-dir "$MODEL_DIR"
    --fps "$FPS"
    --export-dir "$DA3_OUTPUT_DIR"
    --export-format "$EXPORT_FORMAT"
    --process-res "$PROCESS_RES"
    --process-res-method "$PROCESS_RES_METHOD"
    --ref-view-strategy "$REF_VIEW_STRATEGY"
    --conf-thresh-percentile "$CONF_THRESH_PERCENTILE"
    --num-max-points "$NUM_MAX_POINTS"
    --auto-cleanup
  )
  if [[ "$USE_RAY_POSE" == "true" ]]; then
    da3_args+=(--use-ray-pose)
  fi
  da3 "${da3_args[@]}"
  popd >/dev/null
else
  echo
  echo "[1/4] Skipping DA3; reusing $DA3_OUTPUT_DIR"
fi

echo
echo "[2/4] Preparing gsplat dataset..."
python "$PROJECT_DIR/scripts/prepare_gsplat_dataset.py" \
  --da3-output-dir "$DA3_OUTPUT_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --overwrite

if [[ "$SKIP_TRAIN" != "true" ]]; then
  echo
  echo "[3/4] Training gsplat..."
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  mkdir -p "$RESULT_DIR"

  MONITOR_PID=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv -l 1 > "${RESULT_DIR}/vram_peak.log" &
    MONITOR_PID="$!"
  fi

  cleanup_monitor() {
    if [[ -n "$MONITOR_PID" ]]; then
      kill "$MONITOR_PID" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_monitor EXIT

  pushd "$GSPLAT_EXAMPLES_DIR" >/dev/null
  train_args=(
    simple_trainer.py
    default
    --data-dir "$DATASET_DIR"
    --result-dir "$RESULT_DIR"
    --max-steps "$MAX_STEPS"
    --eval-steps "$EVAL_STEPS"
    --data-factor "$DATA_FACTOR"
    --disable-video
  )
  if [[ "$ENABLE_VIEWER" != "true" ]]; then
    train_args+=(--disable-viewer)
  fi
  if [[ "$SAVE_PLY_DURING_TRAIN" == "true" ]]; then
    train_args+=(--save-ply)
  fi
  python "${train_args[@]}"
  popd >/dev/null

  cleanup_monitor
  trap - EXIT

  if [[ -f "${RESULT_DIR}/vram_peak.log" ]]; then
    awk -F, 'NR>1 {if($1>max) max=$1} END {print "Peak VRAM: " max " MiB"}' \
      "${RESULT_DIR}/vram_peak.log"
  fi
else
  echo
  echo "[3/4] Skipping gsplat training; reusing $RESULT_DIR"
fi

if [[ "$MERGE" == "true" ]]; then
  echo
  echo "[4/4] Merging final checkpoints..."
  final_step=$((MAX_STEPS - 1))
  IFS=',' read -r -a gpu_array <<< "$CUDA_DEVICES"
  world_size="${#gpu_array[@]}"
  ckpts=()
  for rank in $(seq 0 $((world_size - 1))); do
    ckpts+=("${RESULT_DIR}/ckpts/ckpt_${final_step}_rank${rank}.pt")
  done

  python "$PROJECT_DIR/scripts/merge_gsplat_checkpoints.py" \
    --ckpt "${ckpts[@]}" \
    --output "$FINAL_PLY"
else
  echo
  echo "[4/4] Skipping checkpoint merge."
fi

echo
echo "Done: $(date)"
echo "Run dir: $RUN_DIR"
echo "Final PLY: $FINAL_PLY"
