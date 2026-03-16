#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

GPU_NUMBER="${GPU_NUMBER:-}"
GPU_OFFSET="${GPU_OFFSET:-0}"
META_ROOT="${META_ROOT:-}"
CHECKPOINT="${CHECKPOINT:-}"
MODE="${MODE:-batch_refresh}"
STREAMING_MODE="${STREAMING_MODE:-causal}"
KEYFRAME_STRIDE="${KEYFRAME_STRIDE:-8}"
REFRESH="${REFRESH:-3}"
CONFIG="${CONFIG:-$ROOT_DIR/configs/longstream_infer.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/multi_gpu}"
DATA_ROOTS_FILE="${DATA_ROOTS_FILE:-data_roots.txt}"
CAMERA="${CAMERA:-}"
RUN_EVAL="${RUN_EVAL:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
MAX_FULL_POINTCLOUD_POINTS="${MAX_FULL_POINTCLOUD_POINTS:-500000}"
MAX_FRAME_POINTCLOUD_POINTS="${MAX_FRAME_POINTCLOUD_POINTS:-50000}"
POINT_F1_THRESHOLD="${POINT_F1_THRESHOLD:-0.25}"
EVAL_MAX_POINTS="${EVAL_MAX_POINTS:-100000}"
EVAL_VOXEL_SIZE="${EVAL_VOXEL_SIZE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  cat <<'EOF'
Usage:
  ./run_multi_gpu.sh --gpu-number 4 --meta-root /path/to/meta_root --checkpoint /path/to/model.pt

Required args:
  --gpu-number <int>    number of GPUs to use
  --meta-root <path>    dataset meta root in generalizable format
  --checkpoint <path>   local checkpoint path

Optional args:
  --gpu-offset <int>          first physical GPU id, default: 0
  --mode <str>                batch_refresh | streaming_refresh, default: batch_refresh
  --streaming-mode <str>      causal | window, default: causal
  --keyframe-stride <int>     default: 8
  --refresh <int>             default: 3, counts both keyframe endpoints
  --config <path>             default: configs/longstream_infer.yaml
  --output-root <path>        default: outputs/multi_gpu
  --data-roots-file <name>    default: data_roots.txt
  --camera <id>               optional camera id, default: auto-pick first camera folder
  --max-frames <int>          optional cap per sequence
  --max-full-pointcloud-points <int>
                              default: 500000
  --max-frame-pointcloud-points <int>
                              default: 50000
  --point-f1-threshold <float>
                              default: 0.25
  --eval-max-points <int>     default: 100000
  --eval-voxel-size <float>   optional voxel downsample size for CD/F1 eval
  --run-eval <0|1>            1 to run eval.py after all workers finish, default: 1
  --python <exe>              python executable, default: python3
  --help                      show this message

Extra args:
  Use `--` to pass remaining args directly to each `run.py` worker.

Example:
  ./run_multi_gpu.sh \
    --gpu-number 4 \
    --meta-root /horizon-bucket/.../kitti/dataset \
    --checkpoint checkpoints/50_longstream.pt \
    --mode batch_refresh \
    --max-full-pointcloud-points 2000000 \
    --max-frame-pointcloud-points 200000 \
    --point-f1-threshold 0.25 \
    --eval-max-points 100000 \
    --output-root outputs/kitti_all_batch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-number)
      GPU_NUMBER="$2"
      shift 2
      ;;
    --gpu-offset)
      GPU_OFFSET="$2"
      shift 2
      ;;
    --meta-root)
      META_ROOT="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --streaming-mode)
      STREAMING_MODE="$2"
      shift 2
      ;;
    --keyframe-stride)
      KEYFRAME_STRIDE="$2"
      shift 2
      ;;
    --refresh)
      REFRESH="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --data-roots-file)
      DATA_ROOTS_FILE="$2"
      shift 2
      ;;
    --camera)
      CAMERA="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --max-full-pointcloud-points)
      MAX_FULL_POINTCLOUD_POINTS="$2"
      shift 2
      ;;
    --max-frame-pointcloud-points)
      MAX_FRAME_POINTCLOUD_POINTS="$2"
      shift 2
      ;;
    --point-f1-threshold)
      POINT_F1_THRESHOLD="$2"
      shift 2
      ;;
    --eval-max-points)
      EVAL_MAX_POINTS="$2"
      shift 2
      ;;
    --eval-voxel-size)
      EVAL_VOXEL_SIZE="$2"
      shift 2
      ;;
    --run-eval)
      RUN_EVAL="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGV=( "$@" )
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$GPU_NUMBER" || -z "$META_ROOT" || -z "$CHECKPOINT" ]]; then
  usage
  exit 1
fi

if ! [[ "$GPU_NUMBER" =~ ^[0-9]+$ ]] || [[ "$GPU_NUMBER" -le 0 ]]; then
  echo "GPU_NUMBER must be a positive integer" >&2
  exit 1
fi

if ! [[ "$GPU_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "GPU_OFFSET must be a non-negative integer" >&2
  exit 1
fi

if [[ ! -d "$META_ROOT" ]]; then
  echo "META_ROOT does not exist: $META_ROOT" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "CHECKPOINT does not exist: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

mapfile -t SEQS < <("$PYTHON" - "$META_ROOT" "$DATA_ROOTS_FILE" <<'PY'
import glob
import os
import sys

meta_root = sys.argv[1]
data_roots_file = sys.argv[2]

def read_list(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seqs.append(line)
    return seqs

if os.path.isdir(os.path.join(meta_root, "images")):
    print(meta_root)
    raise SystemExit(0)

list_candidates = []
if data_roots_file:
    list_candidates.append(os.path.join(meta_root, data_roots_file))
list_candidates.extend([
    os.path.join(meta_root, "data_roots.txt"),
    os.path.join(meta_root, "train_data_roots.txt"),
    os.path.join(meta_root, "validate_data_roots.txt"),
])

seen = set()
for path in list_candidates:
    if path in seen:
        continue
    seen.add(path)
    if os.path.isfile(path):
        for seq in read_list(path):
            print(seq)
        raise SystemExit(0)

scene_roots = []
for images_dir in sorted(glob.glob(os.path.join(meta_root, "**", "images"), recursive=True)):
    scene_root = os.path.dirname(images_dir)
    scene_roots.append(os.path.relpath(scene_root, meta_root))

for seq in sorted(set(scene_roots)):
    print(seq)
PY
)

if [[ "${#SEQS[@]}" -eq 0 ]]; then
  echo "No sequences found under META_ROOT: $META_ROOT" >&2
  exit 1
fi

if [[ -z "${EXTRA_ARGV+x}" ]]; then
  read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"
fi

chunk_size=$(( (${#SEQS[@]} + GPU_NUMBER - 1) / GPU_NUMBER ))
worker_count=0
declare -a pids=()

echo "[longstream] multi-gpu launch"
echo "[longstream] meta_root=$META_ROOT"
echo "[longstream] output_root=$OUTPUT_ROOT"
echo "[longstream] seq_count=${#SEQS[@]}"
echo "[longstream] gpu_number=$GPU_NUMBER gpu_offset=$GPU_OFFSET"
echo "[longstream] max_full_pointcloud_points=$MAX_FULL_POINTCLOUD_POINTS"
echo "[longstream] max_frame_pointcloud_points=$MAX_FRAME_POINTCLOUD_POINTS"
echo "[longstream] point_f1_threshold=$POINT_F1_THRESHOLD eval_max_points=$EVAL_MAX_POINTS eval_voxel_size=${EVAL_VOXEL_SIZE:-null}"

for ((worker_idx=0; worker_idx<GPU_NUMBER; worker_idx++)); do
  start=$((worker_idx * chunk_size))
  if [[ "$start" -ge "${#SEQS[@]}" ]]; then
    break
  fi

  end=$((start + chunk_size))
  if [[ "$end" -gt "${#SEQS[@]}" ]]; then
    end=${#SEQS[@]}
  fi

  chunk=( "${SEQS[@]:start:end-start}" )
  if [[ "${#chunk[@]}" -eq 0 ]]; then
    continue
  fi

  seq_csv="$(IFS=,; echo "${chunk[*]}")"
  gpu_id=$((GPU_OFFSET + worker_idx))
  worker_count=$((worker_count + 1))

  cmd=(
    "$PYTHON" "$ROOT_DIR/run.py"
    --config "$CONFIG"
    --img-path "$META_ROOT"
    --checkpoint "$CHECKPOINT"
    --mode "$MODE"
    --streaming-mode "$STREAMING_MODE"
    --keyframe-stride "$KEYFRAME_STRIDE"
    --refresh "$REFRESH"
    --data-roots-file "$DATA_ROOTS_FILE"
    --output-root "$OUTPUT_ROOT"
    --seq-list "$seq_csv"
    --max-full-pointcloud-points "$MAX_FULL_POINTCLOUD_POINTS"
    --max-frame-pointcloud-points "$MAX_FRAME_POINTCLOUD_POINTS"
    --skip-eval
  )

  if [[ -n "$CAMERA" ]]; then
    cmd+=( --camera "$CAMERA" )
  fi

  if [[ -n "$MAX_FRAMES" ]]; then
    cmd+=( --max-frames "$MAX_FRAMES" )
  fi

  if [[ "${#EXTRA_ARGV[@]}" -gt 0 ]]; then
    cmd+=( "${EXTRA_ARGV[@]}" )
  fi

  echo "[longstream] gpu=$gpu_id seqs=${#chunk[@]} -> ${seq_csv}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    exec "${cmd[@]}"
  ) &
  pids+=( "$!" )
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[longstream] one or more worker runs failed" >&2
  exit "$status"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_cmd=(
    "$PYTHON" "$ROOT_DIR/eval.py"
    --config "$CONFIG"
    --img-path "$META_ROOT"
    --data-roots-file "$DATA_ROOTS_FILE"
    --output-root "$OUTPUT_ROOT"
    --point-f1-threshold "$POINT_F1_THRESHOLD"
    --eval-max-points "$EVAL_MAX_POINTS"
  )

  if [[ -n "$EVAL_VOXEL_SIZE" ]]; then
    eval_cmd+=( --eval-voxel-size "$EVAL_VOXEL_SIZE" )
  fi

  if [[ -n "$CAMERA" ]]; then
    eval_cmd+=( --camera "$CAMERA" )
  fi

  if [[ -n "$MAX_FRAMES" ]]; then
    eval_cmd+=( --max-frames "$MAX_FRAMES" )
  fi

  echo "[longstream] all workers finished, running evaluation"
  "${eval_cmd[@]}"
fi

echo "[longstream] multi-gpu run complete"
