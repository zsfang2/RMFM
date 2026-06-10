#!/usr/bin/env bash
set -euo pipefail

# Edit this file before running:
#   bash scripts/run_token_unet_phase1_train.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

RMFM_DATA_ROOT="${RMFM_DATA_ROOT:-$PROJECT_ROOT/data}"
DATASET_ROOT="${DATASET_ROOT:-$RMFM_DATA_ROOT/RadiomapSeer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/outputs/rmfm/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/radiomapseer_token_unet_flow_dpm}"

# Physical GPU id shown by nvidia-smi. Change this single number as needed.
GPU_ID="${GPU_ID:-1}"
DEVICE="cuda:0"

GAIN_MODES_TEXT="${GAIN_MODES_TEXT:-DPM}"
read -r -a GAIN_MODES <<< "$GAIN_MODES_TEXT"
IMAGE_SIZE=256
DATA_CHANNELS=3
HEATMAP_SIGMA=20.0
THIRD_CHANNEL="auto"

BASE_CHANNELS=64
CHANNEL_MULTS="1,2,4,4"
LAYERS_PER_BLOCK=2
NORM_NUM_GROUPS=32
ATTENTION_HEAD_DIM=8
TRANSFORMER_LAYERS_PER_BLOCK=1

TOKEN_DIM=256
TOKENIZER_LAYERS=2
TOKENIZER_HEADS=4
FOURIER_BANDS=32
K_MAX=256
SPARSE_RATES="0.0,0.0001,0.0003,0.001,0.003,0.005,0.01"

BATCH_SIZE=16
NUM_WORKERS=4
PREFETCH_FACTOR=4
MAX_STEPS=100000
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_ACCUM_STEPS=1
CLIP_GRAD_NORM=1.0
MIXED_PRECISION="fp16"

VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42
MAX_TRAIN_SAMPLES=-1
MAX_VAL_SAMPLES=512
LOG_EVERY=50
VAL_EVERY=1000
SAVE_EVERY=5000
RESUME=""

CHANNELS_LAST=1
ALLOW_TF32=1
COMPILE=0

# Set to 0 if you want the process to stay in the foreground.
RUN_IN_BACKGROUND=1

mkdir -p "$OUTPUT_DIR"

RUN_LOG="$OUTPUT_DIR/run.log"
RUN_PID="$OUTPUT_DIR/run.pid"
RUN_COMMAND="$OUTPUT_DIR/run_command.txt"
LAUNCH_CONFIG="$OUTPUT_DIR/launcher_config.txt"

CMD=(
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/train_token_unet_flow.py"
  --dataset_root "$DATASET_ROOT"
  --output_dir "$OUTPUT_DIR"
  --gain_modes "${GAIN_MODES[@]}"
  --image_size "$IMAGE_SIZE"
  --data_channels "$DATA_CHANNELS"
  --heatmap_sigma "$HEATMAP_SIGMA"
  --third_channel "$THIRD_CHANNEL"
  --base_channels "$BASE_CHANNELS"
  --channel_mults "$CHANNEL_MULTS"
  --layers_per_block "$LAYERS_PER_BLOCK"
  --norm_num_groups "$NORM_NUM_GROUPS"
  --attention_head_dim "$ATTENTION_HEAD_DIM"
  --transformer_layers_per_block "$TRANSFORMER_LAYERS_PER_BLOCK"
  --token_dim "$TOKEN_DIM"
  --tokenizer_layers "$TOKENIZER_LAYERS"
  --tokenizer_heads "$TOKENIZER_HEADS"
  --fourier_bands "$FOURIER_BANDS"
  --k_max "$K_MAX"
  --sparse_rates "$SPARSE_RATES"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --max_steps "$MAX_STEPS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --clip_grad_norm "$CLIP_GRAD_NORM"
  --mixed_precision "$MIXED_PRECISION"
  --device "$DEVICE"
  --val_ratio "$VAL_RATIO"
  --test_ratio "$TEST_RATIO"
  --seed "$SEED"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --max_val_samples "$MAX_VAL_SAMPLES"
  --log_every "$LOG_EVERY"
  --val_every "$VAL_EVERY"
  --save_every "$SAVE_EVERY"
)

if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi
if [[ "$CHANNELS_LAST" == "1" ]]; then
  CMD+=(--channels_last)
fi
if [[ "$ALLOW_TF32" == "1" ]]; then
  CMD+=(--allow_tf32)
fi
if [[ "$COMPILE" == "1" ]]; then
  CMD+=(--compile)
fi

{
  echo "launch_time=$(date -Is)"
  echo "project_root=$PROJECT_ROOT"
  echo "python_exe=$PYTHON_EXE"
  echo "dataset_root=$DATASET_ROOT"
  echo "output_dir=$OUTPUT_DIR"
  echo "gpu_id=$GPU_ID"
  echo "cuda_visible_devices=$GPU_ID"
  echo "device=$DEVICE"
  echo "gain_modes=${GAIN_MODES[*]}"
  echo "image_size=$IMAGE_SIZE"
  echo "base_channels=$BASE_CHANNELS"
  echo "channel_mults=$CHANNEL_MULTS"
  echo "token_dim=$TOKEN_DIM"
  echo "k_max=$K_MAX"
  echo "sparse_rates=$SPARSE_RATES"
  echo "batch_size=$BATCH_SIZE"
  echo "max_steps=$MAX_STEPS"
  echo "mixed_precision=$MIXED_PRECISION"
  echo "channels_last=$CHANNELS_LAST"
  echo "allow_tf32=$ALLOW_TF32"
  echo "compile=$COMPILE"
  echo "run_in_background=$RUN_IN_BACKGROUND"
} > "$LAUNCH_CONFIG"

{
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_ID"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_COMMAND"

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  if [[ -f "$RUN_PID" ]]; then
    old_pid="$(cat "$RUN_PID" || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Refusing to start: existing process $old_pid from $RUN_PID is still running." >&2
      exit 1
    fi
  fi

  : > "$RUN_LOG"
  nohup bash -c '
    set -euo pipefail
    project_root="$1"
    gpu_id="$2"
    shift 2
    cd "$project_root"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    exec "$@"
  ' _ "$PROJECT_ROOT" "$GPU_ID" "${CMD[@]}" >> "$RUN_LOG" 2>&1 &

  pid=$!
  echo "$pid" > "$RUN_PID"

  echo "Started background TokenUNet training."
  echo "PID: $pid"
  echo "Log: $RUN_LOG"
  echo "PID file: $RUN_PID"
  echo "Command record: $RUN_COMMAND"
else
  rm -f "$RUN_PID"
  cd "$PROJECT_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
fi
