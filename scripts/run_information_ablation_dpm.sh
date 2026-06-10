#!/usr/bin/env bash
set -euo pipefail

# Edit this file before running:
#   bash scripts/run_information_ablation_dpm.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

RMFM_DATA_ROOT="${RMFM_DATA_ROOT:-$PROJECT_ROOT/data}"
DATASET_ROOT="${DATASET_ROOT:-$RMFM_DATA_ROOT/RadiomapSeer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/outputs/rmfm/checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/outputs/rmfm/results}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/information_ablation_dpm}"

# Physical GPU id shown by nvidia-smi. Change this single number as needed.
GPU_ID=1

GAIN_MODES=(DPM)
SPLIT="test"
SPLIT_FILE=""

# Ratios, not percentages:
# 0.0001 = 0.01%, 0.0010 = 0.10%.
SAMPLING_RATES=(
  0.0001
  0.0002
  0.0003
  0.0004
  0.0005
  0.0006
  0.0007
  0.0008
  0.0009
  0.0010
)

# The final group is a no-sampling reference with building + source only.
EXPERIMENTS=(
  sampling_only
  sampling_building
  sampling_source
  sampling_building_source
  building_source_no_sampling
)

# Used only by building_source_no_sampling.
NO_SAMPLING_RATE=0.0

NUM_SAMPLES=50
START_INDEX=0
SEED=42
IMAGE_SIZE=256
DATA_CHANNELS=3
HEATMAP_SIGMA=20.0
THIRD_CHANNEL="auto"
MEASUREMENT_NOISE_STD=0.03

NUM_STEPS=20
STEP_SIZE=50.0
DC_ITERS=3
DTYPE="fp16"

CUDA_VISIBLE_DEVICES_VALUE="$GPU_ID"
DEVICE="cuda:0"

# Set to 1 if you want to reuse completed summary.json files.
SKIP_EXISTING=0

# Set to 0 if you want the process to stay in the foreground.
RUN_IN_BACKGROUND=1

mkdir -p "$OUTPUT_DIR"

RUN_LOG="$OUTPUT_DIR/run.log"
RUN_PID="$OUTPUT_DIR/run.pid"
RUN_COMMAND="$OUTPUT_DIR/run_command.txt"
LAUNCH_CONFIG="$OUTPUT_DIR/launcher_config.txt"

CMD=(
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/benchmark_information_ablations.py"
  --dataset_root "$DATASET_ROOT"
  --checkpoint "$CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --gain_modes "${GAIN_MODES[@]}"
  --split "$SPLIT"
  --sampling_rates "${SAMPLING_RATES[@]}"
  --no_sampling_rate "$NO_SAMPLING_RATE"
  --experiments "${EXPERIMENTS[@]}"
  --num_samples "$NUM_SAMPLES"
  --start_index "$START_INDEX"
  --seed "$SEED"
  --image_size "$IMAGE_SIZE"
  --data_channels "$DATA_CHANNELS"
  --heatmap_sigma "$HEATMAP_SIGMA"
  --third_channel "$THIRD_CHANNEL"
  --measurement_noise_std "$MEASUREMENT_NOISE_STD"
  --num_steps "$NUM_STEPS"
  --step_size "$STEP_SIZE"
  --dc_iters "$DC_ITERS"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --no_progress
)

if [[ -n "$SPLIT_FILE" ]]; then
  CMD+=(--split_file "$SPLIT_FILE")
fi

if [[ "$SKIP_EXISTING" == "1" ]]; then
  CMD+=(--skip_existing)
fi

{
  echo "launch_time=$(date -Is)"
  echo "project_root=$PROJECT_ROOT"
  echo "python_exe=$PYTHON_EXE"
  echo "dataset_root=$DATASET_ROOT"
  echo "checkpoint=$CHECKPOINT"
  echo "output_dir=$OUTPUT_DIR"
  echo "gpu_id=$GPU_ID"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES_VALUE"
  echo "device=$DEVICE"
  echo "gain_modes=${GAIN_MODES[*]}"
  echo "split=$SPLIT"
  echo "sampling_rates=${SAMPLING_RATES[*]}"
  echo "no_sampling_rate=$NO_SAMPLING_RATE"
  echo "experiments=${EXPERIMENTS[*]}"
  echo "num_samples=$NUM_SAMPLES"
  echo "num_steps=$NUM_STEPS"
  echo "step_size=$STEP_SIZE"
  echo "dc_iters=$DC_ITERS"
  echo "dtype=$DTYPE"
  echo "skip_existing=$SKIP_EXISTING"
  echo "run_in_background=$RUN_IN_BACKGROUND"
} > "$LAUNCH_CONFIG"

{
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES_VALUE"
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
    cuda_visible_devices="$2"
    shift 2
    cd "$project_root"
    export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
    exec "$@"
  ' _ "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES_VALUE" "${CMD[@]}" >> "$RUN_LOG" 2>&1 &

  pid=$!
  echo "$pid" > "$RUN_PID"

  echo "Started background run."
  echo "PID: $pid"
  echo "Log: $RUN_LOG"
  echo "PID file: $RUN_PID"
  echo "Command record: $RUN_COMMAND"
else
  rm -f "$RUN_PID"
  cd "$PROJECT_ROOT"
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
  exec "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
fi
