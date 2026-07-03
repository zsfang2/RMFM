#!/usr/bin/env bash
set -euo pipefail

# Edit the variables in this block, then run:
#   bash scripts/run_sampling_only_collapse_sweep.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

RMFM_DATA_ROOT="${RMFM_DATA_ROOT:-/home/DataDisk/zsfang/dataset}"
DATASET_ROOT="${DATASET_ROOT:-$RMFM_DATA_ROOT/RadioMapSeer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/DataDisk/zsfang/rmfm/checkpoints/unet}"
RESULT_ROOT="${RESULT_ROOT:-/home/DataDisk/zsfang/rmfm/results/unet}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/sample_only_collapse_sweep_0p01}"

# Physical GPU id shown by nvidia-smi. Change this single number as needed.
GPU_ID=1
DEVICE="cuda:0"

# DPM is in-distribution for the trained DPM model. IRT2/IRT4 are OOD inference sets.
GAIN_MODES=(DPM IRT2 IRT4)
SPLIT="test"
SPLIT_FILE=""

# Ratios, not percentages:
# 0.0001 = 0.01%, 0.01 = 1%, 1.00 = 100%.
# EXTRA_SAMPLING_RATES is useful for keeping very low anchor points while using
# a coarser regular sweep.
EXTRA_SAMPLING_RATES=(0.0001)
SAMPLING_RATE_START=0.01
SAMPLING_RATE_END=1.0
SAMPLING_RATE_STEP=0.01

# Select this many samples from each test city. If a city has fewer samples, all available
# samples are used. Set to 0 to disable city stratification.
SAMPLES_PER_CITY=10

# Keep -1 so samples_per_city controls the sample count. Use a positive value only for debugging.
NUM_SAMPLES=-1
START_INDEX=0

# This sweep is for finding where sample-only inference collapses.
EXPERIMENTS=(sampling_only)

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

# Set to 1 to reuse completed summary.json files when restarting the same output directory.
SKIP_EXISTING=1

# Set to 0 if you want the process to stay in the foreground.
RUN_IN_BACKGROUND=1

mkdir -p "$OUTPUT_DIR"

RUN_LOG="$OUTPUT_DIR/run.log"
RUN_PID="$OUTPUT_DIR/run.pid"
RUN_COMMAND="$OUTPUT_DIR/run_command.txt"
LAUNCH_CONFIG="$OUTPUT_DIR/launcher_config.txt"

mapfile -t RANGE_SAMPLING_RATES < <(
  "$PYTHON_EXE" -c "from decimal import Decimal; start=Decimal('$SAMPLING_RATE_START'); end=Decimal('$SAMPLING_RATE_END'); step=Decimal('$SAMPLING_RATE_STEP'); x=start
while x <= end:
    print(x)
    x += step"
)
SAMPLING_RATES=("${EXTRA_SAMPLING_RATES[@]}" "${RANGE_SAMPLING_RATES[@]}")

CMD=(
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/benchmark_information_ablations.py"
  --dataset_root "$DATASET_ROOT"
  --checkpoint "$CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --gain_modes "${GAIN_MODES[@]}"
  --split "$SPLIT"
  --sampling_rates "${SAMPLING_RATES[@]}"
  --experiments "${EXPERIMENTS[@]}"
  --num_samples "$NUM_SAMPLES"
  --start_index "$START_INDEX"
  --samples_per_city "$SAMPLES_PER_CITY"
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
  echo "cuda_visible_devices=$GPU_ID"
  echo "device=$DEVICE"
  echo "gain_modes=${GAIN_MODES[*]}"
  echo "split=$SPLIT"
  echo "split_file=$SPLIT_FILE"
  echo "extra_sampling_rates=${EXTRA_SAMPLING_RATES[*]}"
  echo "sampling_rate_start=$SAMPLING_RATE_START"
  echo "sampling_rate_end=$SAMPLING_RATE_END"
  echo "sampling_rate_step=$SAMPLING_RATE_STEP"
  echo "sampling_rates=${SAMPLING_RATES[*]}"
  echo "sampling_rate_count=${#SAMPLING_RATES[@]}"
  echo "samples_per_city=$SAMPLES_PER_CITY"
  echo "num_samples=$NUM_SAMPLES"
  echo "start_index=$START_INDEX"
  echo "experiments=${EXPERIMENTS[*]}"
  echo "seed=$SEED"
  echo "image_size=$IMAGE_SIZE"
  echo "data_channels=$DATA_CHANNELS"
  echo "heatmap_sigma=$HEATMAP_SIGMA"
  echo "third_channel=$THIRD_CHANNEL"
  echo "measurement_noise_std=$MEASUREMENT_NOISE_STD"
  echo "num_steps=$NUM_STEPS"
  echo "step_size=$STEP_SIZE"
  echo "dc_iters=$DC_ITERS"
  echo "dtype=$DTYPE"
  echo "skip_existing=$SKIP_EXISTING"
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

  echo "Started background sample-only collapse sweep."
  echo "PID: $pid"
  echo "Log: $RUN_LOG"
  echo "PID file: $RUN_PID"
  echo "Command record: $RUN_COMMAND"
  echo "Launcher config: $LAUNCH_CONFIG"
else
  rm -f "$RUN_PID"
  cd "$PROJECT_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
fi
