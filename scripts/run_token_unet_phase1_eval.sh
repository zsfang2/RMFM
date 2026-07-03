#!/usr/bin/env bash
set -euo pipefail

# Edit this file after TokenUNet training, then run:
#   bash scripts/run_token_unet_phase1_eval.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

RMFM_DATA_ROOT="${RMFM_DATA_ROOT:-/home/DataDisk/zsfang/dataset}"
DATASET_ROOT="${DATASET_ROOT:-$RMFM_DATA_ROOT/RadioMapSeer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/DataDisk/zsfang/rmfm/checkpoints/token_unet}"
RESULT_ROOT="${RESULT_ROOT:-/home/DataDisk/zsfang/rmfm/results/token_unet}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/radiomapseer_token_unet_flow_dpm/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/token_unet_phase1_per_city1_fp32}"

# GPU id in the current visible CUDA device list. Change this single number as needed.
GPU_ID=1

GAIN_MODES=(DPM)
SPLIT="test"
SPLIT_FILE=""

SAMPLING_RATES=(0.0001 0.0003 0.001 0.003 0.01)
SAMPLER_MODES=(no_dc dc)
EXPERIMENTS=(
  sampling_only
  sampling_building
  sampling_source
  sampling_building_source
  building_source_no_sampling
)
NO_SAMPLING_RATE=0.0

NUM_SAMPLES=-1
START_INDEX=0
SAMPLES_PER_CITY=1
SEED=42
IMAGE_SIZE=256
DATA_CHANNELS=3
HEATMAP_SIGMA=20.0
THIRD_CHANNEL="auto"
MEASUREMENT_NOISE_STD=0.03

NUM_STEPS=20
STEP_SIZE=50.0
DC_ITERS=3
K_MAX=256
DTYPE="fp32"

# Set to 1 to reuse completed summary.json files when restarting.
SKIP_EXISTING=1

# Set to 0 if you want the process to stay in the foreground.
RUN_IN_BACKGROUND=1

mkdir -p "$OUTPUT_DIR"

RUN_LOG="$OUTPUT_DIR/run.log"
RUN_PID="$OUTPUT_DIR/run.pid"
RUN_COMMAND="$OUTPUT_DIR/run_command.txt"
LAUNCH_CONFIG="$OUTPUT_DIR/launcher_config.txt"

CMD=(
  "$PYTHON_EXE" "$PROJECT_ROOT/scripts/benchmark_token_unet_phase1.py"
  --dataset_root "$DATASET_ROOT"
  --checkpoint "$CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --gain_modes "${GAIN_MODES[@]}"
  --split "$SPLIT"
  --sampling_rates "${SAMPLING_RATES[@]}"
  --no_sampling_rate "$NO_SAMPLING_RATE"
  --experiments "${EXPERIMENTS[@]}"
  --sampler_modes "${SAMPLER_MODES[@]}"
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
  --k_max "$K_MAX"
  --dtype "$DTYPE"
  -gpu "$GPU_ID"
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
  echo "gain_modes=${GAIN_MODES[*]}"
  echo "split=$SPLIT"
  echo "split_file=$SPLIT_FILE"
  echo "sampling_rates=${SAMPLING_RATES[*]}"
  echo "no_sampling_rate=$NO_SAMPLING_RATE"
  echo "sampler_modes=${SAMPLER_MODES[*]}"
  echo "experiments=${EXPERIMENTS[*]}"
  echo "num_samples=$NUM_SAMPLES"
  echo "samples_per_city=$SAMPLES_PER_CITY"
  echo "num_steps=$NUM_STEPS"
  echo "step_size=$STEP_SIZE"
  echo "dc_iters=$DC_ITERS"
  echo "k_max=$K_MAX"
  echo "dtype=$DTYPE"
  echo "skip_existing=$SKIP_EXISTING"
  echo "run_in_background=$RUN_IN_BACKGROUND"
} > "$LAUNCH_CONFIG"

{
  printf 'cd %q\n' "$PROJECT_ROOT"
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
    shift 1
    cd "$project_root"
    exec "$@"
  ' _ "$PROJECT_ROOT" "${CMD[@]}" >> "$RUN_LOG" 2>&1 &

  pid=$!
  echo "$pid" > "$RUN_PID"

  echo "Started background TokenUNet phase-1 eval."
  echo "PID: $pid"
  echo "Log: $RUN_LOG"
  echo "PID file: $RUN_PID"
  echo "Command record: $RUN_COMMAND"
else
  rm -f "$RUN_PID"
  cd "$PROJECT_ROOT"
  exec "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
fi
