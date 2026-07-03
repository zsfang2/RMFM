#!/usr/bin/env bash
set -euo pipefail

# Supplementary TokenUNet-v1 evaluation.
# This script adds matched no-sampling baselines and a small DC sweep so the
# sampling contribution can be measured against condition-only priors.
#
# Run:
#   bash scripts/run_token_unet_v1_supplement_eval.sh

WORKER_MODE=0
if [[ "${1:-}" == "--worker" ]]; then
  WORKER_MODE=1
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

RMFM_DATA_ROOT="${RMFM_DATA_ROOT:-/home/DataDisk/zsfang/dataset}"
DATASET_ROOT="${DATASET_ROOT:-$RMFM_DATA_ROOT/RadioMapSeer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/DataDisk/zsfang/rmfm/checkpoints/token_unet}"
RESULT_ROOT="${RESULT_ROOT:-/home/DataDisk/zsfang/rmfm/results/token_unet}"
CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_ROOT/radiomapseer_token_unet_flow_dpm/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULT_ROOT/token_unet_v1_supplement_per_city1_fp32}"

# GPU id in the current visible CUDA device list. Change this single number as needed.
GPU_ID=1

GAIN_MODES=(DPM)
SPLIT="test"
SPLIT_FILE=""

# Minimal rates for v1 diagnosis. Add 0.0003/0.003 here if you want denser curves.
SAMPLING_RATES=(0.0001 0.001 0.01)
NO_SAMPLING_RATE=0.0

# Matched condition matrix for sampling-gain analysis.
MATCHED_EXPERIMENTS=(
  no_condition_no_sampling
  building_no_sampling
  source_no_sampling
  building_source_no_sampling
  sampling_only
  sampling_building
  sampling_source
  sampling_building_source
)

# DC is only swept on the sampling-centric settings to keep v1 bounded.
DC_EXPERIMENTS=(
  sampling_only
  sampling_building
)
DC_STEP_SIZES=(1 3 5 10 20 50)
# Add 1 here if you also want to compare single-step DC.
DC_ITERS_SWEEP=(3)

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
K_MAX=256
DTYPE="fp32"

# Set to 1 to reuse completed summary.json files when restarting.
SKIP_EXISTING=1

# Set to 0 to run in the foreground.
RUN_IN_BACKGROUND=1
if [[ "$WORKER_MODE" == "1" ]]; then
  RUN_IN_BACKGROUND=0
fi

mkdir -p "$OUTPUT_ROOT"

RUN_LOG="$OUTPUT_ROOT/run.log"
RUN_PID="$OUTPUT_ROOT/run.pid"
RUN_COMMAND="$OUTPUT_ROOT/run_command.txt"
LAUNCH_CONFIG="$OUTPUT_ROOT/launcher_config.txt"
MATCHED_OUTPUT_DIR="$OUTPUT_ROOT/matched_conditions_no_dc"
DC_SWEEP_OUTPUT_ROOT="$OUTPUT_ROOT/dc_sweep"

write_launch_config() {
  {
    echo "launch_time=$(date -Is)"
    echo "project_root=$PROJECT_ROOT"
    echo "python_exe=$PYTHON_EXE"
    echo "dataset_root=$DATASET_ROOT"
    echo "checkpoint=$CHECKPOINT"
    echo "output_root=$OUTPUT_ROOT"
    echo "gpu_id=$GPU_ID"
    echo "gain_modes=${GAIN_MODES[*]}"
    echo "split=$SPLIT"
    echo "split_file=$SPLIT_FILE"
    echo "sampling_rates=${SAMPLING_RATES[*]}"
    echo "no_sampling_rate=$NO_SAMPLING_RATE"
    echo "matched_experiments=${MATCHED_EXPERIMENTS[*]}"
    echo "dc_experiments=${DC_EXPERIMENTS[*]}"
    echo "dc_step_sizes=${DC_STEP_SIZES[*]}"
    echo "dc_iters_sweep=${DC_ITERS_SWEEP[*]}"
    echo "num_samples=$NUM_SAMPLES"
    echo "samples_per_city=$SAMPLES_PER_CITY"
    echo "num_steps=$NUM_STEPS"
    echo "k_max=$K_MAX"
    echo "dtype=$DTYPE"
    echo "measurement_noise_std=$MEASUREMENT_NOISE_STD"
    echo "skip_existing=$SKIP_EXISTING"
    echo "run_in_background=$RUN_IN_BACKGROUND"
  } > "$LAUNCH_CONFIG"

  {
    printf 'cd %q\n' "$PROJECT_ROOT"
    printf 'bash %q --worker\n' "$PROJECT_ROOT/scripts/run_token_unet_v1_supplement_eval.sh"
  } > "$RUN_COMMAND"
}

run_benchmark() {
  local output_dir="$1"
  local sampler_mode="$2"
  local step_size="$3"
  local dc_iters="$4"
  shift 4
  local experiments=("$@")

  local cmd=(
    "$PYTHON_EXE" "$PROJECT_ROOT/scripts/benchmark_token_unet_phase1.py"
    --dataset_root "$DATASET_ROOT"
    --checkpoint "$CHECKPOINT"
    --output_dir "$output_dir"
    --gain_modes "${GAIN_MODES[@]}"
    --split "$SPLIT"
    --sampling_rates "${SAMPLING_RATES[@]}"
    --no_sampling_rate "$NO_SAMPLING_RATE"
    --experiments "${experiments[@]}"
    --sampler_modes "$sampler_mode"
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
    --step_size "$step_size"
    --dc_iters "$dc_iters"
    --k_max "$K_MAX"
    --dtype "$DTYPE"
    -gpu "$GPU_ID"
    --no_progress
  )

  if [[ -n "$SPLIT_FILE" ]]; then
    cmd+=(--split_file "$SPLIT_FILE")
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip_existing)
  fi

  echo
  echo "Running sampler_mode=$sampler_mode step_size=$step_size dc_iters=$dc_iters"
  echo "Output: $output_dir"
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

main_run() {
  cd "$PROJECT_ROOT"
  echo "Started TokenUNet-v1 supplementary evaluation at $(date -Is)"
  echo "Matched noDC output: $MATCHED_OUTPUT_DIR"
  echo "DC sweep output: $DC_SWEEP_OUTPUT_ROOT"

  run_benchmark "$MATCHED_OUTPUT_DIR" "no_dc" "0.0" "0" "${MATCHED_EXPERIMENTS[@]}"

  for dc_iters in "${DC_ITERS_SWEEP[@]}"; do
    for step_size in "${DC_STEP_SIZES[@]}"; do
      step_tag="${step_size//./p}"
      output_dir="$DC_SWEEP_OUTPUT_ROOT/step_${step_tag}_iters_${dc_iters}"
      run_benchmark "$output_dir" "dc" "$step_size" "$dc_iters" "${DC_EXPERIMENTS[@]}"
    done
  done

  echo
  echo "Finished TokenUNet-v1 supplementary evaluation at $(date -Is)"
  echo "Main summary: $MATCHED_OUTPUT_DIR/token_unet_phase1_summary.csv"
  echo "DC sweep summaries are under: $DC_SWEEP_OUTPUT_ROOT"
}

write_launch_config

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  if [[ -f "$RUN_PID" ]]; then
    old_pid="$(cat "$RUN_PID" || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Refusing to start: existing process $old_pid from $RUN_PID is still running." >&2
      exit 1
    fi
  fi

  : > "$RUN_LOG"
  nohup bash "$PROJECT_ROOT/scripts/run_token_unet_v1_supplement_eval.sh" --worker >> "$RUN_LOG" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_PID"

  echo "Started background TokenUNet-v1 supplementary eval."
  echo "PID: $pid"
  echo "Log: $RUN_LOG"
  echo "PID file: $RUN_PID"
  echo "Command record: $RUN_COMMAND"
else
  if [[ "$WORKER_MODE" == "0" ]]; then
    : > "$RUN_LOG"
    exec > >(tee -a "$RUN_LOG") 2>&1
    rm -f "$RUN_PID"
  fi
  main_run
fi
