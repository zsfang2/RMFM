#!/usr/bin/env bash
set -euo pipefail

# Run this from the project root:
#   cd /path/to/RMFM
#   conda activate rmfm
#   bash scripts/run_token_unet_irt4_train.sh

if [[ ! -f "scripts/train_token_unet_flow.py" || ! -d "rmfm" ]]; then
  echo "Please run this script from the RMFM project root." >&2
  exit 1
fi

PYTHON_EXE="python"

DATASET_ROOT="data/RadiomapSeer"
OUTPUT_DIR="outputs/rmfm/checkpoints/radiomapseer_token_unet_flow_irt4"

GPU_ID=0

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

LOCAL_CONFIG="configs/local_paths.sh"
if [[ -f "$LOCAL_CONFIG" ]]; then
  # Machine-specific paths and overrides live here and are intentionally ignored by git.
  # See configs/local_paths.example.sh for the available variables.
  source "$LOCAL_CONFIG"
fi

if [[ ! -d "$DATASET_ROOT/gain/IRT4" ]]; then
  echo "IRT4 data not found under: $DATASET_ROOT/gain/IRT4" >&2
  echo "Edit configs/local_paths.sh or DATASET_ROOT in this script." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Training TokenUNet on RadioMapSeer IRT4"
echo "python=$(command -v "$PYTHON_EXE")"
echo "dataset_root=$DATASET_ROOT"
echo "output_dir=$OUTPUT_DIR"
echo "gpu_id=$GPU_ID"
echo "batch_size=$BATCH_SIZE"
echo "max_steps=$MAX_STEPS"
echo

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

"$PYTHON_EXE" scripts/train_token_unet_flow.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --gain_modes IRT4 \
  --image_size "$IMAGE_SIZE" \
  --data_channels "$DATA_CHANNELS" \
  --heatmap_sigma "$HEATMAP_SIGMA" \
  --third_channel "$THIRD_CHANNEL" \
  --base_channels "$BASE_CHANNELS" \
  --channel_mults "$CHANNEL_MULTS" \
  --layers_per_block "$LAYERS_PER_BLOCK" \
  --norm_num_groups "$NORM_NUM_GROUPS" \
  --attention_head_dim "$ATTENTION_HEAD_DIM" \
  --transformer_layers_per_block "$TRANSFORMER_LAYERS_PER_BLOCK" \
  --token_dim "$TOKEN_DIM" \
  --tokenizer_layers "$TOKENIZER_LAYERS" \
  --tokenizer_heads "$TOKENIZER_HEADS" \
  --fourier_bands "$FOURIER_BANDS" \
  --k_max "$K_MAX" \
  --sparse_rates "$SPARSE_RATES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --max_steps "$MAX_STEPS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --clip_grad_norm "$CLIP_GRAD_NORM" \
  --mixed_precision "$MIXED_PRECISION" \
  --device cuda:0 \
  --val_ratio "$VAL_RATIO" \
  --test_ratio "$TEST_RATIO" \
  --seed "$SEED" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --log_every "$LOG_EVERY" \
  --val_every "$VAL_EVERY" \
  --save_every "$SAVE_EVERY" \
  --channels_last \
  --allow_tf32
