# Copy this file to configs/local_paths.sh and edit it for each machine.
# configs/local_paths.sh is ignored by git.

DATASET_ROOT="/path/to/RadiomapSeer"
OUTPUT_DIR="/path/to/rmfm/checkpoints/radiomapseer_token_unet_flow_irt4"

GPU_ID=0
BATCH_SIZE=16
MAX_STEPS=100000
MAX_TRAIN_SAMPLES=-1
MAX_VAL_SAMPLES=512
