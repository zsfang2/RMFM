from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


DEFAULT_DATA_ROOT = _path_from_env("RMFM_DATA_ROOT", Path("/home/DataDisk/zsfang/dataset"))
DEFAULT_DATASET_ROOT = _path_from_env(
    "RMFM_DATASET_ROOT",
    DEFAULT_DATA_ROOT / "RadioMapSeer",
)
DEFAULT_ARTIFACT_ROOT = _path_from_env("RMFM_ARTIFACT_ROOT", Path("/home/DataDisk/zsfang/rmfm"))
DEFAULT_CHECKPOINT_ROOT = _path_from_env(
    "RMFM_CHECKPOINT_ROOT",
    DEFAULT_ARTIFACT_ROOT / "checkpoints",
)
DEFAULT_RESULT_ROOT = _path_from_env("RMFM_RESULT_ROOT", DEFAULT_ARTIFACT_ROOT / "results")

DEFAULT_UNET_CHECKPOINT_ROOT = DEFAULT_CHECKPOINT_ROOT / "unet"
DEFAULT_TOKEN_UNET_CHECKPOINT_ROOT = DEFAULT_CHECKPOINT_ROOT / "token_unet"
DEFAULT_UNET_RESULT_ROOT = DEFAULT_RESULT_ROOT / "unet"
DEFAULT_TOKEN_UNET_RESULT_ROOT = DEFAULT_RESULT_ROOT / "token_unet"

DEFAULT_UNET_FLOW_CHECKPOINT_DIR = DEFAULT_UNET_CHECKPOINT_ROOT / "radiomapseer_unet_flow_dpm"
DEFAULT_UNET_FLOW_V2_CHECKPOINT_DIR = DEFAULT_UNET_CHECKPOINT_ROOT / "radiomapseer_unet_flow_dpm_v2"
DEFAULT_RASTER_SPARSE_UNET_FLOW_CHECKPOINT_DIR = (
    DEFAULT_UNET_CHECKPOINT_ROOT / "radiomapseer_raster_sparse_unet_flow_dpm_c1"
)
DEFAULT_TOKEN_UNET_FLOW_CHECKPOINT_DIR = DEFAULT_TOKEN_UNET_CHECKPOINT_ROOT / "radiomapseer_token_unet_flow_dpm"
DEFAULT_SOURCE_RESIDUAL_TOKEN_UNET_FLOW_CHECKPOINT_DIR = (
    DEFAULT_TOKEN_UNET_CHECKPOINT_ROOT / "radiomapseer_source_residual_token_unet_flow_dpm"
)
