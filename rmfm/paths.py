from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


DEFAULT_DATA_ROOT = _path_from_env("RMFM_DATA_ROOT", Path("data"))
DEFAULT_DATASET_ROOT = _path_from_env(
    "RMFM_DATASET_ROOT",
    DEFAULT_DATA_ROOT / "RadiomapSeer",
)
DEFAULT_ARTIFACT_ROOT = _path_from_env("RMFM_ARTIFACT_ROOT", Path("outputs") / "rmfm")
DEFAULT_CHECKPOINT_ROOT = _path_from_env(
    "RMFM_CHECKPOINT_ROOT",
    DEFAULT_ARTIFACT_ROOT / "checkpoints",
)
DEFAULT_RESULT_ROOT = _path_from_env("RMFM_RESULT_ROOT", DEFAULT_ARTIFACT_ROOT / "results")

DEFAULT_UNET_FLOW_CHECKPOINT_DIR = DEFAULT_CHECKPOINT_ROOT / "radiomapseer_unet_flow_dpm"
DEFAULT_TOKEN_UNET_FLOW_CHECKPOINT_DIR = DEFAULT_CHECKPOINT_ROOT / "radiomapseer_token_unet_flow_dpm"
