"""Select the checkpoint returned by a completed training stage."""

from typing import Any

from grava_train.utils.checkpoint_utils import validate_checkpoint


def training_checkpoint(result: dict[str, Any], *, required: bool = True) -> str | None:
    """Use the checkpoint reported by this sft_main call, never an older run."""
    path = result["last_model_checkpoint"]
    if not path:
        if required:
            raise RuntimeError(
                "SFT did not save a checkpoint; enable --save_strategy steps or epoch"
            )
        return None
    return validate_checkpoint(path)
