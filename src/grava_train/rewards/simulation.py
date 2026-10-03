"""Shared sim-engine access, failure handling, and bounded rollout score cache."""

import hashlib
import json
import logging
import os
from typing import Any, Dict

from grava_train.eval.sim_client import DrivingScoreClient
from grava_train.trajectory.trajectory_parser import extract_trajectory_from_text

logger = logging.getLogger(__name__)

_TRAJECTORY_FORMAT = os.environ.get("TRAJECTORY_DECODER", "direct_waypoints")

_sim_client = None

_DEBUG_ROLLOUT_FILE = os.environ.get("DEBUG_ROLLOUT_FILE", "")

_debug_rollout_count = 0

_DEBUG_ROLLOUT_MAX = 20  # save first N calls

_SIM_CACHE_MAX_SIZE = 2048  # Sufficient for one GRPO step (batch×generations×2_modes)

_sim_result_cache: dict[tuple, Dict[str, Any]] = {}

_SIM_DATASET = os.environ.get("SIM_ENGINE_DATASET", "navtest")

_SIM_TRAJECTORY_FRAME = os.environ.get("SIM_ENGINE_TRAJECTORY_FRAME", "grava")


def _get_sim_client():
    """Shared driving-score client."""
    global _sim_client
    if _sim_client is None:

        url = os.environ.get("SIM_ENGINE_URL", "http://localhost:8100")
        _sim_client = DrivingScoreClient(url, timeout=30)
        logger.info("DrivingScoreClient initialized: %s", url)
    return _sim_client


def _debug_save_rollout(prediction: str, log_name: str, scene_token: str, traj, result: dict):
    """Save rollout sample to debug file. Failures always saved, successes capped."""
    global _debug_rollout_count
    if not _DEBUG_ROLLOUT_FILE:
        return
    is_failure = traj is None
    if not is_failure and _debug_rollout_count >= _DEBUG_ROLLOUT_MAX:
        return
    _debug_rollout_count += 1
    entry = {
        "idx": _debug_rollout_count,
        "log_name": log_name,
        "scene_token": scene_token,
        "prediction": prediction[:2000],
        "traj_extracted": traj is not None,
        "traj_first": traj[0] if traj else None,
        "result": result,
        "failure": is_failure,
    }
    with open(_DEBUG_ROLLOUT_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _get_sim_result(
    prediction: str,
    log_name: str,
    scene_token: str,
    scoring_mode: str = "continuous",
    context: str | None = None,
) -> Dict[str, Any]:
    """Score trajectory via sim-engine and return full result dict, with per-generation cache.

    scoring_mode: "continuous", "discrete", "pdms", or "cds"
    Invalid model output receives the default score; scoring failures propagate.
    """
    _maybe_evict_cache()
    key = (hashlib.md5(prediction.encode()).hexdigest(), log_name, scene_token,
           scoring_mode, _TRAJECTORY_FORMAT, context)
    cached = _sim_result_cache.get(key)
    if cached is not None:
        return cached

    trajectory = extract_trajectory_from_text(
        prediction,
        output_format=_TRAJECTORY_FORMAT,
        context=context,
    )
    if trajectory is None:
        logger.warning(
            "Trajectory extraction failed. decoder=%s, prediction[:200]=%s",
            _TRAJECTORY_FORMAT,
            prediction[:200],
        )
        result = {"_extraction_failed": True}
        _debug_save_rollout(prediction, log_name, scene_token, None, result)
        _sim_result_cache[key] = result
        return result

    traj = trajectory.tolist()
    client = _get_sim_client()
    _, result = client.score(
        trajectory=traj, scene_token=scene_token, log_name=log_name,
        dataset=_SIM_DATASET, trajectory_frame=_SIM_TRAJECTORY_FRAME,
        scoring_mode=scoring_mode,
    )

    _debug_save_rollout(prediction, log_name, scene_token, traj, result)
    _sim_result_cache[key] = result
    return result


def _clear_sim_cache():
    """Clear sim result cache between GRPO generations."""
    _sim_result_cache.clear()


def _maybe_evict_cache():
    if len(_sim_result_cache) > _SIM_CACHE_MAX_SIZE:
        _sim_result_cache.clear()


def _get_metric(
    comp: str,
    parsed: dict,
    metric_key: str,
    scoring_mode: str = "continuous",
) -> float:
    """Read a required metric; only invalid generated trajectories receive zero."""
    result = _get_sim_result(
        comp, parsed["log_name"], parsed["scene_token"],
        scoring_mode=scoring_mode, context=parsed["context"],
    )
    if result.get("_extraction_failed"):
        return 0.0
    return float(result[metric_key])
