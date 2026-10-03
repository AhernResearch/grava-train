"""Parse a final trajectory answer in the explicitly selected output format."""

import json
import re

import numpy as np

from grava_train.trajectory.trajectory_codec import ExecutablePlannerCodec, TrajectoryFormatError

_PLANNER_CODEC = ExecutablePlannerCodec()
_EGO_SPEED_RE = re.compile(r"Speed:\s*([-+]?\d+(?:\.\d+)?)\s*m/s")
_EGO_ACCEL_RE = re.compile(r"Acceleration:\s*([-+]?\d+(?:\.\d+)?)\s*m/s")


def extract_ego_motion_from_text(text: str) -> tuple[float, float]:
    """Read the observed motion from a prepared dataset's rendered prompt."""
    return (
        float(_EGO_SPEED_RE.search(text).group(1)),
        float(_EGO_ACCEL_RE.search(text).group(1)),
    )


def extract_trajectory_from_text(
    text: str,
    *,
    output_format: str,
    context: str | None = None,
    ego_speed: float | None = None,
    ego_accel: float = 0.0,
) -> np.ndarray | None:
    """Return eight waypoints, or None for malformed model output.

    Planner decoding requires the prompt's ego state or an explicit speed.
    Caller/configuration errors and numerical decoding failures propagate.
    """
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    content = answer.group(1) if answer else text.split("</think>", 1)[-1]
    if output_format == "executable_planner":
        if ego_speed is None:
            if context is None:
                raise ValueError("Executable Planner decoding requires context or ego_speed")
            ego_speed, ego_accel = extract_ego_motion_from_text(context)
        try:
            return _PLANNER_CODEC.decode(content, ego_speed=ego_speed, ego_accel=ego_accel)
        except (TrajectoryFormatError, json.JSONDecodeError):
            return None
    if output_format != "direct_waypoints":
        raise ValueError(f"Unsupported trajectory format: {output_format}")
    if re.search(r'"planner"\s*:', content):
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[\s*\[", content):
        try:
            points, _ = decoder.raw_decode(content[match.start():])
            trajectory = np.asarray(points, dtype=np.float64)
        except (ValueError, TypeError, OverflowError):
            continue
        if (trajectory.ndim == 2 and trajectory.shape[0] == 8
                and trajectory.shape[1] in (2, 3) and np.isfinite(trajectory).all()):
            return trajectory
    return None
