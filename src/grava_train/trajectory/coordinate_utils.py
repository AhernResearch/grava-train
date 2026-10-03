"""GRAVA uses x-right/y-forward; nuPlan uses x-forward/y-left, in meters."""

import numpy as np


def to_nuplan(trajectory: np.ndarray, frame: str) -> np.ndarray:
    """Convert XY to nuPlan; pass additional columns through as the scoring API expects."""
    points = np.asarray(trajectory, dtype=np.float64)
    if frame == "nuplan":
        return points.copy()
    if frame != "grava":
        raise ValueError(f"Unknown trajectory frame: {frame}")
    converted = points.copy()
    converted[:, 0] = points[:, 1]
    converted[:, 1] = -points[:, 0]
    return converted


def nuplan_to_grava(trajectory: np.ndarray) -> np.ndarray:
    """Convert XY to GRAVA, preserving any additional columns."""
    points = np.asarray(trajectory, dtype=np.float64)
    converted = points.copy()
    converted[:, 0] = -points[:, 1]
    converted[:, 1] = points[:, 0]
    return converted
