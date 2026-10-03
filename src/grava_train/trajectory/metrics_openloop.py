"""Open-loop trajectory metrics on decoded waypoint arrays."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _align_trajectories(pred: np.ndarray, gt: np.ndarray):
    """Truncate pred and gt to the same length (the shorter one)."""
    n = min(len(pred), len(gt))
    return pred[:n], gt[:n]


def compute_ade(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    return_per_step: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """Mean Euclidean XY error in meters for aligned (T, 2|3) arrays; optionally return per-step errors."""
    assert pred_trajectory.shape == gt_trajectory.shape, \
        f"Shape mismatch: {pred_trajectory.shape} vs {gt_trajectory.shape}"
    assert pred_trajectory.ndim >= 2, \
        f"Expected 2D array, got {pred_trajectory.ndim}D"

    pred_xy = pred_trajectory[:, :2]
    gt_xy = gt_trajectory[:, :2]

    distances = np.linalg.norm(pred_xy - gt_xy, axis=1)

    ade = float(np.mean(distances))

    if return_per_step:
        return ade, distances
    return ade


def compute_fde(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> float:
    """Final Euclidean XY error in meters for trajectories of equal length."""
    assert pred_trajectory.shape[0] == gt_trajectory.shape[0], \
        f"Length mismatch: {pred_trajectory.shape[0]} vs {gt_trajectory.shape[0]}"

    pred_final = pred_trajectory[-1, :2]
    gt_final = gt_trajectory[-1, :2]

    fde = float(np.linalg.norm(pred_final - gt_final))

    return fde


def compute_heading_error(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray
) -> Tuple[float, float]:
    """Mean and final absolute heading error in radians, wrapped to [-pi, pi]."""
    assert pred_trajectory.shape[1] >= 3 and gt_trajectory.shape[1] >= 3, \
        "Heading required for heading error computation"

    pred_heading = pred_trajectory[:, 2]
    gt_heading = gt_trajectory[:, 2]

    # Compute heading differences (handle angle wrapping)
    heading_diffs = np.arctan2(
        np.sin(pred_heading - gt_heading),
        np.cos(pred_heading - gt_heading)
    )

    ahe = float(np.mean(np.abs(heading_diffs)))
    fhe = float(np.abs(heading_diffs[-1]))

    return ahe, fhe


def compute_miss_rate(
    fde: float,
    threshold: float = 2.0
) -> float:
    """Return 1.0 when FDE exceeds the threshold in meters, otherwise 0.0."""
    return 1.0 if fde > threshold else 0.0


def compute_multi_horizon_metrics(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    timesteps_per_second: float = 2.0,  # NavSim uses 0.5s intervals
    horizons: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute ADE/FDE at horizons in seconds; default sampling is 2 Hz."""
    horizons = horizons or [3.0, 5.0, 8.0]
    results = {}

    for horizon in horizons:
        # Calculate number of timesteps for this horizon
        n_steps = int(horizon * timesteps_per_second)

        if n_steps > len(pred_trajectory):
            # Skip if trajectory is shorter than horizon
            continue

        # Slice trajectories
        pred_slice = pred_trajectory[:n_steps]
        gt_slice = gt_trajectory[:n_steps]

        # Compute metrics
        ade = compute_ade(pred_slice, gt_slice)
        fde = compute_fde(pred_slice, gt_slice)

        results[f'ade_{int(horizon)}s'] = ade
        results[f'fde_{int(horizon)}s'] = fde

    return results
