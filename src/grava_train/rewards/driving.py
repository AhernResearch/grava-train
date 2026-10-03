"""Outcome, submetric, and shaping rewards retained for driving experiments."""

import math
import os

import numpy as np

from grava_train.trajectory.metrics_openloop import compute_fde
from grava_train.trajectory.trajectory_parser import extract_trajectory_from_text

from .base import TrajectoryReward
from .simulation import _TRAJECTORY_FORMAT, _get_metric, _get_sim_result

REWARD_MODE = os.environ.get("REWARD_MODE", "continuous")

SAFETY_COLLISION_SIGNAL = os.environ.get("SAFETY_COLLISION_SIGNAL", "overlap")

_BINARY_THRESHOLD = 0.999  # >= this counts as pass; 0.999 rather than 1.0 to handle floating-point imprecision from sim-engine

if SAFETY_COLLISION_SIGNAL not in {"overlap", "penetration_distance"}:
    raise ValueError("SAFETY_COLLISION_SIGNAL must be overlap or penetration_distance")


class ProgressReward(TrajectoryReward):
    """Ego progress reward: how far did the ego travel along the route.

    - Decision types → exact/soft match [0, 0.5, 1.0]
    - Trajectory types → sim-engine ego_progress (EP) [0, 1]
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "ego_progress")


class FDEProgressReward(TrajectoryReward):
    """FDE-based trajectory fidelity reward.

    Adds a smoother open-loop signal without changing the meaning of the
    existing progress reward:
        reward = exp(-fde / tau)

    - Decision types → exact/soft match [0, 0.5, 1.0]
    - Trajectory types → exp(-FDE / tau) in [0, 1]
    """

    def __init__(self, tau: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        gt_arr = np.array(parsed["gt_trajectory"], dtype=np.float64)[:, :2]
        pred_arr = extract_trajectory_from_text(
            comp, output_format=_TRAJECTORY_FORMAT, context=parsed["context"],
        )
        if pred_arr is None:
            return 0.0
        min_len = min(pred_arr.shape[0], gt_arr.shape[0])
        fde = compute_fde(pred_arr[:min_len], gt_arr[:min_len])
        return math.exp(-fde / self.tau)


class PDMSFDEReward(FDEProgressReward):
    """Experimental safety-gated FDE reward replacing the progress term.

    The safety term keeps the existing NC × DAC structure, while the progress
    term is replaced by FDE-based progress:
        reward = (NC × DAC) * exp(-fde / tau)
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        safety = _get_metric(
            comp, parsed, "no_at_fault_collisions"
        ) * _get_metric(comp, parsed, "drivable_area_compliance")
        return safety * super()._trajectory_reward(comp, parsed)


class SafetyReward(TrajectoryReward):
    """Safety reward from raw physics: obstacle_safety + boundary_safety.

    Boundary keeps sqrt(area-ratio) shaping. Collision signal is configurable:
    - overlap: collision penalty from sqrt(max_collision_overlap)
    - penetration_distance: collision penalty from max_collision_penetration_distance

    discrete mode: (NC_bin + DAC_bin + TTC_bin) / 3, binary {0,1} per metric
    """

    # Normalization constants (tunable without sim-engine redeploy)
    OBSTACLE_FALLBACK_DIST = 5.0  # meters — fallback when no obstacles within 5m
    BOUNDARY_SAFE_DIST = 2.0  # meters — boundary distance giving full score
    COLLISION_PENETRATION_SCALE = 2.0  # meters — penetration distance giving full penalty
    COLLISION_EPS = 1e-4

    def _collision_penalty(self, comp: str, parsed: dict) -> float:
        if SAFETY_COLLISION_SIGNAL == "penetration_distance":
            penetration_dist = _get_metric(
                comp,
                parsed,
                "max_collision_penetration_distance",
            )
            if penetration_dist > self.COLLISION_EPS:
                return -min(penetration_dist / self.COLLISION_PENETRATION_SCALE, 1.0)
            return 0.0

        collision_overlap = _get_metric(comp, parsed, "max_collision_overlap")
        if collision_overlap > self.COLLISION_EPS:
            return -math.sqrt(collision_overlap)
        return 0.0

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        if REWARD_MODE == "discrete":
            nc = _get_metric(comp, parsed, "no_at_fault_collisions")
            dac = _get_metric(comp, parsed, "drivable_area_compliance")
            ttc = _get_metric(comp, parsed, "time_to_collision")
            nc_bin = 1.0 if nc >= _BINARY_THRESHOLD else 0.0
            dac_bin = 1.0 if dac >= _BINARY_THRESHOLD else 0.0
            ttc_bin = 1.0 if ttc >= _BINARY_THRESHOLD else 0.0
            return (nc_bin + dac_bin + ttc_bin) / 3.0

        obstacle_dist = _get_metric(comp, parsed, "min_obstacle_distance")
        out_of_road = 1.0 - _get_metric(comp, parsed, "drivable_area_compliance")
        boundary_dist = _get_metric(comp, parsed, "min_boundary_distance")
        mean_obs_5m = _get_metric(
            comp,
            parsed,
            "mean_obstacle_distance_5m",
        )
        half_lane_width = _get_metric(
            comp,
            parsed,
            "half_lane_width",
        )

        obstacle = self._collision_penalty(comp, parsed)
        if obstacle == 0.0:
            denom = mean_obs_5m if mean_obs_5m > 0 else self.OBSTACLE_FALLBACK_DIST
            obstacle = min(obstacle_dist / denom, 1.0)

        if out_of_road > self.COLLISION_EPS:
            boundary = -math.sqrt(out_of_road)
        else:
            effective_hlw = min(max(half_lane_width, self.COLLISION_EPS), self.BOUNDARY_SAFE_DIST)
            boundary = min(boundary_dist / effective_hlw, 1.0)

        return 0.5 * obstacle + 0.5 * boundary


class ComfortReward(TrajectoryReward):
    """Comfort reward: trajectory smoothness and comfort.

    - Decision types → exact/soft match [0, 0.5, 1.0]
    - Trajectory types → sim-engine history_comfort (HC) [0, 1]
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "history_comfort")


class TTCReward(TrajectoryReward):
    """Time-to-collision reward.

    - Decision types → exact/soft match [0, 0.5, 1.0]
    - Trajectory types → sim-engine time_to_collision [0, 1]
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "time_to_collision")


class PDMSReward(TrajectoryReward):
    """NAVSIM v1 PDMS outcome reward from the simulation service."""

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "pdm_score", scoring_mode="pdms")


class CDSReward(TrajectoryReward):
    """Closed-loop Driving Score (CDS), returned by the scoring engine."""

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "cds", scoring_mode="cds")


class SafetyGateReward(TrajectoryReward):
    """Discrete NC × DAC safety gate for trajectory rewards."""

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "safety_gate", scoring_mode="discrete")


class GatedProgressReward(TrajectoryReward):
    """Progress in meters × NC × DAC × TTC × HC.

    - Trajectory types → centerline progress (meters), 0 if safety_gate=0
    - Decision types → exact/soft match [0, 0.5, 1.0]
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "gated_progress", scoring_mode="discrete")


class PDMSMonitorReward(TrajectoryReward):
    """PDMS score for monitoring only (weight=0, no gradient).

    Reuses the official PDMS cache when PDMSReward is also enabled.
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        return _get_metric(comp, parsed, "pdm_score", scoring_mode="pdms")


class NCReward(TrajectoryReward):
    """No-collision binary reward: 1.0 if no at-fault collision, 0.0 otherwise.

    Uses sim-engine continuous ``no_at_fault_collisions`` metric.
    Binary signal — collision is non-negotiable.
    """

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        nc = _get_metric(comp, parsed, "no_at_fault_collisions")
        return 1.0 if nc >= _BINARY_THRESHOLD else 0.0


class BoundaryReward(TrajectoryReward):
    """Boundary distance reward: linear ramp from 0 (on edge) to 1 (>= 1m clearance).

    Uses sim-engine ``min_boundary_distance`` (signed: positive=inside, negative=outside).
    Saturates at SATURATION_DIST — beyond this gives full score.

    GDPO group normalization handles narrow-road scenes automatically:
    all rollouts get similar low scores → std≈0 → advantage≈0 → no gradient.
    """

    SATURATION_DIST = 0.5  # meters: >= 0.5m from boundary = full score

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        boundary_dist = _get_metric(comp, parsed, "min_boundary_distance")
        return min(max(boundary_dist, 0.0), self.SATURATION_DIST) / self.SATURATION_DIST


class BoundaryMarginReward(TrajectoryReward):
    """Dense signed boundary-margin reward for executable planner RL.

    Official PDMS/DAC already gates out-of-road failures, but it is sparse near
    the boundary. This reward gives a small shaping signal before DAC collapses:
        d <= -0.2m -> 0
        d >=  0.5m -> 1
    where d is sim-engine ``min_boundary_distance``.
    """

    NEG_MARGIN = -0.2
    SAFE_MARGIN = 0.5

    def _trajectory_reward(self, comp: str, parsed: dict) -> float:
        result = _get_sim_result(
            comp,
            parsed["log_name"],
            parsed["scene_token"],
            context=parsed["context"],
        )
        if result.get("_extraction_failed"):
            return 0.0
        boundary_dist = float(result["min_boundary_distance"])
        span = self.SAFE_MARGIN - self.NEG_MARGIN
        return min(max((boundary_dist - self.NEG_MARGIN) / span, 0.0), 1.0)
