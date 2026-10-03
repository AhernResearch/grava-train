"""Unified scoring pipeline for eval and rollout.

All eval and rollout scripts use ScoringPipeline.score() as the single
entry point for scoring predictions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grava_train.eval.scored_results import ScoredRecord
from grava_train.eval.scoring.sim_reward_calculator import SimRewardCalculator
from grava_train.trajectory.coordinate_utils import to_nuplan
from grava_train.trajectory.metrics_openloop import compute_ade, compute_fde

_RESERVED_DETAIL_KEYS = {
    "pdm_score",
    "rl_score",
    "cds",
    "has_answer_tag",
    "error",
    "parse_error",
}


class ScoringPipeline:
    """Score predictions and produce standardized ScoredRecord instances."""

    def __init__(self, scorer: SimRewardCalculator):
        self._scorer = scorer

    def _build_record(
        self,
        prediction: str,
        sample_id: str,
        rollout_idx: int,
        reward_result,
        gt_trajectory: Optional[List[List[float]]] = None,
    ) -> ScoredRecord:
        detail = reward_result.detail
        rl_score = detail.get("rl_score")
        cds = detail.get("cds")
        pdm_score = 0.0
        if reward_result.trajectory is not None:
            if self._scorer.scoring_mode == "pdms":
                pdm_score = reward_result.reward
            elif self._scorer.scoring_mode in ("continuous", "discrete"):
                # The RL endpoint exposes its monitoring PDMS under this field.
                pdm_score = float(detail["pdm_score"])
        error = reward_result.detail.get("error")

        sub_metrics: Dict[str, float] = {
            k: v for k, v in reward_result.detail.items()
            if (
                k not in _RESERVED_DETAIL_KEYS
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            )
        }
        extra_metrics: Dict[str, Any] = {
            k: v for k, v in reward_result.detail.items()
            if (
                k not in _RESERVED_DETAIL_KEYS
                and (not isinstance(v, (int, float)) or isinstance(v, bool))
            )
        }

        pred_traj = reward_result.trajectory
        traj_extracted = pred_traj is not None
        ade: Optional[float] = None
        fde: Optional[float] = None

        if pred_traj is not None and gt_trajectory:
            ref = np.array(gt_trajectory)
            n = min(len(pred_traj), len(ref))
            if n > 0:
                pred_xy = to_nuplan(pred_traj[:n, :2], self._scorer.trajectory_frame)
                ade = float(compute_ade(pred_xy, ref[:n, :2]))
                fde = float(compute_fde(pred_xy, ref[:n, :2]))

        return ScoredRecord(
            id=sample_id,
            rollout_idx=rollout_idx,
            prediction=prediction,
            scored=True,
            pdm_score=pdm_score,
            rl_score=float(rl_score) if rl_score is not None else None,
            cds=(
                float(cds) if cds is not None else None
            ),
            error=error,
            parse_error=detail.get("parse_error"),
            traj_extracted=traj_extracted,
            mode=reward_result.mode,
            sub_metrics=sub_metrics,
            extra_metrics=extra_metrics,
            ade=ade,
            fde=fde,
        )

    def score(
        self,
        prediction: str,
        sample_id: str,
        rollout_idx: int = 0,
        gt_trajectory: Optional[List[List[float]]] = None,
        context: Optional[str] = None,
        *,
        scene_info: Tuple[str, str],
    ) -> ScoredRecord:
        """Score a single prediction and return a ScoredRecord."""
        reward_result = self._scorer.compute(
            prediction,
            context=context,
            scene_info=scene_info,
        )
        return self._build_record(
            prediction=prediction,
            sample_id=sample_id,
            rollout_idx=rollout_idx,
            reward_result=reward_result,
            gt_trajectory=gt_trajectory,
        )

    def score_batch(
        self,
        predictions: List[str],
        sample_id: str,
        gt_trajectory: Optional[List[List[float]]] = None,
        rollout_idx_start: int = 0,
        rollout_indices: Optional[Sequence[int]] = None,
        context: Optional[str] = None,
        *,
        scene_info: Tuple[str, str],
    ) -> List[ScoredRecord]:
        """Score multiple predictions for the same case via the scorer batch API."""
        reward_results = self._scorer.compute_batch(
            predictions,
            context=context,
            scene_info=scene_info,
        )
        if rollout_indices is None:
            rollout_indices = [rollout_idx_start + i for i in range(len(predictions))]
        return [
            self._build_record(
                prediction=pred,
                sample_id=sample_id,
                rollout_idx=int(rollout_idx),
                reward_result=reward_result,
                gt_trajectory=gt_trajectory,
            )
            for pred, rollout_idx, reward_result in zip(
                predictions, rollout_indices, reward_results, strict=True
            )
        ]
