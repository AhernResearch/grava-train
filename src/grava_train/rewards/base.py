"""Shared sample dispatch and decision matching for reward components."""

import json
from abc import abstractmethod
from typing import List

from swift.rewards import ORM

from grava_train.rewards.cot_parser import extract_decision

DECISION_TYPES = {
    "ObjectDecision",
    "ObjectDecisionChain",
    "EgoDecision",
    "EgoDecisionChain",
}

TRAJECTORY_TYPES = {"Trajectory", "TrajectoryChain"}

SOFT_MATCH_PAIRS = {
    frozenset({"Caution", "Follow"}),
}


def _compute_decision_reward(prediction: str, reference: str) -> float:
    """Decision exact/soft match reward."""
    pred = extract_decision(prediction)
    ref = extract_decision(reference)
    if pred is None or ref is None:
        return 0.0
    if pred == ref:
        return 1.0
    if frozenset({pred, ref}) in SOFT_MATCH_PAIRS:
        return 0.5
    return 0.0


def _parse_solution(solution: str) -> dict:
    """Parse the solution field (JSON string from dataset).

    Expected fields:
        qa_type, reference, gt_trajectory, obstacles, log_name, scene_token
    """
    return json.loads(solution)


def _iter_labeled_completions(completions: List[str], solution: List[str]):
    return [(comp, _parse_solution(sol)) for comp, sol in zip(completions, solution, strict=True)]


def _compute_decision_or_trajectory_reward(
    comp: str,
    parsed: dict,
    trajectory_fn,
) -> float:
    qa_type = parsed["qa_type"]
    if qa_type in DECISION_TYPES:
        return _compute_decision_reward(comp, parsed["reference"])
    if qa_type in TRAJECTORY_TYPES:
        return trajectory_fn(comp, parsed)
    raise ValueError(f"Unsupported reward qa_type: {qa_type}")


class TrajectoryReward(ORM):
    """Apply a trajectory reward or the shared decision-label reward per sample."""

    @abstractmethod
    def _trajectory_reward(self, completion: str, sample: dict) -> float:
        """Score one trajectory completion."""
        raise NotImplementedError

    def __call__(
        self, completions: List[str], solution: List[str], **kwargs
    ) -> List[float]:
        return [
            _compute_decision_or_trajectory_reward(comp, parsed, self._trajectory_reward)
            for comp, parsed in _iter_labeled_completions(completions, solution)
        ]
