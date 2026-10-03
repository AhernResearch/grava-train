"""Single and batched simulation scoring for supported trajectory outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from grava_train.eval.sim_client import SCORE_KEYS, DrivingScoreClient
from grava_train.rewards.cot_parser import detect_reasoning_mode, extract_answer_strict
from grava_train.trajectory.trajectory_parser import extract_trajectory_from_text


@dataclass
class RewardResult:
    """Simulation score and the trajectory used to calculate it."""

    reward: float
    mode: str = "none"
    detail: dict = field(default_factory=dict)
    trajectory: np.ndarray | None = None


class SimRewardCalculator:
    """Calculate reward for trajectories using closed-loop simulation.

    Supported scoring modes:
    - scoring_mode="continuous"/"discrete": RL reward (additive aggregation)
    - scoring_mode="pdms": NavSim PDMS (multiplicative aggregation)
    - scoring_mode="cds": Closed-loop Driving Score
    """

    def __init__(
        self,
        sim_url: str = "http://localhost:8100",
        scoring_mode: str = "continuous",
        dataset: str = "default",
        timeout: int = 30,
        trajectory_format: str = "direct_waypoints",
        trajectory_frame: str = "nuplan",
    ):
        urls = [u.strip() for u in sim_url.split(",") if u.strip()]
        self._clients = [DrivingScoreClient(u, timeout=timeout) for u in urls]
        self._client_idx = 0
        self.scoring_mode = scoring_mode
        self.dataset = dataset
        self.trajectory_frame = trajectory_frame
        self.trajectory_format = trajectory_format

    @property
    def client(self) -> DrivingScoreClient:
        """Round-robin across sim-engine instances."""
        c = self._clients[self._client_idx % len(self._clients)]
        self._client_idx += 1
        return c

    def _score_single(self, trajectory, scene_token, log_name):
        return self.client.score(
            trajectory=trajectory, scene_token=scene_token, log_name=log_name,
            dataset=self.dataset, trajectory_frame=self.trajectory_frame,
            scoring_mode=self.scoring_mode,
        )

    def _score_batch(self, trajectories, scene_token, log_name):
        return self.client.score_batch(
            trajectories=trajectories, scene_token=scene_token, log_name=log_name,
            dataset=self.dataset, trajectory_frame=self.trajectory_frame,
            scoring_mode=self.scoring_mode,
        )

    @property
    def _score_key(self) -> str:
        return SCORE_KEYS[self.scoring_mode]

    def _build_reward_result(self, res, mode, has_answer, trajectory):
        score = float(res[self._score_key])
        detail = {**res, "has_answer_tag": has_answer}
        return RewardResult(
            reward=score, mode=mode, detail=detail, trajectory=trajectory,
        )

    def compute(
        self,
        prediction: str,
        *,
        scene_info: Tuple[str, str],
        context: str | None = None,
    ) -> RewardResult:
        """Parse and score one trajectory for the supplied scene."""
        mode = detect_reasoning_mode(prediction)
        has_answer = extract_answer_strict(prediction) is not None

        log_name, scene_token = scene_info
        trajectory = extract_trajectory_from_text(
            prediction, output_format=self.trajectory_format, context=context,
        )
        if trajectory is None:
            return RewardResult(reward=0.0, mode=mode,
                detail={self._score_key: 0.0, "parse_error": "Failed to extract trajectory",
                        "has_answer_tag": has_answer})

        _, result = self._score_single(trajectory.tolist(), scene_token, log_name)
        return self._build_reward_result(result, mode, has_answer, trajectory)

    def compute_batch(
        self,
        predictions: List[str],
        *,
        scene_info: Tuple[str, str],
        context: str | None = None,
    ) -> List[RewardResult]:
        """Score multiple trajectories for the SAME scene via batch API.

        Only sends valid trajectories to the sim engine; error/extract-failure
        predictions are short-circuited without network calls.
        """
        log_name, scene_token = scene_info

        trajectories = [
            extract_trajectory_from_text(
                prediction, output_format=self.trajectory_format, context=context,
            )
            for prediction in predictions
        ]
        valid_indices = [i for i, trajectory in enumerate(trajectories) if trajectory is not None]
        results = self._score_batch(
            [trajectories[i].tolist() for i in valid_indices], scene_token, log_name,
        ) if valid_indices else []
        scored = dict(zip(valid_indices, results, strict=True))
        out = []
        for i, (prediction, trajectory) in enumerate(zip(predictions, trajectories, strict=True)):
            mode = detect_reasoning_mode(prediction)
            has_answer = extract_answer_strict(prediction) is not None
            if trajectory is None:
                out.append(RewardResult(reward=0.0, mode=mode,
                    detail={self._score_key: 0.0, "parse_error": "Failed to extract trajectory",
                        "has_answer_tag": has_answer}))
            else:
                out.append(self._build_reward_result(
                    scored[i], mode, has_answer, trajectory,
                ))
        return out
