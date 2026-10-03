"""HTTP scoring in nuPlan coordinates with public PDMS, CDS and RL metric names."""

import json
from urllib.request import Request, urlopen

import numpy as np

from grava_train.trajectory.coordinate_utils import nuplan_to_grava, to_nuplan

SCORE_KEYS = {"pdms": "pdm_score", "cds": "cds", "continuous": "rl_score", "discrete": "rl_score"}


class DrivingScoreClient:
    """Score decoded trajectories; HTTP, JSON and service errors propagate."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _read_score(response: dict, scoring_mode: str) -> dict:
        if response.get("error"):
            raise RuntimeError(f"sim-engine {scoring_mode}: {response['error']}")
        result = dict(response)
        key = SCORE_KEYS[scoring_mode]
        result[key] = float(result[key])
        # Auxiliary route geometry is returned to callers in GRAVA coordinates.
        details = dict(result.get("details", {}))
        points = details.get("local_centerline_points")
        if scoring_mode != "cds" and points:
            details["local_centerline_points"] = nuplan_to_grava(np.asarray(points)[:, :2]).tolist()
            result["details"] = details
        return result

    def score(
        self, trajectory: list[list[float]], scene_token: str, log_name: str, dataset: str,
        trajectory_frame: str = "grava", scoring_mode: str = "pdms",
        *, config_overrides: dict[str, float] | None = None, include_details: bool = False,
    ) -> tuple[float, dict]:
        payload = {
            "trajectory": to_nuplan(trajectory, trajectory_frame).tolist(),
            "scene_token": scene_token, "log_name": log_name, "dataset": dataset,
            "scoring_mode": scoring_mode, "config_overrides": config_overrides or {},
            "include_details": include_details,
        }
        result = self._read_score(self._post("/v1/score", payload), scoring_mode)
        return result[SCORE_KEYS[scoring_mode]], result

    def score_batch(
        self, trajectories: list[list[list[float]]], scene_token: str, log_name: str, dataset: str,
        trajectory_frame: str = "grava", scoring_mode: str = "pdms",
        *, config_overrides: dict[str, float] | None = None, include_details: bool = False,
    ) -> list[dict]:
        payload = {
            "trajectories": [to_nuplan(t, trajectory_frame).tolist() for t in trajectories],
            "scene_token": scene_token, "log_name": log_name, "dataset": dataset,
            "scoring_mode": scoring_mode, "config_overrides": config_overrides or {},
            "include_details": include_details,
        }
        response = self._post("/v1/score/batch", payload)
        return [self._read_score(result, scoring_mode)
                for _, result in zip(trajectories, response["results"], strict=True)]
