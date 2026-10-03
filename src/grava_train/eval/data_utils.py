"""Evaluation metadata from released JSONL records."""

import json

import numpy as np

from grava_train.data.dataset_utils import read_jsonl
from grava_train.trajectory.coordinate_utils import to_nuplan


def load_trajectory_samples(
    path: str, max_samples: int | None = None, data_root: str | None = None,
) -> list[dict]:
    """Read trajectory samples with image paths resolved for this machine."""
    samples = []
    for sample in read_jsonl(path, data_root):
        if sample["qa_type"] != "Trajectory":
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def parse_solution(sample: dict) -> dict:
    return json.loads(sample["solution"])


def scene_info_from_sample(sample: dict) -> tuple[str, str]:
    solution = parse_solution(sample)
    return solution["log_name"], solution["scene_token"]


def extract_gt_trajectory(sample: dict) -> list[list[float]] | None:
    """Return original GT in nuPlan coordinates; SFT answers are not ground truth.

    Released files explicitly declare the frame
    and whether their trajectory is original GT or a decoded training reference.
    """
    if "solution" not in sample:
        return None
    solution = parse_solution(sample)
    if solution.get("gt_trajectory_source") == "decoded_reference":
        return None
    trajectory = solution.get("gt_trajectory")
    if not trajectory:
        return None
    return to_nuplan(np.asarray(trajectory)[:, :2], solution["gt_trajectory_frame"]).tolist()
