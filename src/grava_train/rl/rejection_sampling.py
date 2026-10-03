"""Choose complete rollout answers for SFT, with supervised-reference fallback."""

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from grava_train.eval.scored_results import ScoredRecord
from grava_train.eval.sim_client import SCORE_KEYS
from grava_train.trajectory.trajectory_parser import extract_trajectory_from_text
from grava_train.utils.jsonl_utils import iter_jsonl

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _has_reasoning(text: str) -> bool:
    match = _THINK.search(text)
    return match is not None and bool(match.group(1).strip())


def select_best_rollouts(
    paths: Iterable[Path], num_rollouts: int, scoring_mode: str = "pdms",
) -> dict[str, ScoredRecord | None]:
    """Keep one valid answer per complete group; ties use the lowest rollout index.

    Only the best prediction text is retained in memory. Scoring failures and
    incomplete groups are execution errors, not reasons to use reference labels.
    """
    metric = SCORE_KEYS[scoring_mode]
    indices: dict[str, set[int]] = defaultdict(set)
    best: dict[str, ScoredRecord | None] = {}
    for path in paths:
        for row in iter_jsonl(path):
            record = ScoredRecord.from_dict(row)
            sid, index = record.id, record.rollout_idx
            if not record.scored or record.error:
                raise ValueError(
                    f"Rollout is not successfully scored: {sid}:{index}: {record.error}"
                )
            score = float(getattr(record, metric))
            if not math.isfinite(score):
                raise ValueError(f"Non-finite {metric}: {sid}:{index}")
            if index in indices[sid]:
                raise ValueError(f"Duplicate rollout: {sid}:{index}")
            indices[sid].add(index)
            best.setdefault(sid, None)
            if (record.parse_error or not record.traj_extracted
                    or not _has_reasoning(record.prediction)):
                continue
            if scoring_mode == "cds" and record.extra_metrics["sample_valid"] is not True:
                continue
            previous = best[sid]
            if previous is None or (score, -index) > (
                float(getattr(previous, metric)), -previous.rollout_idx,
            ):
                best[sid] = record

    expected = set(range(num_rollouts))
    for sid, seen in indices.items():
        if seen != expected:
            raise ValueError(f"Incomplete rollout group: {sid}: {sorted(seen)}")
    return best


def build_sft_sample(
    sample: dict[str, Any], best: ScoredRecord | None, *,
    scoring_mode: str = "pdms", trajectory_decoder: str = "executable_planner",
) -> tuple[dict[str, Any], str]:
    """Append the selected answer verbatim, or use solution.reference at zero reward."""
    use_rollout = best is not None and float(getattr(best, SCORE_KEYS[scoring_mode])) > 0
    origin = "rollout" if use_rollout else "reference"
    answer = best.prediction if use_rollout else json.loads(sample["solution"])["reference"]
    context = sample["messages"][0]["content"]
    if not _has_reasoning(answer) or extract_trajectory_from_text(
        answer, output_format=trajectory_decoder, context=context,
    ) is None:
        raise ValueError(f"Invalid {origin} target for {sample['id']}: need think and trajectory")
    output = {key: value for key, value in sample.items() if key != "solution"}
    output["messages"] = [*sample["messages"], {"role": "assistant", "content": answer}]
    return output, origin
