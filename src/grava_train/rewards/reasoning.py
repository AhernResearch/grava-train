"""Reasoning-mode bonuses and trajectory format validation."""

import re
from typing import List

from grava_train.trajectory.trajectory_parser import extract_trajectory_from_text

from .base import (
    DECISION_TYPES,
    ORM,
    TRAJECTORY_TYPES,
    _compute_decision_reward,
    _iter_labeled_completions,
    _parse_solution,
)
from .simulation import _TRAJECTORY_FORMAT, _get_metric

MODE_DECISION_THRESHOLD = 1.0

PDMS_CORRECT_THRESHOLD = 0.9

CHAIN_LEN_THRESHOLD = 600

CHAIN_CORRECT_MAX = 1.0

THINK_CORRECT_REWARD = 0.3

THINK_WRONG_REWARD = 0.0


def _detect_mode(text: str) -> str:
    """Detect reasoning mode from output: 'chain', 'think', or 'none'.

    In new Qwen3.5-native format, all reasoning is in <think>.
    Chain DSL content inside <think> is detected by content pattern.
    Legacy <chain> tag still supported for backward compat.
    """
    has_chain = bool(re.search(r"<chain>.*?</chain>", text, re.DOTALL | re.IGNORECASE))
    has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE))
    if has_think:
        # Check if think content looks like chain DSL
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if m and re.search(r"scene_context\.|ego_decision\(|\.decision\(", m.group(1)):
            return "chain"
        return "think"
    if has_chain:
        return "chain"
    return "none"


class ModeAppropriatenessReward(ORM):
    """Mode appropriateness reward: correct short chains receive a larger bonus.

    Chain mode scoring:
      - correct + short → efficiency × 1.0  (ideal: fast reasoning, correct result)
      - correct + long  → efficiency → ~0   (shell game: long chain ≈ think in chain tag)
      - wrong           → 0.0               (no correctness bonus)
    Think mode scoring:
      - correct → 0.3  (safe baseline: slow but correct)
      - wrong   → 0.0  (at least tried, no penalty)

    Efficiency decay: efficiency = max(0, 1 - chain_len / CHAIN_LEN_THRESHOLD)
    "Correct" uses pdm_score > 0.9 (trajectory) or exact match (decision).
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        for comp, parsed in _iter_labeled_completions(completions, solution):
            qa_type = parsed["qa_type"]
            if qa_type not in DECISION_TYPES | TRAJECTORY_TYPES:
                raise ValueError(f"Unsupported reward qa_type: {qa_type}")
            mode = _detect_mode(comp)
            if mode == "none":
                rewards.append(0.0)
                continue

            if qa_type in DECISION_TYPES:
                correct = _compute_decision_reward(comp, parsed["reference"]) >= MODE_DECISION_THRESHOLD
            else:
                correct = _get_metric(
                    comp, parsed, "pdm_score", scoring_mode="discrete",
                ) > PDMS_CORRECT_THRESHOLD

            if mode == "chain":
                if not correct:
                    rewards.append(0.0)  # no penalty, just no bonus
                else:
                    chain_content = re.search(r"<chain>(.*?)</chain>", comp, re.DOTALL)
                    chain_len = len(chain_content.group(1)) if chain_content else 0
                    efficiency = max(0.0, 1.0 - chain_len / CHAIN_LEN_THRESHOLD)
                    rewards.append(efficiency * CHAIN_CORRECT_MAX)
            else:
                rewards.append(THINK_CORRECT_REWARD if correct else THINK_WRONG_REWARD)
        return rewards


class GRAFormatReward(ORM):
    """Trajectory extraction check — uses the same extraction logic as PDMS reward.

    1.0 if extract_trajectory_from_text succeeds in the configured format, 0.0 otherwise.
    This ensures format reward is consistent with actual reward scoring.
    """

    def __call__(self, completions: List[str], solution: List[str] = None, **kwargs) -> List[float]:
        rewards = []
        parsed_solutions = (
            [_parse_solution(sol) for sol in solution]
            if solution is not None
            else [None for _ in completions]
        )
        for comp, parsed in zip(completions, parsed_solutions, strict=True):
            traj = extract_trajectory_from_text(
                comp,
                output_format=_TRAJECTORY_FORMAT,
                context=parsed["context"] if parsed is not None else None,
            )
            rewards.append(1.0 if traj is not None else 0.0)
        return rewards
