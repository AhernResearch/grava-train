"""Aggregate scored rollouts and select canonical training records by ID."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CaseScores:
    id: str
    scores: tuple[float, ...]
    errors: int = 0

    def statistics(self) -> dict:
        scores = np.asarray(self.scores)
        return {
            'id': self.id, 'n': len(scores), 'errors': self.errors,
            'min': float(scores.min()), 'max': float(scores.max()),
            'mean': float(scores.mean()), 'median': float(np.median(scores)),
            'std': float(scores.std(ddof=0)), 'range': float(np.ptp(scores)),
            'scores': list(self.scores),
        }


@dataclass(frozen=True)
class GreedyRule:
    threshold: float = 0.9

    def accepts(self, stats: dict) -> bool:
        return stats['max'] < self.threshold


@dataclass(frozen=True)
class HighVarianceRule:
    min_score_lt: float = 0.8
    max_score_ge: float = 0.9
    std_ge: float = 0.1

    def accepts(self, stats: dict) -> bool:
        return (stats['min'] < self.min_score_lt
                and stats['max'] >= self.max_score_ge
                and stats['std'] >= self.std_ge)


def aggregate_rollouts(
    paths: Iterable[Path], expected_rollouts: int, metric: str = 'pdm_score',
) -> dict[str, CaseScores]:
    """Read complete groups, rejecting duplicate indices and unfinished scoring.

    Records must explicitly declare their scoring status.
    Invalid generated trajectories keep their recorded
    zero reward; scoring-service failures are not converted into training signals.
    """
    grouped = defaultdict(dict)
    errors = defaultdict(int)
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                sid, index = row['id'], row['rollout_idx']
                if not row['scored']:
                    raise ValueError(f'Unscored rollout: {sid}:{index}')
                error = row.get('error')
                if error:
                    raise ValueError(f'Failed scoring: {sid}:{index}: {error}')
                score = float(row[metric])
                if not math.isfinite(score):
                    raise ValueError(f'Non-finite score: {sid}:{index}')
                if index in grouped[sid]:
                    raise ValueError(f'Duplicate rollout: {sid}:{index}')
                grouped[sid][index] = score
                errors[sid] += bool(row.get('parse_error'))
    result = {}
    indices = set(range(expected_rollouts))
    for sid, scores in grouped.items():
        if scores.keys() != indices:
            raise ValueError(f'Incomplete rollout group: {sid}: {sorted(scores)}')
        result[sid] = CaseScores(sid, tuple(scores[i] for i in range(expected_rollouts)), errors[sid])
    return result


def export_selected_rows(source: Path, output: Path, selected: set[str]) -> int:
    """Copy selected canonical rows byte-for-byte, preserving order and labels."""
    if source.resolve() == output.resolve():
        raise ValueError('Selection output must differ from the source dataset')
    seen = set()
    found = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + '.tmp')
    with source.open('rb') as stream, pending.open('wb') as target:
        for line in stream:
            row = json.loads(line)
            sid = row['id']
            if sid in seen:
                raise ValueError(f'Duplicate canonical ID: {sid}')
            seen.add(sid)
            if sid in selected:
                assert row['messages'] and row['images'], sid
                solution = json.loads(row['solution'])
                assert solution['log_name'] and solution['scene_token'], sid
                target.write(line)
                found.add(sid)
    if found != selected:
        raise ValueError(f'Selected IDs absent from canonical pool: {sorted(selected - found)[:5]}')
    pending.replace(output)
    return len(found)
