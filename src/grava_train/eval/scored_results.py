"""Standard scored record schema for eval and rollout outputs.

Trajectory evaluation (eval_trajectory.py) and rollout scripts (rollout.py) produce
ScoredRecord instances.

This is the single source of truth for the scored output format.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ScoredRecord:
    """A single scored prediction (one rollout of one case)."""

    id: str
    rollout_idx: int  # 0 for greedy eval, 0..K-1 for rollout
    prediction: str
    pdm_score: float
    scored: bool
    rl_score: Optional[float] = None
    cds: Optional[float] = None
    error: Optional[str] = None
    parse_error: Optional[str] = None
    generation_attempts: int = 0  # Zero when the generation history is unknown.
    traj_extracted: bool = False
    mode: str = "none"  # "think" / "chain" / "none"
    sub_metrics: Dict[str, float] = field(default_factory=dict)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    ade: Optional[float] = None
    fde: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScoredRecord:
        """Create from dict. Expects standard field names — no fallbacks."""
        return cls(
            id=d["id"],
            rollout_idx=d["rollout_idx"],
            prediction=d["prediction"],
            scored=d["scored"],
            pdm_score=float(d["pdm_score"]),
            rl_score=float(d["rl_score"]) if d.get("rl_score") is not None else None,
            cds=(
                float(d["cds"]) if d.get("cds") is not None else None
            ),
            error=d.get("error"),
            parse_error=d.get("parse_error"),
            generation_attempts=d.get("generation_attempts", 0),
            traj_extracted=d["traj_extracted"],
            mode=d["mode"],
            sub_metrics=d.get("sub_metrics", {}),
            extra_metrics=d.get("extra_metrics", {}),
            ade=d.get("ade"),
            fde=d.get("fde"),
        )


def load_scored_records(path: str) -> List[ScoredRecord]:
    """Load scored records from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(ScoredRecord.from_dict(json.loads(line)))
    return records


def save_scored_records(records: List[ScoredRecord], path: str) -> None:
    """Save scored records to JSONL."""
    with open(path, "w") as f:
        for r in records:
            f.write(r.to_json() + "\n")


class EvalReportBuilder:
    """Aggregate scored evaluation results into structured reports."""

    @staticmethod
    def _summary(vals: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(vals)), 4) if vals else 0,
            "median": round(float(np.median(vals)), 4) if vals else 0,
            "std": round(float(np.std(vals)), 4) if vals else 0,
            "p25": round(float(np.percentile(vals, 25)), 4) if vals else 0,
            "p75": round(float(np.percentile(vals, 75)), 4) if vals else 0,
        }

    @staticmethod
    def build_report(
        scored: Union[List[ScoredRecord], List[Dict]],
        model: str,
        dataset: str,
        score_key: str = "pdm_score",
        report_key: str = "pdms",
        require_sample_valid: bool = False,
    ) -> Dict[str, Any]:
        """Build an aggregate score / open-loop report.

        Accepts either List[ScoredRecord] or List[Dict] with standard fields.
        """
        # Normalize to dicts
        entries = [
            s.to_dict() if isinstance(s, ScoredRecord) else s
            for s in scored
        ]

        score_vals = [
            float(s[score_key]) for s in entries
            if not s.get("error")
            and (
                not require_sample_valid
                or bool(s.get("parse_error"))
                or (s.get("extra_metrics") or {}).get("sample_valid") is True
            )
        ]
        errors = [s for s in entries if s.get("error")]
        extracted = sum(1 for s in entries if s.get("traj_extracted"))
        ade_vals = [s["ade"] for s in entries if "ade" in s]
        fde_vals = [s["fde"] for s in entries if "fde" in s]

        # Aggregate sub-metrics
        sub_keys: set = set()
        for s in entries:
            sub_keys.update(s.get("sub_metrics", {}).keys())
        sub_metrics: Dict[str, Dict] = {}
        for key in sorted(sub_keys):
            vals = [
                s["sub_metrics"][key]
                for s in entries
                if key in s.get("sub_metrics", {})
            ]
            if vals:
                sub_metrics[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }

        report = {
            "model": model,
            "dataset": dataset,
            "total_samples": len(entries),
            "evaluated": len(score_vals),
            "errors": len(errors),
            "parse_failures": sum(bool(s.get("parse_error")) for s in entries),
            "generation_attempts": sum(s.get("generation_attempts", 0) for s in entries),
            "score_key": score_key,
            "require_sample_valid": require_sample_valid,
            report_key: EvalReportBuilder._summary(score_vals),
            "sub_metrics": sub_metrics,
            "openloop": {
                "ade": round(float(np.mean(ade_vals)), 4) if ade_vals else None,
                "fde": round(float(np.mean(fde_vals)), 4) if fde_vals else None,
                "trajectory_extract_rate": (
                    round(extracted / len(entries), 4) if entries else 0
                ),
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if report_key != "pdms":
            report["pdms"] = EvalReportBuilder._summary([
                float(s["pdm_score"]) for s in entries
                if not s.get("error") and "pdm_score" in s
            ])
        return report

    @staticmethod
    def extract_failed_stats(
        scored: Union[List[ScoredRecord], List[Dict]],
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Summarize common low-score sub-metrics for quick failure inspection."""
        entries = [
            s.to_dict() if isinstance(s, ScoredRecord) else s
            for s in scored
        ]
        failed = [
            s for s in entries
            if not s.get("error")
            and float(s["pdm_score"]) <= threshold
        ]
        reasons: Counter[str] = Counter()
        for entry in failed:
            for key, value in entry.get("sub_metrics", {}).items():
                if isinstance(value, (int, float)) and value <= threshold:
                    reasons[key] += 1
        return {
            "failed_count": len(failed),
            "total_evaluated": len([s for s in entries if not s.get("error")]),
            "top_failure_reasons": reasons.most_common(10),
        }
