#!/usr/bin/env python3
"""Unified rollout pipeline: sample K predictions per case and score via sim-engine.

Writes standard ScoredRecord JSONL for sampling analysis and offline scoring.
Outputs feed Active RL selection and rejection-sampled SFT.

Usage::

    # Full rollout: infer + score
    python scripts/eval/rollout.py \
        --dataset-file train.jsonl --data-root /path/to/dataset \
        --model model_name --vllm-url http://host:8900/v1 \
        --sim-url http://host:8100 --sim-dataset navtrain \
        --num-rollouts 8 --temperature 0.7 \
        --trajectory-decoder executable_planner \
        --max-cases 500 --output scored_rollouts.jsonl

    # Infer only (no sim-engine needed)
    python scripts/eval/rollout.py --infer-only \
        --dataset-file train.jsonl --data-root /path/to/dataset \
        --model model_name --vllm-url http://host:8900/v1 \
        --num-rollouts 8 --temperature 0.7 \
        --max-cases 500 --output rollouts_inferred.jsonl

    # Score only (read existing inferred rollouts)
    python scripts/eval/rollout.py --score-only \
        --dataset-file train.jsonl \
        --sim-url http://host:8100 --sim-dataset navtrain \
        --trajectory-decoder executable_planner \
        --output rollouts_inferred.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from grava_train.eval.data_utils import (
    extract_gt_trajectory,
    load_trajectory_samples,
    scene_info_from_sample,
)
from grava_train.eval.scored_results import ScoredRecord, load_scored_records
from grava_train.eval.scoring.scoring_pipeline import ScoringPipeline
from grava_train.eval.scoring.sim_reward_calculator import SimRewardCalculator
from grava_train.inference.vllm_client import VLLMInferenceClient
from grava_train.rewards.cot_parser import detect_reasoning_mode
from grava_train.trajectory.trajectory_parser import extract_trajectory_from_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified rollout pipeline")

    # Data
    p.add_argument("--dataset-file", required=True)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--data-root", default=None, help="Root for dataset and image paths")

    # Inference
    p.add_argument("--model", default="")
    p.add_argument("--vllm-url", default="http://localhost:8900/v1",
                   help="Comma-separated vLLM URLs")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048)

    # Scoring
    p.add_argument("--sim-url", default="http://localhost:8100")
    p.add_argument("--sim-dataset", default="navtrain")
    p.add_argument("--scoring-mode", default="pdms", choices=["pdms", "cds", "continuous", "discrete"])
    p.add_argument("--trajectory-decoder", default="direct_waypoints",
                   choices=["direct_waypoints", "executable_planner"])
    p.add_argument("--trajectory-frame", default="grava",
                   choices=["grava", "nuplan"],
                   help="Coordinate frame of model output. 'grava'=[lateral_right, forward, heading], "
                        "'nuplan'=[forward, lateral_left, heading].")

    # Rollout config
    p.add_argument("--num-rollouts", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=64)

    # Output
    p.add_argument("--output", required=True)

    # Mode
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--infer-only", action="store_true")
    mode_group.add_argument("--score-only", action="store_true")

    # Resume
    p.add_argument("--resume", action="store_true")

    p.add_argument("--max-generation-attempts", type=int, default=3,
                   help="Maximum generations per rollout, including the first; retry only parse failures")
    args = p.parse_args(argv)
    if args.max_generation_attempts < 1:
        p.error("--max-generation-attempts must be at least 1")
    return args


def _load_resume_state(output_path: str) -> Dict[str, set[int]]:
    """Load the actual completed indices so holes can be filled on resume."""
    counts: Dict[str, set[int]] = defaultdict(set)
    p = Path(output_path)
    if not p.exists():
        return counts
    for record in load_scored_records(str(p)):
        counts[record.id].add(record.rollout_idx)
    return counts


async def _run_infer_and_score(args: argparse.Namespace) -> None:
    """Full or infer-only mode."""
    samples = load_trajectory_samples(args.dataset_file, args.max_cases, args.data_root)
    print(f"  {len(samples)} Trajectory cases")

    # Resume
    completed: Dict[str, set[int]] = {}
    if args.resume:
        completed = _load_resume_state(args.output)
        before = len(samples)
        samples = [s for s in samples if not set(range(args.num_rollouts)).issubset(completed.get(s["id"], set()))]
        print(f"  Resume: {before - len(samples)} done, {len(samples)} remaining")

    if not samples:
        print("Nothing to do.")
        return

    # Inference client
    urls = [u.strip() for u in args.vllm_url.split(",") if u.strip()]
    client = VLLMInferenceClient(
        urls=urls, model=args.model,
        max_tokens=args.max_tokens, temperature=args.temperature,
        top_p=args.top_p,
    )

    # Scoring pipeline (skip for infer-only)
    pipeline: Optional[ScoringPipeline] = None
    if not args.infer_only:
        scorer = SimRewardCalculator(
            sim_url=args.sim_url, scoring_mode=args.scoring_mode,
            dataset=args.sim_dataset, trajectory_format=args.trajectory_decoder,
            trajectory_frame=args.trajectory_frame,
        )
        pipeline = ScoringPipeline(scorer=scorer)

    # Pre-extract GT trajectories
    gt_map = {s["id"]: extract_gt_trajectory(s) for s in samples}
    context_map = {
        s["id"]: s["messages"][0]["content"] for s in samples
    }

    # Pipeline: infer with global concurrency, then score each case as one batch.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.resume else "w"

    case_items: list[tuple[dict, list[int]]] = []
    for sample in samples:
        sid = sample["id"]
        already_done = completed.get(sid, set())
        rollout_indices = sorted(set(range(args.num_rollouts)) - already_done)
        if rollout_indices:
            case_items.append((sample, rollout_indices))

    total_tasks = sum(len(indices) for _, indices in case_items)

    print(f"  Dispatching {total_tasks} rollout tasks "
          f"(infer_concurrency={args.max_workers})...")

    infer_sem = asyncio.Semaphore(args.max_workers)
    t0 = time.time()
    infer_done = 0
    score_done = 0

    async def _infer_one(sample: dict, rollout_idx: int) -> str:
        nonlocal infer_done
        async with infer_sem:
            pred = await client.infer(sample, client_idx=rollout_idx, enable_thinking=args.enable_thinking)
        infer_done += 1
        if infer_done % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [generation calls {infer_done}] "
                  f"{infer_done/elapsed:.1f} infer/s")
        return pred

    async def _run_case(sample: dict, rollout_indices: list[int]) -> list[ScoredRecord]:
        nonlocal score_done
        sid = sample["id"]
        remaining = rollout_indices
        by_index = {}
        for attempt in range(1, args.max_generation_attempts + 1):
            predictions = await asyncio.gather(*[
                _infer_one(sample, rollout_idx) for rollout_idx in remaining
            ])
            if pipeline is not None:
                attempted = await asyncio.to_thread(
                    pipeline.score_batch,
                    scene_info=scene_info_from_sample(sample),
                    predictions=predictions,
                    sample_id=sid,
                    gt_trajectory=gt_map[sid],
                    rollout_indices=remaining,
                    context=context_map[sid],
                )
            else:
                attempted = []
                for pred, rollout_idx in zip(predictions, remaining, strict=True):
                    trajectory = extract_trajectory_from_text(
                        pred, output_format=args.trajectory_decoder, context=context_map[sid],
                    )
                    attempted.append(ScoredRecord(
                        id=sid, rollout_idx=rollout_idx, prediction=pred,
                        pdm_score=0.0, scored=False,
                        traj_extracted=trajectory is not None, mode=detect_reasoning_mode(pred),
                        parse_error="Failed to extract trajectory" if trajectory is None else None,
                    ))
            for record in attempted:
                record.generation_attempts = attempt
                by_index[record.rollout_idx] = record
            remaining = [r.rollout_idx for r in attempted if not r.traj_extracted]
            if not remaining:
                break
        records = [by_index[index] for index in rollout_indices]

        score_done += len(records)
        if score_done % 100 == 0 or score_done == total_tasks:
            elapsed = time.time() - t0
            print(f"  [score {score_done}/{total_tasks}] "
                  f"{score_done/elapsed:.1f} scored/s")
        return records

    # --- Run pipeline ---
    with open(output_path, file_mode) as fout:
        case_tasks = [
            asyncio.create_task(_run_case(sample, rollout_indices))
            for sample, rollout_indices in case_items
        ]
        for task in asyncio.as_completed(case_tasks):
            for record in await task:
                fout.write(record.to_json() + "\n")
            fout.flush()
        fout.flush()

    elapsed = time.time() - t0
    print(f"\nDone: {total_tasks} rollouts in {elapsed:.1f}s → {args.output}")


async def _run_score_only(args: argparse.Namespace) -> None:
    """Score existing inferred rollouts."""
    print(f"Loading inferred rollouts from {args.output}...")
    records = load_scored_records(args.output)
    unscored = [r for r in records if not r.scored]
    print(f"  {len(records)} total, {len(unscored)} unscored")

    if not unscored:
        print("Nothing to score.")
        return

    # Load GT trajectories
    gt_map: Dict[str, Optional[List]] = {}
    context_map: Dict[str, str] = {}
    scene_map = {}
    print(f"Loading GT from {args.dataset_file}...")
    all_samples = load_trajectory_samples(args.dataset_file, data_root=args.data_root)
    for s in all_samples:
        scene_map[s["id"]] = scene_info_from_sample(s)
        gt_map[s["id"]] = extract_gt_trajectory(s)
        context_map[s["id"]] = s["messages"][0]["content"]

    # Build scoring pipeline
    scorer = SimRewardCalculator(
        sim_url=args.sim_url, scoring_mode=args.scoring_mode,
        dataset=args.sim_dataset, trajectory_format=args.trajectory_decoder,
        trajectory_frame=args.trajectory_frame,
    )
    pipeline = ScoringPipeline(scorer=scorer)

    # Score and rewrite
    scored_map: Dict[str, Dict[int, ScoredRecord]] = defaultdict(dict)
    for r in records:
        scored_map[r.id][r.rollout_idx] = r

    unscored_by_id: Dict[str, List[ScoredRecord]] = defaultdict(list)
    for r in unscored:
        unscored_by_id[r.id].append(r)

    def _score_group(sid: str, group: List[ScoredRecord]) -> tuple[str, List[ScoredRecord]]:
        group = sorted(group, key=lambda x: x.rollout_idx)
        scored = pipeline.score_batch(
            scene_info=scene_map[sid],
            predictions=[r.prediction for r in group],
            sample_id=sid,
            gt_trajectory=gt_map.get(sid),
            rollout_indices=[r.rollout_idx for r in group],
            context=context_map.get(sid),
        )

        for record, previous in zip(scored, group, strict=True):
            record.generation_attempts = previous.generation_attempts
        return sid, scored

    t0 = time.time()
    done = 0
    score_workers = max(1, min(args.max_workers, 64))
    print(f"  Batch scoring {len(unscored_by_id)} cases with {score_workers} workers")
    with ThreadPoolExecutor(max_workers=score_workers) as pool:
        futures = {
            pool.submit(_score_group, sid, group): sid
            for sid, group in unscored_by_id.items()
        }
        for future in as_completed(futures):
            sid, new_records = future.result()
            for new_record in new_records:
                scored_map[sid][new_record.rollout_idx] = new_record
            done += len(new_records)
            if done % 100 == 0 or done == len(unscored):
                elapsed = time.time() - t0
                print(f"  Scored {done}/{len(unscored)} ({done / max(elapsed, 1e-6):.1f}/s)")

    # Rewrite
    all_records = []
    for sid_records in scored_map.values():
        for record in sorted(sid_records.values(), key=lambda r: r.rollout_idx):
            all_records.append(record)

    pending = Path(args.output).with_suffix(".jsonl.tmp")
    with pending.open("w") as f:
        for r in all_records:
            f.write(r.to_json() + "\n")
    pending.replace(args.output)

    elapsed = time.time() - t0
    print(f"\nDone: scored {len(unscored)} rollouts in {elapsed:.1f}s → {args.output}")


def main():
    args = parse_args()

    source = Path(args.dataset_file).expanduser()
    if args.data_root is not None:
        source = Path(args.data_root).expanduser() / source
    source = source.resolve(strict=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    invocation = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "args": vars(args), "dataset": str(source)}
    with output.with_suffix(".runs.jsonl").open("a") as stream:
        stream.write(json.dumps(invocation) + "\n")
    print(f"Loading {args.dataset_file}...")
    if args.score_only:
        asyncio.run(_run_score_only(args))
    else:
        asyncio.run(_run_infer_and_score(args))

    # Summary
    if Path(args.output).exists():
        records = load_scored_records(args.output)
        if records:
            key = {"pdms": "pdm_score", "cds": "cds", "continuous": "rl_score",
                   "discrete": "rl_score"}[args.scoring_mode]
            scored = [r for r in records if r.scored and r.error is None]
            scores = np.array([getattr(r, key) for r in scored if getattr(r, key) is not None])
            n_cases = len({r.id for r in records})
            print(f"\nSummary: {len(records)} rollouts, {n_cases} cases")
            if len(scores):
                n_zero = int((scores == 0).sum())
                print(f"  {args.scoring_mode.upper()}: mean={scores.mean():.4f}, "
                      f"median={np.median(scores):.4f}")
                print(f"  Zero rate: {n_zero}/{len(scores)} "
                      f"({100*n_zero/len(scores):.1f}%)")


if __name__ == "__main__":
    main()
