#!/usr/bin/env python3
"""Greedy trajectory evaluation with PDMS or CDS scoring.

Inference orchestration stays here; parsing, scoring and reporting are components.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List

from grava_train.eval.data_utils import (
    extract_gt_trajectory,
    load_trajectory_samples,
    scene_info_from_sample,
)
from grava_train.eval.scored_results import EvalReportBuilder, ScoredRecord
from grava_train.eval.scoring.scoring_pipeline import ScoringPipeline
from grava_train.eval.scoring.sim_reward_calculator import SimRewardCalculator
from grava_train.inference.vllm_client import VLLMInferenceClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trajectory evaluation (greedy, single prediction per case)",
    )
    p.add_argument("--metric", choices=["pdms", "cds"], default="pdms")
    p.add_argument("--dataset-file", required=True, help="Input JSONL path")
    p.add_argument("--data-root", default=None, help="Root for dataset and image paths")
    p.add_argument("--model", required=True, help="vLLM served model name")
    p.add_argument("--vllm-url", default="http://localhost:8000/v1",
                   help="Comma-separated vLLM URLs")
    p.add_argument("--sim-url", default="http://localhost:8100")
    p.add_argument("--dataset", default=None, help="Scoring dataset; defaults to navtest for PDMS, internal for CDS")
    p.add_argument("--max-workers", type=int, default=32, help="Concurrency")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--details", default=None)
    p.add_argument("--trajectory-decoder", default="direct_waypoints",
                   choices=["direct_waypoints", "executable_planner"])
    p.add_argument("--trajectory-frame", default="grava",
                   choices=["grava", "nuplan"],
                   help="Coordinate frame of model output.")
    p.add_argument("--enable-thinking", action="store_true", default=False,
                   help="Allow model to think (default: no-think)")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max tokens for generation")
    p.add_argument("--max-generation-attempts", type=int, default=3,
                   help="Maximum generations per sample, including the first; retry only parse failures")
    args = p.parse_args(argv)
    if args.max_generation_attempts < 1:
        p.error("--max-generation-attempts must be at least 1")
    if args.dataset is None:
        args.dataset = "navtest" if args.metric == "pdms" else "internal"
    if args.output is None:
        args.output = f"{args.metric}_report.json"
    if args.details is None:
        args.details = f"{args.metric}_details.jsonl"
    return args


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Loading {args.dataset_file}...")
    samples = load_trajectory_samples(
        args.dataset_file,
        args.max_samples,
        args.data_root,
    )
    print(f"  {len(samples)} samples")

    urls = [u.strip() for u in args.vllm_url.split(",") if u.strip()]
    client = VLLMInferenceClient(
        urls=urls,
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    print(f"  vLLM: {len(urls)} endpoint(s)")

    scorer = SimRewardCalculator(
        sim_url=args.sim_url,
        scoring_mode=args.metric,
        dataset=args.dataset,
        trajectory_format=args.trajectory_decoder,
        trajectory_frame=args.trajectory_frame,
    )
    pipeline = ScoringPipeline(scorer=scorer)

    gt_map = {s["id"]: extract_gt_trajectory(s) for s in samples}
    scene_info_map = {s["id"]: scene_info_from_sample(s) for s in samples}

    print(f"Running {args.metric.upper()} eval (greedy, concurrency={args.max_workers})...")
    sem = asyncio.Semaphore(args.max_workers)
    t0 = time.time()
    completed = 0
    total = len(samples)

    async def _infer_and_score(i: int, sample: dict) -> ScoredRecord:
        nonlocal completed
        sample_id = sample["id"]
        for attempt in range(1, args.max_generation_attempts + 1):
            async with sem:
                pred = await client.infer(sample, client_idx=i, enable_thinking=args.enable_thinking)
            record = await asyncio.to_thread(
                pipeline.score,
                prediction=pred,
                sample_id=sample_id,
                rollout_idx=0,
                gt_trajectory=gt_map[sample_id],
                context=sample["messages"][0]["content"],
                scene_info=scene_info_map[sample_id],
            )
            record.generation_attempts = attempt
            if record.traj_extracted:
                break
        completed += 1
        if completed % 100 == 0 or completed == total:
            elapsed = time.time() - t0
            print(f"  Progress: {completed}/{total} "
                  f"({completed / elapsed:.1f} samples/s)")
        return record

    details_path = Path(args.details)
    details_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [_infer_and_score(i, s) for i, s in enumerate(samples)]
    records: List[ScoredRecord] = []
    with details_path.open("w", encoding="utf-8") as fout:
        for coro in asyncio.as_completed(tasks):
            record = await coro
            records.append(record)
            fout.write(record.to_json() + "\n")
            if completed % 500 == 0:
                fout.flush()
        fout.flush()

    elapsed = time.time() - t0
    report = EvalReportBuilder.build_report(
        [r.to_dict() for r in records],
        model=args.model,
        dataset=args.dataset,
        score_key="cds" if args.metric == "cds" else "pdm_score",
        report_key=args.metric,
        require_sample_valid=args.metric == "cds",
    )
    report["generation"] = {
        "max_attempts": args.max_generation_attempts,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to {args.output}")
    print(f"{args.metric.upper()}: mean={report[args.metric]['mean']:.4f}, "
          f"median={report[args.metric]['median']:.4f}")
    print(f"Speed: {len(records) / elapsed:.2f} samples/s")
    if report["openloop"]["ade"] is not None:
        print(f"Open-loop: ADE={report['openloop']['ade']:.4f}, "
              f"FDE={report['openloop']['fde']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
