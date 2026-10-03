#!/usr/bin/env python3
"""Run scored rollouts and export one complete SFT target per input prompt."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from grava_train.rl.rejection_sampling import build_sft_sample, select_best_rollouts
from grava_train.utils.jsonl_utils import iter_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", required=True, help="Full prompt-only trajectory JSONL")
    parser.add_argument("--data-root", help="Root for relative dataset and image paths")
    parser.add_argument("--output", required=True, help="Output SFT JSONL")
    parser.add_argument("--rollouts", nargs="+", help="Rollout file, or existing shards to select")
    parser.add_argument("--select-only", action="store_true", help="Select without model calls")
    parser.add_argument("--resume", action="store_true", help="Resume missing rollout indices")
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--model", help="Model name served by vLLM")
    parser.add_argument("--vllm-url", default="http://localhost:8900/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--max-generation-attempts", type=int, default=3)
    parser.add_argument("--sim-url", default="http://localhost:8100")
    parser.add_argument("--sim-dataset", default="navtrain")
    parser.add_argument(
        "--scoring-mode", choices=["pdms", "cds", "continuous", "discrete"], default="pdms",
    )
    parser.add_argument(
        "--trajectory-decoder", choices=["executable_planner", "direct_waypoints"],
        default="executable_planner",
    )
    parser.add_argument("--trajectory-frame", choices=["grava", "nuplan"], default="grava")
    args = parser.parse_args(argv)
    if args.num_rollouts < 1 or args.max_generation_attempts < 1:
        parser.error("--num-rollouts and --max-generation-attempts must be positive")
    if not args.select_only and not args.model:
        parser.error("--model is required when generating rollouts")
    if not args.select_only and args.rollouts and len(args.rollouts) != 1:
        parser.error("Generating rollouts requires one --rollouts output file")
    if args.select_only and args.resume:
        parser.error("Use --resume when generating rollouts, not with --select-only")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.dataset_file).expanduser()
    if args.data_root:
        source = Path(args.data_root).expanduser() / source
    source = source.resolve(strict=True)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else source.parent
    output = Path(args.output).expanduser().resolve()
    rollouts = [Path(p).expanduser().resolve() for p in args.rollouts] if args.rollouts else [
        output.with_suffix(".rollouts.jsonl"),
    ]
    pending = output.with_suffix(output.suffix + ".tmp")
    stats_path = output.with_suffix(".stats.json")
    paths = [source, output, pending, stats_path, *rollouts]
    if not args.select_only:
        paths.append(rollouts[0].with_suffix(".runs.jsonl"))
    if len(set(paths)) != len(paths):
        raise ValueError("Dataset, rollout, SFT output and statistics paths must be distinct")

    source_ids = set()
    for sample in iter_jsonl(source):
        sid = sample["id"]
        if sid in source_ids:
            raise ValueError(f"Duplicate input ID: {sid}")
        if sample["qa_type"] != "Trajectory" or [m["role"] for m in sample["messages"]] != ["user"]:
            raise ValueError(f"Expected a single user trajectory prompt: {sid}")
        source_ids.add(sid)

    if not args.select_only:
        rollout_script = Path(__file__).resolve().parents[1] / "eval" / "rollout.py"
        command = [
            sys.executable, str(rollout_script), "--dataset-file", str(source),
            "--data-root", str(data_root), "--output", str(rollouts[0]), "--enable-thinking",
        ]
        for name in (
            "model", "vllm_url", "temperature", "top_p", "max_tokens", "max_workers",
            "num_rollouts", "max_generation_attempts", "sim_url", "sim_dataset",
            "scoring_mode", "trajectory_decoder", "trajectory_frame",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)

    best = select_best_rollouts(rollouts, args.num_rollouts, args.scoring_mode)
    if best.keys() != source_ids:
        raise ValueError("Rollout IDs must cover every input prompt exactly")

    counts = {"rollout": 0, "reference": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("w", encoding="utf-8") as stream:
        for sample in iter_jsonl(source):
            target, origin = build_sft_sample(
                sample, best[sample["id"]], scoring_mode=args.scoring_mode,
                trajectory_decoder=args.trajectory_decoder,
            )
            stream.write(json.dumps(target, ensure_ascii=False) + "\n")
            counts[origin] += 1
    pending.replace(output)
    stats = {
        "input_cases": len(source_ids), "output_cases": sum(counts.values()),
        "rollout_targets": counts["rollout"], "reference_targets": counts["reference"],
        "num_rollouts": args.num_rollouts, "scoring_mode": args.scoring_mode,
        "trajectory_decoder": args.trajectory_decoder,
        "source_dataset": str(source), "rollouts": [str(p) for p in rollouts],
        "image_paths_relative_to": str(data_root), "output": str(output),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
