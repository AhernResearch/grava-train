#!/usr/bin/env python3
"""Compose single- or multi-stage GRAVA SFT with ms-swift 4.x.

See docs/sft.md for explicit datasets, stage recipes and checkpoint resume.
"""

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import qwen_vl_utils.vision_process as vision
import torch.distributed as dist
from swift.arguments import SftArguments
from swift.pipelines import sft_main

from grava_train.training.checkpoints import training_checkpoint
from grava_train.training.data import register_released_datasets
from grava_train.training.sft_args import build_sft_kwargs
from grava_train.training.stages import load_stages, resume_stage_index
from grava_train.utils.checkpoint_sync import broadcast_string, sync_checkpoint_across_nodes
from grava_train.utils.checkpoint_utils import validate_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SFT training for GRAVA")

    # Model
    parser.add_argument("--model", type=str, required=True, help="Model name or local path")
    parser.add_argument("--model_type", type=str, default=None, help="Model type (e.g., qwen3_vl)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="JSON/JSONL dataset path(s), comma-separated")
    source.add_argument("--stage_config", help="YAML/JSON recipe path or inline JSON")
    parser.add_argument("--data_root", default=None, help="Root for dataset and image paths")

    # Training
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3,
        help="Number of training epochs (global default)",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Default learning rate; stage lr takes precedence",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument(
        "--loss_scale",
        type=str,
        default="default",
        help='ms-swift loss scale strategy, e.g. "default+ignore_empty_think"',
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint saving frequency")
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Checkpoint saving strategy",
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep"
    )

    # LoRA (default: disabled, use --use_lora to enable)
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Use LoRA for efficient fine-tuning (default: False)",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")

    # DataLoader
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--dataloader_prefetch_factor", type=int, default=4, help="Dataloader prefetch factor"
    )

    parser.add_argument(
        "--freeze_vision",
        action="store_true",
        default=False,
        help="Freeze vision encoder (default: unfreeze for domain adaptation)",
    )

    # Performance options
    parser.add_argument(
        "--padding_free",
        action="store_true",
        default=False,
        help="Flatten batch tokens to avoid padding (requires flash_attention)",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        default=False,
        help="Enable sequence packing (requires flash_attention)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default=None,
        help="Attention implementation (e.g., flash_attention_2, sdpa)",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--fsdp",
        type=str,
        default=None,
        help='FSDP config name or path (e.g., "fsdp2" for PyTorch native FSDP2)',
    )

    # Resume training from checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., output/stage3/.../checkpoint-500)",
    )

    parser.add_argument(
        "--resume_stage", help="Interrupted stage name (required for multi-stage resume)"
    )
    parser.add_argument(
        "--image_max_tokens",
        type=int,
        default=int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1200")),
        help="Per-image token limit (default: IMAGE_MAX_TOKEN_NUM or 1200)",
    )

    args = parser.parse_args(argv)
    for name in (
        "num_train_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "image_max_tokens",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be positive and finite")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("--learning_rate must be positive and finite")
    if args.deepspeed and args.fsdp:
        parser.error("--deepspeed and --fsdp are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages, special_tokens = load_stages(
        dataset=args.dataset,
        stage_config=args.stage_config,
        data_dir=args.data_root or ".",
    )
    register_released_datasets(
        [path for stage in stages for path in stage.dataset], args.data_root,
    )
    start = resume_stage_index(stages, args.resume_from_checkpoint, args.resume_stage)
    if args.save_strategy == "no" and len(stages) - start > 1:
        raise ValueError("Stage transitions require checkpoint saving")

    is_main = int(os.environ.get("RANK", "0")) == 0
    model, adapter = args.model, None
    resume = args.resume_from_checkpoint
    if resume:
        # Full optimizer state must already be available on every training node;
        # the model-only checkpoint transport intentionally skips ZeRO state.
        resume = validate_checkpoint(resume, resume=True)
        if args.use_lora:
            adapter = resume
        else:
            model = resume
    progress = {"start_time": datetime.now().isoformat(), "args": vars(args), "stages": {}}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    vision.IMAGE_MAX_TOKEN_NUM = args.image_max_tokens

    for index in range(start, len(stages)):
        stage = stages[index]
        stage_dir = output / f"stage{index + 1}_{stage.name}" if args.stage_config else output
        if is_main:
            print(f"SFT {index + 1}/{len(stages)}: {stage.name}; data={stage.dataset}")
        kwargs = build_sft_kwargs(
            args,
            stage,
            model=model,
            adapter=adapter,
            output_dir=str(stage_dir),
            is_multi_stage=bool(args.stage_config),
            resume_checkpoint=resume,
            special_tokens=special_tokens if index == 0 and not resume else (),
        )
        train_args = SftArguments(**kwargs)
        result = sft_main(train_args)
        resume = None  # Later stages initialize weights with fresh optimizer/scheduler state.

        if dist.is_initialized():
            dist.barrier()
        # Broadcast errors as well as paths so every rank leaves a failed transition.
        outcome = {}
        if is_main:
            try:
                outcome["checkpoint"] = training_checkpoint(
                    result,
                    required=args.save_strategy != "no",
                )
            except Exception:
                outcome["error"] = traceback.format_exc()
        outcome = json.loads(broadcast_string(json.dumps(outcome)))
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        checkpoint = outcome["checkpoint"]
        if checkpoint:
            if index + 1 < len(stages):
                sync_checkpoint_across_nodes(checkpoint)
                validate_checkpoint(checkpoint)
            if args.use_lora:
                adapter = checkpoint
            else:
                model = checkpoint

        error = ""
        if is_main:
            try:
                progress["stages"][stage.name] = {
                    "checkpoint": checkpoint,
                    "completed_at": datetime.now().isoformat(),
                }
                progress_file = output / "training_progress.json"
                pending = progress_file.with_suffix(".json.tmp")
                pending.write_text(json.dumps(progress, indent=2))
                pending.replace(progress_file)
            except Exception:
                error = traceback.format_exc()
        error = broadcast_string(error)
        if error:
            raise RuntimeError(error)
    if is_main:
        print(f"SFT complete. Output: {output}")


if __name__ == "__main__":
    main()
