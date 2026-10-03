"""Pure translation from script options and a stage into ms-swift SFT arguments."""

from argparse import Namespace
from typing import Any

from .stages import StageSpec


def build_sft_kwargs(
    args: Namespace,
    stage: StageSpec,
    *,
    model: str,
    output_dir: str,
    is_multi_stage: bool = False,
    adapter: str | None = None,
    resume_checkpoint: str | None = None,
    special_tokens: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Stage overrides take precedence; resume applies only when explicitly passed."""
    dataset = list(stage.dataset)

    kwargs = {
        "model": model,
        "dataset": dataset,
        "strict": True,
        "output_dir": output_dir,
        "num_train_epochs": stage.epochs if stage.epochs is not None else args.num_train_epochs,
        "per_device_train_batch_size": stage.batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": stage.lr if stage.lr is not None else (args.learning_rate or 1e-5),
        "warmup_ratio": args.warmup_ratio,
        "max_length": stage.max_length or args.max_length,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "bf16": args.bf16 and not args.no_bf16,
        "gradient_checkpointing": args.gradient_checkpointing
        and not args.no_gradient_checkpointing,
        "loss_scale": args.loss_scale,
        "do_train": True,
    }

    # Padding-free/packing and attention implementation
    if args.padding_free:
        kwargs["padding_free"] = True
    if args.packing:
        kwargs["packing"] = True
    if args.attn_impl:
        kwargs["attn_impl"] = args.attn_impl

    # FSDP (PyTorch native, alternative to DeepSpeed for Qwen3.5 GDN compatibility)
    if args.fsdp:
        kwargs["fsdp"] = args.fsdp
        # FSDP2 uses its own activation_checkpointing, disable HF gradient_checkpointing
        kwargs["gradient_checkpointing"] = False

    # Resume from checkpoint if specified
    if resume_checkpoint:
        kwargs["resume_from_checkpoint"] = resume_checkpoint

    # Add LoRA config if enabled
    if args.use_lora:
        kwargs["train_type"] = "lora"
        kwargs["lora_rank"] = args.lora_rank
        kwargs["lora_alpha"] = args.lora_alpha
        kwargs["lora_dropout"] = args.lora_dropout
    else:
        kwargs["train_type"] = "full"

    # Add DeepSpeed config if provided
    # Supports both ms-swift built-in presets ('zero2', 'zero3') and custom json paths.
    # Using built-in presets is recommended as ms-swift handles fp16/bf16 compatibility.
    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed

    # Add model type if specified
    if args.model_type:
        kwargs["model_type"] = args.model_type

    # Vision encoder: unfreeze by default for domain adaptation
    if not args.freeze_vision:
        kwargs["freeze_vit"] = False
        kwargs["freeze_aligner"] = False

    # Data loading workers (respect CLI args instead of hardcoding)
    kwargs["dataloader_num_workers"] = args.dataloader_num_workers
    kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor

    # Optional experiment-specific tokens; Planner parameters use ordinary text.
    if special_tokens:
        kwargs["new_special_tokens"] = list(special_tokens)

    # Multi-stage specific settings
    if is_multi_stage:
        kwargs["check_model"] = False
        kwargs["load_args"] = False

    if adapter:
        kwargs["adapters"] = [adapter]
    return kwargs
