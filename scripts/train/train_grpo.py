#!/usr/bin/env python3
"""GRPO training on released JSONL, with ms-swift DAPO/GDPO options.

See docs/data.md for data paths and docs/training.md for launch settings.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from swift.arguments import RLHFArguments
from swift.pipelines import rlhf_main

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grava_train.training.data import register_released_datasets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GRAVA GRPO training (DAPO + GDPO)')

    # Model
    parser.add_argument('--model', type=str, required=True,
                        help='Base model path or HF name')
    parser.add_argument('--model_type', type=str, default='qwen3_vl',
                        help='Model type (default: qwen3_vl)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='SFT checkpoint directory to resume from')

    # Data
    parser.add_argument("--data_root", default=None, help="Root for dataset and image paths")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset JSON for RL training')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Max input sequence length')
    parser.add_argument('--max_completion_length', type=int, default=4096,
                        help='Max completion length (think mode can be 2000+ tokens)')

    # Performance options
    parser.add_argument('--padding_free', action='store_true', default=False,
                        help='Flatten batch tokens to avoid padding (requires flash attention)')
    parser.add_argument('--attn_impl', type=str, default=None,
                        help='Attention implementation (e.g., flash_attention_2, sdpa)')

    # GRPO generation
    parser.add_argument('--num_generations', type=int, default=8,
                        help='Number of generations per prompt (G in GRPO)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling')

    # Training
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num_iterations', type=int, default=500,
                        help='Number of RL iterations (max_steps)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='Learning rate (10x lower than SFT)')
    parser.add_argument('--beta', type=float, default=0.001,
                        help='KL divergence coefficient (beta)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        help='LR scheduler type (e.g., cosine, constant, linear)')
    parser.add_argument('--warmup_ratio', type=float, default=None,
                        help='Warmup ratio for LR scheduler')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for LR scheduler')

    # Training mode (full fine-tuning vs LoRA)
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Use LoRA (default: full fine-tuning)')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')

    # DAPO improvements
    parser.add_argument('--loss_type', type=str, default='dapo',
                        help='Loss type: dapo (token-level) or grpo')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Clip low (standard GRPO)')
    parser.add_argument('--epsilon_high', type=float, default=0.28,
                        help='Clip high (DAPO Clip-Higher, prevents entropy collapse)')
    parser.add_argument('--dynamic_sample', action='store_true', default=True,
                        help='DAPO Dynamic Sampling (filter zero-variance groups)')
    parser.add_argument('--no_dynamic_sample', dest='dynamic_sample', action='store_false',
                        help='Disable dynamic sampling')
    parser.add_argument('--max_resample_times', type=int, default=3,
                        help='Max resample attempts for dynamic sampling')
    parser.add_argument('--overlong_filter', action='store_true', default=True,
                        help='DAPO Overlong Filtering')
    parser.add_argument('--no_overlong_filter', dest='overlong_filter', action='store_false',
                        help='Disable overlong filtering')

    # GDPO (per-reward normalization)
    parser.add_argument('--scale_rewards', type=str, default='gdpo',
                        help='Reward scaling: gdpo (per-reward normalization) or default')

    # Reward configuration
    parser.add_argument('--reward_funcs', type=str, nargs='+',
                        default=['pdms'],
                        help='Reward function names (registered in reward_plugin.py)')
    parser.add_argument('--reward_weights', type=float, nargs='+',
                        default=[1.0],
                        help='Weights for each reward function')
    parser.add_argument('--external_plugins', type=str, nargs='+',
                        default=[str(Path(__file__).resolve().parent / 'plugins/reward_plugin.py')],
                        help='Path(s) to ORM plugin file(s)')

    # Trajectory decoder
    parser.add_argument('--trajectory_decoder', type=str, default='direct_waypoints',
                        choices=['direct_waypoints', 'executable_planner'],
                        help='Trajectory decoder for reward scoring (default: direct_waypoints)')

    # vLLM inference
    parser.add_argument('--use_vllm', action='store_true', default=True,
                        help='Use vLLM for generation')
    parser.add_argument('--no_vllm', dest='use_vllm', action='store_false',
                        help='Disable vLLM')
    parser.add_argument('--vllm_mode', type=str, default='colocate',
                        help='vLLM mode: colocate or server')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.5,
                        help='vLLM GPU memory fraction (colocate mode)')
    parser.add_argument('--vllm_server_host', type=str, nargs='+', default=None,
                        help='vLLM server host(s) (server mode, space-separated for multi-server)')
    parser.add_argument('--vllm_server_port', type=int, nargs='+', default=[8000],
                        help='vLLM server port(s) (server mode, space-separated for multi-server)')
    parser.add_argument('--vllm_enforce_eager', action='store_true', default=False,
                        help='Disable CUDA Graph for vLLM')

    # Colocate optimization (train-inference overlap & memory)
    parser.add_argument('--async_generate', action='store_true', default=False,
                        help='1-step off-policy overlap: rollout uses prev-step weights')
    parser.add_argument('--offload_optimizer', action='store_true', default=False,
                        help='Offload optimizer state to CPU during inference')
    parser.add_argument('--offload_model', action='store_true', default=False,
                        help='Offload model weights to CPU during inference')
    parser.add_argument('--sleep_level', type=int, default=0,
                        help='vLLM memory release level during training (0=none, 1=partial, 2=full)')
    parser.add_argument('--steps_per_generation', type=int, default=1,
                        help='Reuse one rollout for N training steps (reduces inference frequency)')

    # Off-policy correction (for async_generate)
    parser.add_argument('--rollout_importance_sampling_mode', type=str, default=None,
                        help='IS correction: token_truncate, token_mask, sequence_truncate, sequence_mask')
    parser.add_argument('--rollout_importance_sampling_threshold', type=float, default=2.0,
                        help='IS ratio clipping threshold')

    # DeepSpeed
    parser.add_argument('--deepspeed', type=str, default='',
                        help='DeepSpeed config file path')

    # Thinking control
    parser.add_argument('--enable_thinking', type=str, default=None,
                        choices=['true', 'false'],
                        help='Enable/disable thinking mode (Qwen3.5 hybrid). '
                             'false=add non-thinking prefix, suppresses <think> generation')

    # Other
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Save checkpoint every N steps')
    parser.add_argument('--save_total_limit', type=int, default=5,
                        help='Max checkpoints to keep')
    parser.add_argument('--logging_steps', type=int, default=5,
                        help='Log every N steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args(argv)


def main():
    args = parse_args()
    dataset = Path(args.dataset).expanduser()
    if args.data_root:
        dataset = Path(args.data_root).expanduser() / dataset
    args.dataset = str(dataset.resolve(strict=True))
    register_released_datasets([args.dataset], args.data_root)

    reward_mode = os.environ.get('REWARD_MODE', 'continuous')
    if reward_mode == 'hybrid':
        raise ValueError(
            'REWARD_MODE=hybrid (GRPO with auxiliary SFT) has been removed. '
            'Use REWARD_MODE=continuous or discrete for GRPO.'
        )

    # Validate reward_funcs and reward_weights length
    if len(args.reward_funcs) != len(args.reward_weights):
        raise ValueError(
            f"--reward_funcs ({len(args.reward_funcs)}) and "
            f"--reward_weights ({len(args.reward_weights)}) must have same length"
        )

    # Print configuration
    print("=" * 80)
    print("GRAVA GRPO Training (DAPO + GDPO)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.ckpt_dir or 'None (from base model)'}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Training: {'LoRA' if args.use_lora else 'Full fine-tuning'}")
    print(f"Reward funcs: {args.reward_funcs}")
    print(f"Reward weights: {args.reward_weights}")
    print(f"DAPO: loss_type={args.loss_type}, eps=[{args.epsilon}, {args.epsilon_high}]")
    print(f"       dynamic_sample={args.dynamic_sample}, overlong_filter={args.overlong_filter}")
    print(f"GDPO: scale_rewards={args.scale_rewards}")
    print(f"Generation: G={args.num_generations}, T={args.temperature}, beta={args.beta}")
    print(f"vLLM: {args.use_vllm} (mode={args.vllm_mode})")
    print(f"Colocate opt: async_generate={args.async_generate}, offload_optimizer={args.offload_optimizer}, "
          f"sleep_level={args.sleep_level}, steps_per_generation={args.steps_per_generation}")
    print(f"Reward mode: {reward_mode}")
    print(f"Trajectory decoder: {args.trajectory_decoder}")
    print("=" * 80)

    # Set trajectory decoder for reward plugin (plugin reads from env on import)
    os.environ['TRAJECTORY_DECODER'] = args.trajectory_decoder

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save training config
    config = {
        'reward_funcs': args.reward_funcs,
        'reward_weights': args.reward_weights,
        'loss_type': args.loss_type,
        'epsilon': args.epsilon,
        'epsilon_high': args.epsilon_high,
        'dynamic_sample': args.dynamic_sample,
        'overlong_filter': args.overlong_filter,
        'scale_rewards': args.scale_rewards,
        'num_generations': args.num_generations,
        'temperature': args.temperature,
        'beta': args.beta,
        'learning_rate': args.learning_rate,
        'lr_scheduler_type': args.lr_scheduler_type,
        'warmup_ratio': args.warmup_ratio,
        'warmup_steps': args.warmup_steps,
        'use_lora': args.use_lora,
        'use_vllm': args.use_vllm,
        'vllm_mode': args.vllm_mode,
        'async_generate': args.async_generate,
        'offload_optimizer': args.offload_optimizer,
        'sleep_level': args.sleep_level,
        'steps_per_generation': args.steps_per_generation,
        'trajectory_decoder': args.trajectory_decoder,
        'padding_free': args.padding_free,
        'attn_impl': args.attn_impl,
    }
    if int(os.environ.get('RANK', '0')) == 0:
        with open(Path(args.output_dir) / 'grpo_config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # Build GRPO arguments for ms-swift 4.0
    rlhf_kwargs = {
        # Model
        'model': args.model,
        'model_type': args.model_type,
        # Data
        'dataset': [args.dataset],
        'strict': True,
        'max_length': args.max_length,
        # RLHF
        'rlhf_type': 'grpo',
        'output_dir': args.output_dir,
        'num_train_epochs': 1,
        'max_steps': args.num_iterations,
        'per_device_train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'lr_scheduler_type': args.lr_scheduler_type,
        'save_steps': args.save_steps,
        'save_total_limit': args.save_total_limit,
        'logging_steps': args.logging_steps,
        'seed': args.seed,
        'bf16': True,
        'gradient_checkpointing': True,
        'check_model': False,
        'load_args': False,
        # Override Qwen3.5 default '<think>\n' prefix — let model freely choose <think> or <chain>
        'response_prefix': '',
        # GRPO generation
        'num_generations': args.num_generations,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_completion_length': args.max_completion_length,
        # KL
        'beta': args.beta,
        # DAPO improvements
        'loss_type': args.loss_type,
        'epsilon': args.epsilon,
        'epsilon_high': args.epsilon_high,
        'dynamic_sample': args.dynamic_sample,
        'max_resample_times': args.max_resample_times,
        'overlong_filter': args.overlong_filter,
        # GDPO
        'scale_rewards': args.scale_rewards,
        # ORM reward functions (registered by external_plugins)
        'reward_funcs': args.reward_funcs,
        'reward_weights': args.reward_weights,
        'external_plugins': args.external_plugins,
    }
    if args.warmup_ratio is not None:
        rlhf_kwargs['warmup_ratio'] = args.warmup_ratio
    if args.warmup_steps:
        rlhf_kwargs['warmup_steps'] = args.warmup_steps

    if args.padding_free:
        rlhf_kwargs['padding_free'] = True
    if args.attn_impl:
        rlhf_kwargs['attn_impl'] = args.attn_impl

    # Training type: full fine-tuning or LoRA
    if args.use_lora:
        rlhf_kwargs['train_type'] = 'lora'
        rlhf_kwargs['lora_rank'] = args.lora_rank
        rlhf_kwargs['lora_alpha'] = args.lora_alpha
    else:
        rlhf_kwargs['train_type'] = 'full'

    # vLLM
    if args.use_vllm:
        rlhf_kwargs['use_vllm'] = True
        rlhf_kwargs['vllm_mode'] = args.vllm_mode
        rlhf_kwargs['vllm_gpu_memory_utilization'] = args.vllm_gpu_memory_utilization
        if args.vllm_server_host:
            rlhf_kwargs['vllm_server_host'] = args.vllm_server_host
            rlhf_kwargs['vllm_server_port'] = args.vllm_server_port
        if args.vllm_enforce_eager:
            rlhf_kwargs['vllm_enforce_eager'] = True

    # Colocate optimization
    if args.async_generate:
        rlhf_kwargs['async_generate'] = True
    if args.offload_optimizer:
        rlhf_kwargs['offload_optimizer'] = True
    if args.offload_model:
        rlhf_kwargs['offload_model'] = True
    if args.sleep_level > 0:
        rlhf_kwargs['sleep_level'] = args.sleep_level
    if args.steps_per_generation > 1:
        rlhf_kwargs['steps_per_generation'] = args.steps_per_generation

    # Off-policy correction (only meaningful with async_generate)
    if args.rollout_importance_sampling_mode:
        rlhf_kwargs['rollout_importance_sampling_mode'] = args.rollout_importance_sampling_mode
        rlhf_kwargs['rollout_importance_sampling_threshold'] = args.rollout_importance_sampling_threshold

    # Checkpoint: override model path to load from checkpoint
    if args.ckpt_dir:
        rlhf_kwargs['model'] = args.ckpt_dir
        print(f"\nLoading checkpoint from: {args.ckpt_dir}")

    # DeepSpeed
    if args.deepspeed:
        rlhf_kwargs['deepspeed'] = args.deepspeed

    # Thinking control
    if args.enable_thinking is not None:
        rlhf_kwargs['enable_thinking'] = (args.enable_thinking == 'true')


    rlhf_args = RLHFArguments(**rlhf_kwargs)

    # Run GRPO training
    print("\nStarting GRPO training...")
    print("-" * 80)

    result = rlhf_main(rlhf_args)
    print("-" * 80)
    print("GRPO Training Complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    return result


if __name__ == '__main__':
    main()
