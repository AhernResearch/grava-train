<div align="center">
  <img src="assets/grava_logo.png" alt="GRAVA logo" width="144">
  <h1>GRAVA Train</h1>
  <p><strong>Train grounded reasoning-to-action models for autonomous driving.</strong></p>
  <p>
    <a href="https://github.com/AhernResearch/grava"><img alt="Project" src="https://img.shields.io/badge/Project-GRAVA-2563eb"></a>
    <a href="https://arxiv.org/abs/2609.15169"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-6366f1"></a>
    <a href="#data--checkpoints"><img alt="Data" src="https://img.shields.io/badge/Data-GR--NavSim-8b5cf6"></a>
    <a href="#documentation"><img alt="Documentation" src="https://img.shields.io/badge/Read-Documentation-475569"></a>
  </p>
  <p><strong>English</strong> | <a href="README_zh.md">简体中文</a></p>
  <p>
    <a href="#training-workflow">Workflow</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#training--evaluation">Training &amp; Evaluation</a> ·
    <a href="#data--checkpoints">Data &amp; Checkpoints</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

## Overview

GRAVA Train trains and evaluates vision-language-action models (VLAs) for
autonomous driving, with support for supervised fine-tuning, reinforcement learning
and trajectory evaluation. Use the recipes to reproduce
[GRAVA](https://github.com/AhernResearch/grava) training or build your own experiments.

Thanks to the [Qwen](https://github.com/QwenLM/Qwen3-VL) team for their excellent
open models and the [ms-swift](https://github.com/modelscope/ms-swift) team for
their training infrastructure. We build on both to connect trajectory generation,
simulation scoring and reinforcement learning for autonomous driving.

Reinforcement learning supports separate training and inference. Training,
inference and simulation scoring can run on different machines, with independent
scheduling and maintenance. Scale training and inference resources separately as
models or sampling workloads grow. Active RL selects scenes for the next training
round based on the current model's performance.

| Capability | What you can run |
| --- | --- |
| Supervised learning | GRA pretraining and teacher-supervised action warm-up, with full fine-tuning or LoRA |
| Self-distillation | Rejection sampling selects complete model answers for another round of SFT |
| Reinforcement learning | GRPO with driving rewards, separate training and inference, and asynchronous generation |
| Active RL | Policy rollout, recoverable-case selection, and another GRPO round |
| Evaluation | PDMS / CDS scoring and open-loop ADE / FDE against original ground-truth trajectories |

## Training Workflow

[![Figure 4(c): GRAVA four-stage training pipeline](assets/training_workflow.png)](assets/training_workflow.png)

Training pipeline from Figure 4(c) of the [GRAVA paper](https://arxiv.org/abs/2609.15169).
See [SFT recipes](docs/sft.md#recorded-recipes), [self-distillation](docs/self_distillation.md)
and [Active RL](docs/active_rl.md)
for training instructions, and [Data & Checkpoints](#data--checkpoints) for release status.

## Quick Start

### 1. Install

Clone the repository and create a Python 3.10 environment:

```bash
git clone https://github.com/AhernResearch/grava-train.git
cd grava-train
conda create -n grava-train python=3.10 -y
conda activate grava-train

python -m pip install "torch==2.10.0"
python -m pip install "ms-swift==4.0.1" "transformers==5.2.0" \
  "qwen-vl-utils==0.0.14" "deepspeed==0.18.5" "PyYAML==6.0.3"
python -m pip install -e ".[eval]"
```

Use a PyTorch wheel compatible with your CUDA driver. These versions match the
local component-check environment; GPU reproduction and optional vLLM require
separate validation. See [training setup](docs/training.md#dependencies).

### 2. Prepare paths

See [Data & Checkpoints](#data--checkpoints) for data availability.
Once you have the prepared bundle, extract `navsim_image/` beside the JSONL files.
No annotation generation or format conversion is required.

```text
dataset/
├── pretrain.jsonl
├── action_warm_up_sft.jsonl
├── grpo_initial.jsonl
├── grpo_pool.jsonl
├── eval.jsonl
└── navsim_image/<log_name>/CAM_F0/<image_token>.jpg
```

Replace these example paths with your base Qwen3-VL model, dataset directory and
output directory. Keep datasets outside the code checkout.

```bash
export MODEL=/path/to/base-qwen3-vl
export DATA_ROOT=/path/to/dataset
export OUTPUT_DIR=/path/to/outputs/grava
```

### 3. Start pretraining

```bash
python scripts/train/train_sft.py \
  --model "$MODEL" --model_type qwen3_vl \
  --stage_config configs/sft/pretrain.yaml \
  --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR/pretrain"
```

The recipe sets one epoch, learning rate `1e-5`, per-device batch size `4` and
maximum sequence length `4096`. It uses full fine-tuning by default. For multiple
GPUs, use the [torchrun launcher](docs/training.md#sft); match the recorded training
settings in [SFT recipes](docs/sft.md#recorded-recipes) when reproducing an experiment.

## Training & Evaluation

| Task | Entry point | Guide |
| --- | --- | --- |
| Pretraining / action SFT | `scripts/train/train_sft.py` | [Recipes and resume](docs/sft.md) |
| Self-distillation | `scripts/rl/rejection_sampling.py` → `scripts/train/train_sft.py` | [Select targets → SFT](docs/self_distillation.md) |
| GRPO | `scripts/train/train_grpo.py` | [Training and generation modes](docs/training.md#grpo) |
| Separate vLLM server | `train_grpo.py --vllm_mode server` | [Infrastructure and server configuration](#separate-training-and-inference) |
| Active RL | `scripts/eval/rollout.py` → `scripts/rl/select_rollouts.py` | [Rollout → select → train](docs/active_rl.md#one-round) |
| Trajectory evaluation | `scripts/eval/eval_trajectory.py` | [Evaluation setup](docs/training.md#evaluation) |

### Action warm-up

Set `PRETRAIN_CHECKPOINT` to the checkpoint produced above:

```bash
export PRETRAIN_CHECKPOINT=/path/to/pretrain/checkpoint
python scripts/train/train_sft.py \
  --model "$PRETRAIN_CHECKPOINT" --model_type qwen3_vl \
  --stage_config configs/sft/action_warm_up_sft.yaml \
  --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR/action_warm_up_sft"
```

### Self-distillation

**One round of rejection-sampling selection followed by one round of SFT is
self-distillation.** For each prompt in the full pool, sample multiple answers and
keep the highest-scoring valid answer, including its complete think block and action.
If the best score is zero or all answers are invalid, use the original supervised
answer from `solution.reference` as the GT fallback.

Serve the action warm-up checkpoint through vLLM and register the training scenes
on the scoring service. `SERVED_MODEL` must point to that same checkpoint:

```bash
export WARM_UP_CHECKPOINT=/path/to/action-warm-up/checkpoint
export SERVED_MODEL=grava
export VLLM_URL=http://localhost:8000/v1
export SIM_URL=http://localhost:8100

python scripts/rl/rejection_sampling.py \
  --dataset-file grpo_pool.jsonl --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --sim-dataset navtrain \
  --scoring-mode pdms --trajectory-decoder executable_planner \
  --num-rollouts 8 --max-generation-attempts 3 \
  --output "$OUTPUT_DIR/self_distill/self_distill.jsonl"

python scripts/train/train_sft.py \
  --model "$WARM_UP_CHECKPOINT" --model_type qwen3_vl \
  --dataset "$OUTPUT_DIR/self_distill/self_distill.jsonl" --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR/self_distill/checkpoints" \
  --num_train_epochs 1 --learning_rate 5e-6 --batch_size 4 \
  --gradient_accumulation_steps 1 --save_strategy epoch
```

The first command writes a trainable JSONL with one target per input prompt; the
second trains on it. Replace `navtrain` with your scoring service's dataset name.
See the [self-distillation guide](docs/self_distillation.md) for selection rules,
rollout resume and selection from existing rollout files.

### Reinforcement learning (GRPO)

For the first round, initialize from the corresponding SFT / self-distillation
checkpoint and set the scoring service URL.
`SIM_ENGINE_DATASET` must name the training scenes registered on that service;
`navtrain` below is an example registration name.
The command below uses Transformers for generation; vLLM also supports colocated
generation and [a separate rollout server](#separate-training-and-inference).

```bash
export GRPO_INIT_CHECKPOINT=/path/to/grpo-initialization/checkpoint
export SIM_ENGINE_URL=http://localhost:8100
export SIM_ENGINE_DATASET=navtrain
export SIM_ENGINE_TRAJECTORY_FRAME=grava

python scripts/train/train_grpo.py \
  --model "$GRPO_INIT_CHECKPOINT" --dataset grpo_initial.jsonl \
  --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR/grpo" \
  --trajectory_decoder executable_planner --reward_funcs pdms --no_vllm
```

Later rounds use the current policy to select from `grpo_pool.jsonl`, following
[Active RL](docs/active_rl.md). Adjust the learning rate, sampling parameters and
GPU count for your experiment.

#### Separate training and inference

RL for driving models repeatedly samples reasoning and action sequences, then
scores their trajectories in simulation. GRAVA Train separates these workloads:
vLLM generates rollouts, the sim-engine computes trajectory rewards, and the
trainer updates the model with ms-swift and DeepSpeed.

Training and inference use separate GPUs, with their own GPU counts and parallelism
settings. Add training nodes for larger models, or rollout capacity for more
samples. After a parameter update, the trainer synchronizes weights to vLLM for
subsequent sampling. Separate services also let us schedule training, simulation
and inference independently, making asynchronous execution and independent
maintenance easier.

[![Active RL infrastructure: rollout server, sim-engine and training nodes, with model parameter synchronization](assets/rl_infra.png)](assets/rl_infra.png)

| Service | Role | Example deployment |
| --- | --- | --- |
| Rollout server | Generate reasoning-to-action sequences with vLLM | 1 node × 8 H20 GPUs; tensor parallelism, 4,096-token limit and prefix caching |
| Sim-engine | Simulate decoded trajectories and return rewards and simulation details | A separate node with scene caches and dataset-specific evaluators |
| Training nodes | Update all model parameters with GRPO / DAPO | 3 nodes × 8 H20 GPUs; PyTorch distributed data parallelism and DeepSpeed ZeRO-2 |

For each prompt, vLLM samples eight completions. The fixed decoder converts each
valid action into an eight-waypoint trajectory for the sim-engine. After all
completions in a prompt group have been scored, the trainer normalizes the group
rewards and updates the policy. Unparsable actions receive zero reward; overlong
completions are filtered and zero-variance groups are resampled within the
configured limit.

The sim-engine provides NAVSIM PDMS and CDS through dataset-specific
scoring backends. Active-set screening, online RL and checkpoint evaluation reuse
the scoring endpoints, while reward components reuse cached simulation results.

To use a separate rollout server, replace `--no_vllm` in the GRPO command above
with `--vllm_mode server --vllm_server_host HOST --vllm_server_port PORT`, substituting
the server's hostname and port. Use an ms-swift-compatible rollout server that
supports model-weight synchronization. Enable `--async_generate` to overlap
generation with training; this uses the previous step's weights for rollout.
See [server configuration](docs/training.md#grpo)
and [multi-node launch](docs/training.md#multi-node-execution) for setup.

### Evaluation

Serve a trained checkpoint through a vLLM multimodal chat endpoint
and start a compatible scoring service with the evaluation scenes registered.
Set `SERVED_MODEL` to the model name exposed by vLLM.

```bash
export SERVED_MODEL=grava
export VLLM_URL=http://localhost:8000/v1
export SIM_URL=http://localhost:8100
export EVAL_DATASET=navtest

python scripts/eval/eval_trajectory.py --metric pdms \
  --dataset-file eval.jsonl --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --dataset "$EVAL_DATASET" \
  --trajectory-decoder executable_planner --enable-thinking \
  --max-generation-attempts 3 --output "$OUTPUT_DIR/eval_pdms.json" \
  --details "$OUTPUT_DIR/eval_pdms_details.jsonl"
```

Evaluation and Active RL retry unparseable generations up to three total attempts
by default; exhausted outputs receive zero. Use `--max-generation-attempts 1`
for single-attempt results. See [retry semantics](docs/active_rl.md#parse-retries).
CDS requires its corresponding scoring dataset; open-loop metrics use original
GT, as described in [evaluation setup](docs/training.md#evaluation).

## Data & Checkpoints

Follow [GR-NavSim](https://github.com/AhernResearch/gr-navsim) for data releases and
[GRAVA](https://github.com/AhernResearch/grava#release-roadmap) for model releases.
Public bundle and checkpoint download URLs are pending.

| Prepared file | Rows | Purpose |
| --- | ---: | --- |
| `pretrain.jsonl` | 1,185,412 | Scene and object QA pretraining |
| `action_warm_up_sft.jsonl` | 62,111 | Teacher-supervised action warm-up |
| `grpo_initial.jsonl` | 14,938 | Historical first GRPO round |
| `grpo_pool.jsonl` | 102,861 | Full prompt pool for self-distillation and Active RL selection |
| `eval.jsonl` | 12,146 | NAVSIM navtest prompts and original trajectories |

Two output formats are supported: **Executable Planner** (`executable_planner`),
which decodes planner parameters into a trajectory, and **Direct Waypoints**
(`direct_waypoints`), which outputs trajectory coordinates as text.

The prepared bundle includes front-camera JPEG images and Executable Planner
supervision. A Direct Waypoints training bundle is not included. Later GRPO
inputs are generated with the public rollout and selection scripts.

PDMS / CDS evaluation and simulation rewards also need a compatible scoring
service and its scene assets / metric caches, supplied separately from JSONL and
images. For the scoring service, see
[grava-sim-engine](https://github.com/AhernResearch/grava-sim-engine). Configure its startup datasets and unified scoring API as described in
[training setup](docs/training.md).
See [data documentation](docs/data.md) for the directory layout and field definitions.

## Documentation

| Guide | Contents |
| --- | --- |
| [Training setup](docs/training.md) | Dependencies, GPU launchers, GRPO modes and evaluation |
| [SFT](docs/sft.md) | Recipes, stage handoff and checkpoint resume |
| [Self-distillation](docs/self_distillation.md) | Rejection sampling, GT fallback and SFT |
| [Active RL](docs/active_rl.md) | Rollouts, selection rules and generation retries |
| [Data](docs/data.md) | Prepared files, image paths and trajectory labels |
| [Rewards](docs/rewards.md) | Driving and reasoning reward experiments |
| [Code structure](docs/code_structure.md) | Script entry points and reusable components |

For the method, qualitative examples and release roadmap, visit the
[GRAVA paper repository](https://github.com/AhernResearch/grava).

## Citation

```bibtex
@misc{liu2026grava,
  title         = {GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving},
  author        = {Xiao Liu and Haoyu Li and Jianghao Leng and Lin Wang and Chao Sun},
  year          = {2026},
  eprint        = {2609.15169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.15169}
}
```

## Acknowledgements

GRAVA Train builds on Qwen3-VL, ms-swift, Transformers, vLLM and the NAVSIM
community's evaluation tools. We thank their authors and contributors.
