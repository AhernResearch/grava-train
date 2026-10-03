# Training setup

SFT and GRPO are external Python workflows built on ms-swift 4.x. Launchers only
invoke torchrun; they do not install packages, patch dependencies, mount storage,
copy models, choose GPUs or start a monitoring service.

## Dependencies

The local environment used for component checks has the following versions:

| Package | Version |
| --- | --- |
| Python | 3.10 |
| torch | 2.10.0 |
| ms-swift | 4.0.1 |
| transformers | 5.2.0 |
| qwen-vl-utils | 0.0.14 |
| deepspeed | 0.18.5 |
| PyYAML | 6.0.3 |

Install a torch build suitable for your CUDA environment, then install the
training libraries explicitly before launching. No startup script changes them.
This is the checked local package snapshot, not a validated GPU reproduction
environment or a complete dependency lock. vLLM is optional and was not installed
in this local environment; its GPU compatibility needs separate validation.

Coordinate conversions and the HTTP scoring client are included in `grava-train`.
The client sends nuPlan coordinates to the existing scoring endpoints; running
GRPO or simulation evaluation still requires the scoring service and scene assets.
Training and inference libraries listed above must be installed separately.

## Data and images

Prepare JSON/JSONL datasets and ordinary image files accessible on each training
node. Relative image paths resolve against `--data_root`, or against the JSONL
parent directory when no root is supplied. Absolute image paths also work.
The prepared bundle contains the current front-camera image; any motion history
is already included in the prompt. The launchers preserve the working directory.

Old `tar.gz::member` image paths must be replaced with extracted image paths in
prepared datasets. Neither training nor evaluation extracts archives. Historical
recipe filenames identify the data lineage; they do not guarantee the original
image paths work on another machine. No source dataset is modified automatically.

## Released data

Use the prepared JSONL and image bundle described in [data.md](data.md). Original
annotation generation and dataset production are maintained outside this repository.
The training entrypoints resolve relative images using `--data_root` or the JSONL
parent directory; users do not run a data conversion step.

## SFT

Set `MODEL`, `DATASET` and `OUTPUT_DIR` to your model, prepared data and output
paths, then run:

```bash
NPROC_PER_NODE=8 bash scripts/launch/sft.sh \
  --model "$MODEL" --dataset "$DATASET" --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 --deepspeed zero2
```

Use `--stage_config` instead of `--dataset` for a staged recipe. The script owns
the stage loop, checkpoint handoff and progress recording; components in `src`
parse configurations, build framework arguments and validate checkpoints.
The public stage names are `pretrain` and `action_warm_up_sft`. The action warm-up
recipe uses teacher supervision and initializes from the pretrain checkpoint.
See [SFT recipes and resume](sft.md) for precedence and reproduction settings.

## GRPO

Use `grpo_initial.jsonl` for the historical first round. Generate later inputs
with [rollout and selection](active_rl.md) from `grpo_pool.jsonl`. Set `MODEL` to
the corresponding checkpoint and `DATASET` to the selected GRPO dataset:

```bash
NPROC_PER_NODE=8 bash scripts/launch/grpo.sh \
  --model "$MODEL" --dataset "$DATASET" --output_dir "$OUTPUT_DIR" \
  --reward_funcs pdms --reward_weights 1.0 \
  --trajectory_decoder executable_planner --deepspeed zero2 --no_vllm
```

This example disables optional vLLM. To use it, install a compatible version and
omit `--no_vllm`; select `--vllm_mode colocate` or `server` and supply server hosts
and ports as needed. Async generation, importance sampling, CPU offload and sleep
settings remain explicit framework options, not launcher defaults. These options
can affect training behavior; use the original experiment settings for reproduction.

For separate-server generation, start an ms-swift-compatible rollout server on
the inference GPUs, then replace `--no_vllm` in the GRPO command with:

```bash
--vllm_mode server --vllm_server_host "$ROLLOUT_HOST" --vllm_server_port 8000
```

Set `ROLLOUT_HOST` to its hostname or IP, without a URL scheme. This training
rollout server must support ms-swift's weight-update protocol; the ordinary
OpenAI-compatible chat endpoint used by evaluation is a different interface.

Driving rewards require the scoring engine to be running. Configure
`SIM_ENGINE_URL`, `SIM_ENGINE_DATASET` and `SIM_ENGINE_TRAJECTORY_FRAME` for your
prepared evaluation assets. The engine's HTTP API computes rewards and is retained;
it is independent of the removed remote administration service. Small per-rollout
score reuse also remains. See [reward experiments](rewards.md).

For example, if the training scene collection is registered as `navtrain`:

```bash
export SIM_ENGINE_URL=http://localhost:8100
export SIM_ENGINE_DATASET=navtrain
export SIM_ENGINE_TRAJECTORY_FRAME=grava
```

Set these variables before launching GRPO. The scoring dataset must match the
training rows, rather than the client's default evaluation collection.

For the paper's Closed-loop Driving Score, select `--reward_funcs cds` and the
corresponding scoring dataset. Evaluate it with `scripts/eval/eval_trajectory.py --metric cds`.
Both prediction records and report summaries use `cds` as the metric key.

Learning-rate warmup (`--warmup_ratio`, `--warmup_steps`) remains available.
Image-cache prefetch has been removed.

## Evaluation

Install the remote inference client for evaluation and rollout:

```bash
python -m pip install -e ".[eval]"
```

The client sends front-camera images to a running vLLM multimodal chat endpoint;
`--model` must match that endpoint's served model name. Set `VLLM_URL` to its
API root, such as `http://localhost:8000/v1`.

Start [grava-sim-engine](https://github.com/AhernResearch/grava-sim-engine)
separately and configure its datasets at startup with `--dataset NAME=PATH`.
All trajectory modes use `/v1/score` and `/v1/score/batch`, selected by
`scoring_mode=pdms|cds|continuous|discrete`. The service reports each dataset's
supported modes at `GET /v1/datasets`; CDS requires its key-action scene assets.
Images and training JSONL are separate from simulation scene assets.

RL responses use `pdm_score` for monitoring and keep discrete components in
`pdms_metrics`. The discrete progress reward is named `gated_progress` and uses
progress in meters × NC × DAC × TTC × HC. Per-step arrays are available through
`include_details=true`; scalar rewards keep full precision. HTTP or scoring
failures stop the workflow and do not become zero rewards or reference targets.

After setting `DATA_ROOT`, `SERVED_MODEL`, `VLLM_URL`, `SIM_URL` and `OUTPUT_DIR`,
set `EVAL_DATASET` to the evaluation collection registered on the scorer:

```bash
export EVAL_DATASET=navtest
python scripts/eval/eval_trajectory.py --metric pdms \
  --dataset-file eval.jsonl --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --dataset "$EVAL_DATASET" \
  --trajectory-decoder executable_planner --enable-thinking \
  --max-generation-attempts 3 --output "$OUTPUT_DIR/eval_pdms.json" \
  --details "$OUTPUT_DIR/eval_pdms_details.jsonl"
```

The command writes a summary report and per-sample predictions / scores. ADE and
FDE are also computed for parseable predictions with original GT trajectories;
decoded training references are excluded. This entrypoint requires the scoring
service even when inspecting its open-loop metrics.

For CDS, use `--metric cds`, a CDS-compatible input JSONL and its registered
scoring collection, and choose separate output filenames. The NAVSIM bundle is
not a release of the paper's CDS scene assets. See
[parse retries](active_rl.md#parse-retries) for generation budgets, zero scores
on exhausted attempts and the CDS validity filter.

## Multi-node execution

Run the same launch command on each node, setting `NNODES`, `NODE_RANK` (0-based),
`NPROC_PER_NODE`, `MASTER_ADDR` and `MASTER_PORT`. Supply models and datasets before
launch; use matching paths on every node. Standard torchrun variables identify
the ranks, with no cloud-specific environment detection.

SFT preserves checkpoint HTTP transfer between stages: rank 0 serves model files
on port 18900 and other nodes download missing files. Full training resume still
requires optimizer/scheduler/RNG state on every node; the model transfer does not
copy ZeRO optimizer directories. There is no automatic checkpoint backup.

For help without starting a distributed process, run either Python entrypoint
directly with `--help` after installing its dependencies. All imports are at the
top of each module. Training options are CLI arguments; the retired environment
variable interface (`DATASET=...`, `STAGE_CONFIG=...`, etc.) is not read by launchers.

File, JSON, inference and scoring-service errors propagate to the caller with
their traceback. Invalid model-generated trajectories remain an explicit scoring
outcome; they are distinct from an unavailable scoring service. Cross-node error
handling broadcasts failures with their traceback so every rank can exit.

## Development checks

```bash
python -m pip install -e ".[dev]"
ruff check src scripts
bash -n scripts/launch/sft.sh scripts/launch/grpo.sh
```
