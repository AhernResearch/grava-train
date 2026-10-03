# Self-distillation

One round of rejection sampling followed by one round of SFT is self-distillation.
The model generates its own candidate answers; trajectory rewards select the SFT
targets. Each target keeps the think block and action from the same completion.

## Rejection sampling

Serve the action warm-up checkpoint at a vLLM multimodal chat endpoint and start
the scoring service with the training scenes registered. Use the same checkpoint
throughout the rollout. The input is the full `grpo_pool.jsonl`, without the
greedy or high-variance filters used by Active RL.

Each input row has one user message, image paths, a unique `id`,
`qa_type: "Trajectory"`, and the existing JSON-string `solution` field.
`solution` provides the scene metadata for scoring and `reference` for fallback.
The reference must contain the full supervised answer, including think and action.

After [installing the package](training.md#dependencies), set your paths and
the served model name:

```bash
export DATA_ROOT=/path/to/dataset
export WARM_UP_CHECKPOINT=/path/to/action-warm-up/checkpoint
export DISTILL_DIR=/path/to/outputs/self_distill
export SERVED_MODEL=grava
export VLLM_URL=http://localhost:8000/v1
export SIM_URL=http://localhost:8100

python scripts/rl/rejection_sampling.py \
  --dataset-file grpo_pool.jsonl --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --sim-dataset navtrain \
  --scoring-mode pdms --trajectory-decoder executable_planner \
  --num-rollouts 8 --temperature 0.7 --top-p 1.0 --max-tokens 4096 \
  --max-generation-attempts 3 --output "$DISTILL_DIR/self_distill.jsonl"
```

`SERVED_MODEL` must serve `WARM_UP_CHECKPOINT`. Replace `navtrain` with the training
collection registered on your scoring service.

The script calls `scripts/eval/rollout.py` with thinking enabled, then selects
one answer for every prompt:

- Candidates must be scored, have a parseable trajectory and contain a complete,
  nonempty `<think>...</think>` block. CDS also requires `sample_valid: true`.
- Select the candidate with the highest score. Ties use the smallest `rollout_idx`.
- If its score is positive, use its full `prediction` as the assistant answer.
- If the best score is zero or no candidate is valid, use `solution.reference`.
  For scoring modes that allow negative rewards, nonpositive best scores also
  use the reference. A missing or invalid reference stops the export.

Prompt text, IDs, image paths and row order are preserved. The output drops
`solution` and appends the selected answer as an assistant message. It does not
rewrite the think block, re-encode the action, or combine parts of different answers.
Relative image paths still use `DATA_ROOT` when loading the exported SFT file.

The default selection metric is PDMS. `--scoring-mode` also accepts `cds`,
`continuous` and `discrete`, using `cds` or `rl_score` from the rollout records.
`--trajectory-decoder direct_waypoints` uses direct-coordinate answers for both
candidate selection and reference validation.

## Output and resume

For the example above, the script writes:

| File | Contents |
| --- | --- |
| `self_distill.jsonl` | One trainable SFT record per input prompt |
| `self_distill.rollouts.jsonl` | All scored candidates, including invalid and zero-score outputs |
| `self_distill.stats.json` | Rollout/reference target counts, input paths and selection settings |

The rollout script also writes `self_distill.rollouts.runs.jsonl` with invocation
settings. Use `--rollouts PATH` to choose a different rollout file.

Add `--resume` to the generation command after an interrupted run. It fills
missing rollout indices, then rebuilds the SFT file. Keep the prompt dataset,
served checkpoint, scoring and generation settings unchanged when resuming.
Parsing retries follow the shared [rollout rules](active_rl.md#parse-retries).
Missing think content is rejected at selection time; it does not add another
generation budget.

To select from existing rollout files or shards without generating again:

```bash
python scripts/rl/rejection_sampling.py \
  --dataset-file grpo_pool.jsonl --data-root "$DATA_ROOT" \
  --select-only --rollouts "$DISTILL_DIR/self_distill.rollouts.jsonl" \
  --num-rollouts 8 --scoring-mode pdms --trajectory-decoder executable_planner \
  --output "$DISTILL_DIR/self_distill.jsonl"
```

The rollout files must cover every input ID with indices `0..K-1` exactly once.
Incomplete groups, duplicate indices, unscored records and service errors stop
selection. They do not trigger reference fallback. The SFT file is published
only after all rows have been exported successfully.

## SFT

Initialize SFT from the same action warm-up checkpoint used for sampling:

```bash
python scripts/train/train_sft.py \
  --model "$WARM_UP_CHECKPOINT" --model_type qwen3_vl \
  --dataset "$DISTILL_DIR/self_distill.jsonl" --data_root "$DATA_ROOT" \
  --output_dir "$DISTILL_DIR/checkpoints" \
  --num_train_epochs 1 --learning_rate 5e-6 --batch_size 4 \
  --gradient_accumulation_steps 1 --save_strategy epoch
```

For multiple GPUs, use the [SFT launcher](training.md#sft) with the same arguments.
The resulting checkpoint initializes the next GRPO or [Active RL](active_rl.md)
round. The rejection-sampling script only prepares the dataset; it does not launch SFT.
