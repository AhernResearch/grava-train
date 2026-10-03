# Active RL: rollout, select, train

The released data separates the initial training set from the pool scanned by
subsequent policies:

- `grpo_initial.jsonl`: historical first GRPO input, 14,938 narrow-curve cases.
- `grpo_pool.jsonl`: the full canonical candidate pool for fresh on-policy selection.

The former initializes the historical sequence after the corresponding SFT /
[self-distillation](self_distillation.md) checkpoint. Its original selection used CURVE cases with
reference DAC >= 0.999, reference PDMS >= 0.8, boundary margin < 0.5, trajectory
length/progress >= 3 and absolute heading change <= 150 degrees. The prepared
initial JSONL is released directly; users need not repeat annotation processing.

Later round datasets are generated locally from the policy being trained. Each
round scans the full pool again, so cases excluded in one round can enter later.
The source pool already contains user messages, image paths, scene metadata and
reward references. Selection copies its rows byte-for-byte, in source order.
Sampled responses determine selection scores and never replace training labels.

## One round

Set `DATA_ROOT` to the released dataset directory, `ROUND_DIR` to a new output
directory for this round, and serve the selected checkpoint at `VLLM_URL`.
Keep the same served checkpoint for both rollout passes.

```bash
mkdir -p "$ROUND_DIR"

# 1. Greedy scan over the full pool.
python scripts/eval/rollout.py \
  --dataset-file grpo_pool.jsonl --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --sim-dataset navtrain \
  --trajectory-decoder executable_planner --enable-thinking \
  --num-rollouts 1 --temperature 0 --top-p 1 --max-tokens 4096 \
  --output "$ROUND_DIR/greedy.jsonl"

# 2. Retain greedy score < 0.9, exporting the corresponding pool rows.
python scripts/rl/select_rollouts.py \
  --dataset-file grpo_pool.jsonl --data-root "$DATA_ROOT" \
  --rollouts "$ROUND_DIR/greedy.jsonl" --rule greedy \
  --output "$ROUND_DIR/candidates.jsonl"

# 3. Sample eight predictions per retained case.
python scripts/eval/rollout.py \
  --dataset-file "$ROUND_DIR/candidates.jsonl" --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --sim-dataset navtrain \
  --trajectory-decoder executable_planner --enable-thinking \
  --num-rollouts 8 --temperature 0.8 --top-p 0.95 --max-tokens 4096 \
  --output "$ROUND_DIR/sampled.jsonl"

# 4. Retain recoverable high-variance cases and export this round's GRPO input.
python scripts/rl/select_rollouts.py \
  --dataset-file grpo_pool.jsonl --data-root "$DATA_ROOT" \
  --rollouts "$ROUND_DIR/sampled.jsonl" --rule high_variance \
  --num-rollouts 8 --min-score-lt 0.8 --max-score-ge 0.9 --std-ge 0.1 \
  --output "$ROUND_DIR/grpo.jsonl"

# 5. Train from the checkpoint used for this round's rollout.
python scripts/train/train_grpo.py \
  --model "$ROUND_CHECKPOINT" --dataset "$ROUND_DIR/grpo.jsonl" \
  --data_root "$DATA_ROOT" --output_dir "$ROUND_DIR/checkpoints" \
  --trajectory_decoder executable_planner --reward_funcs pdms --no_vllm
```

Set `ROUND_DIR` to an absolute path so the generated dataset path remains
unambiguous when combined with `--data_root`. Training options above only show
data wiring; set optimizer, generation and distributed parameters explicitly.
Evaluation and selection do not submit training jobs. After evaluating the next
checkpoint, use a new round directory and repeat from the full pool.

## Rule and artifacts

The default high-variance rule uses eight complete samples per case:

```text
minimum score < 0.8
maximum score >= 0.9
population standard deviation >= 0.10   # ddof=0
```

The earlier wide bucket omitted the standard-deviation threshold; it can be
represented with `--std-ge 0`. The earlier Curious exact-filter experiment used a
different rule, not implemented by this high-variance selector. The public default
reproduces the later mainline rule rather than claiming all historical buckets
used the same algorithm.

Each selection writes the JSONL, `.stats.json`, `.case_stats.jsonl` and
`.manifest.json`. The manifest records source and rollout paths, rule thresholds,
sample counts and the image-path root. Rollout `.runs.jsonl` files record each
invocation, generation options and the input dataset path. Rollout shards can be passed together to
`--rollouts`; duplicate indices, incomplete groups, unscored rows and scoring
service errors fail explicitly. Invalid generated trajectories retain their
recorded zero reward as policy failures. Greedy selection requires coverage of
the full supplied pool.

`rollout.py --resume` fills missing rollout indices. `--infer-only` followed by
`--score-only` is supported. New records distinguish `scored` from reasoning mode:
a valid zero score or a response without a think block is still a scored result.
`scored` and `pdm_score` are required fields. Historical files must be normalized
explicitly before use; the public reader does not infer status or rename metrics.

The historical 4,676-case intermediate dataset is kept only as private audit
material. It is not the public first-round training input.

Validation against archived mainline rollouts recovered exactly the historical
4,676 selected IDs. That audit explicitly excluded 114 cases whose eight scores
were recorded as errors; none belonged to the selected set. Public selection
fails on such scoring errors and requires them to be resolved before selection.

## Parse retries

`rollout.py` and `eval_trajectory.py` accept `--max-generation-attempts N`
(default: 3, including the first generation). Set it to 1 for single-attempt
inference. Each attempt uses the same prompt and generation settings. A parse
failure requests a new model output; a valid trajectory stops retries even if
its driving score is zero. Greedy generation may produce the same invalid
answer again; retries do not change temperature or repair the answer text.

Rollout retries only failed indices and keeps the first parseable output for
each index. Exhausted outputs remain in the dataset with zero for the selected
metric (`pdm_score`, `cds` or `rl_score`). They count in evaluation means and in
Active RL selection. Service, file and configuration errors propagate immediately.

Records store `generation_attempts` and, on final parse failure, `parse_error`.
`error` is reserved for execution/service errors. Evaluation reports include
`parse_failures`, total `generation_attempts` and the configured generation budget.
For CDS, exhausted generations count as zero even without a simulator validity
result; successfully scored samples still follow the `sample_valid` filter.

`--infer-only` applies the generation budget before saving predictions.
`--score-only` scores the saved text and preserves its attempt count; it does not
regenerate. `--resume` keeps completed indices, including exhausted ones, without
granting an additional retry budget. GRPO online training is unchanged.
