# Prepared datasets

The bundle is prepared locally; a public download URL has not yet been published.
Follow [GR-NavSim](https://github.com/AhernResearch/gr-navsim) for the data release.
Once available, download the bundle and extract the images beside the JSONL files.
The initial training sets need no annotation conversion, QA generation or
trajectory encoding. Later Active RL rounds select training rows using the
current policy, as described below.

```text
dataset/
├── README.md
├── manifest.json
├── pretrain.jsonl
├── action_warm_up_sft.jsonl
├── grpo_initial.jsonl
├── grpo_pool.jsonl
├── eval.jsonl
└── navsim_image/<log_name>/CAM_F0/<image_token>.jpg
```

| File | Rows | Use |
| --- | ---: | --- |
| `pretrain.jsonl` | 1,185,412 | Combined scene/object QA pretraining. |
| `action_warm_up_sft.jsonl` | 62,111 | Teacher-supervised action warm-up after pretraining. |
| `grpo_initial.jsonl` | 14,938 | Historical first GRPO training set after the corresponding SFT checkpoint. |
| `grpo_pool.jsonl` | 102,861 | Full candidate pool for each new policy rollout and selection round. |
| `eval.jsonl` | 12,146 | NAVSIM navtest prompts and original GT trajectories. |

This bundle contains the executable-planner experiment. The code also supports
`direct_waypoints`; a separately verified direct-waypoint training dataset is not
included in this bundle. Subsequent GRPO datasets are generated with the public
[rollout and selection workflow](active_rl.md); intermediate round datasets are
not included as fixed public training inputs.

## Paths and labels

Each record contains `id`, `qa_type`, `messages` and `images`. Images are ordinary
JPEG files; only the current front view is included. Motion history, where used,
is already rendered in the user prompt. All image references are relative to the
dataset root, for example `navsim_image/<log_name>/CAM_F0/<image_token>.jpg`.

Training accepts `--data_root`; evaluation accepts `--data-root`. Relative dataset
and image paths resolve against that directory. If no root is supplied, provide a
dataset path and its images resolve relative to the JSONL's parent. Absolute image
paths also work. The adapter resolves paths in memory without rewriting JSONL or
changing the working directory. Missing files and malformed records fail normally;
training enables swift's strict dataset loading.

SFT messages contain user and assistant turns. GRPO and eval messages contain only
the user turn. `solution` remains a JSON string for the ms-swift reward interface.
Training messages and reference answers retain their original text.

Executable-planner outputs use x-right/y-forward coordinates and eight future
points at 0.5-second intervals. Dataset metadata calls this format
`executable_planner`; its parameters are ordinary text tokens.

`solution.gt_trajectory_frame` and `solution.gt_trajectory_source` describe the
stored trajectory:

- Eval: `nuplan` (x-forward/y-left), `original`; used for ADE/FDE.
- GRPO pool: `grava` (x-right/y-forward), `decoded_reference`; retained for
  training rewards, excluded from open-loop GT evaluation.
- Initial GRPO: the original trajectory field is empty (`unavailable`); the
  reference answer and scene identifiers are preserved for simulation rewards.

SFT assistant answers are not used as original GT. Scene metadata is read from
`solution.log_name` and `solution.scene_token` for scoring.

## Run

From the code repository, after installing its dependencies:

```bash
python scripts/train/train_sft.py \
  --model "$MODEL" --stage_config configs/sft/pretrain.yaml \
  --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR"

python scripts/train/train_sft.py \
  --model "$PRETRAIN_CHECKPOINT" --stage_config configs/sft/action_warm_up_sft.yaml \
  --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR"

python scripts/train/train_grpo.py \
  --model "$GRPO_INIT_CHECKPOINT" --dataset grpo_initial.jsonl \
  --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" \
  --trajectory_decoder executable_planner --reward_funcs pdms --no_vllm

python scripts/eval/eval_trajectory.py --metric pdms \
  --dataset-file eval.jsonl --data-root "$DATA_ROOT" \
  --model "$SERVED_MODEL" --vllm-url "$VLLM_URL" \
  --sim-url "$SIM_URL" --trajectory-decoder executable_planner --enable-thinking
```

Evaluation defaults to at most three generation attempts on parse failure. Use
`--max-generation-attempts 1` for single-attempt results. See
[parse retries](active_rl.md#parse-retries) for zero-score and reporting semantics.

Match the original checkpoint and optimizer/generation settings for the specific
experiment; these commands illustrate data wiring. See [SFT settings](sft.md) and
[training setup](training.md).

PDMS/CDS and simulation-based GRPO rewards require the scoring service and its
scene assets/metric caches in addition to JSONL and images. Those assets are not
included in the image bundle. Evaluation uses multimodal chat requests; it does
not require a local copy of the served model's processor.

`manifest.json` records sample counts, file sizes and image coverage.

The pretrain source contains repeated IDs: 1,185,412 records and 1,158,488
unique IDs. Original rows and their ordering are preserved; release packaging
does not deduplicate or change sampling weights. All other files have unique IDs.
