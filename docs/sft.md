# SFT entrypoint and components

`scripts/train/train_sft.py` owns the stage loop and calls `sft_main` directly.
`src/grava_train/training/` contains independently usable components:

| Module | Responsibility |
| --- | --- |
| `data.py` | Register prepared JSONL and resolve relative image paths |
| `stages.py` | `StageSpec`, recipe parsing, input validation, resume-stage selection |
| `sft_args.py` | Pure translation of CLI defaults and stage overrides into swift arguments |
| `checkpoints.py` | Select the checkpoint returned by the completed training stage |

`utils/checkpoint_utils.py` validates model/adapter files and weight shards.
`utils/checkpoint_sync.py` provides HTTP checkpoint transfer between nodes. The stage
loop stays in the external script. All imports are at module scope, so even
`--help` requires the training dependencies. It does not start training.

## Inputs and parameter precedence

Specify exactly one of `--dataset` (one path or comma-separated paths) and
`--stage_config` (YAML/JSON file or inline JSON). Both become a list of stages.
Relative dataset paths resolve against `--data_root`, defaulting to the current
working directory. All listed files are checked before runtime initialization.
Dataset contents and trajectory answers pass through unchanged. `images` must
contain ordinary image file paths accessible on each node. Archive-member paths
such as `archive.tar.gz::image.jpg` are unsupported. Extract the released image
bundle beside the JSONL files; no dataset conversion is needed.

Each recipe stage requires `name` and `dataset`; `dataset` also accepts a list.
Optional `epochs`, `lr`, `batch_size` and `max_length` override CLI defaults.
Names must be unique and contain only letters, digits, underscores or hyphens.
A top-level `special_tokens` list is supported for explicit experiments and is
applied only to the first stage of a fresh run. Executable Planner needs no
additional tokenizer tokens.

Removed options: `--stages`, `--merge_stages`, `--stage_epochs` and the cache/backup options.
Replace numbered stages with recipe entries, and merged stages with one entry
whose `dataset` is a list. There is no implicit Stage12 split or filename search.
Training options are CLI arguments; the launch scripts only read torchrun topology variables.

Single-dataset runs save beneath `--output_dir`. Recipe runs use
`--output_dir/stage{index}_{name}`. ms-swift can add its own version directory.
Only rank 0 writes `training_progress.json`.

## Recorded recipes

These recipes preserve the dataset-relative paths and stage parameters recorded
in the project wiki. Historical filenames retain `hybrid_v6` for traceability;
the public output format is called **Executable Planner**.

| Recipe | Input model | Recorded data/run |
| --- | --- | --- |
| `configs/sft/pretrain.yaml` | Base Qwen3-VL model | Pretrain on combined Stage12 data, 1 epoch |
| `configs/sft/action_warm_up_sft.yaml` | Pretrain checkpoint | Teacher-supervised action warm-up, reconstructed R1 data, 2 epochs |

The retained action warm-up recipe uses teacher supervision. Its source path
was checked against real dev training data; the reconstructed teacher file is
distinct from the unavailable original file.
The YAML files describe stage inputs and overrides, not complete hardware recipes.
Pretrain was recorded on 4 nodes × 8 H20; the R1 run on 2 nodes × 8 H20, with
gradient accumulation 1. Changing world size changes effective batch size.

Public stage names are `pretrain` and `action_warm_up_sft`. Action warm-up starts
from the pretrain checkpoint. Teacher names and historical experiment IDs remain
provenance, not stage names. The mixed teacher/self-rollout R2 recipe is not part
of the retained mainline. [Self-distillation](self_distillation.md) runs one round
of rejection sampling followed by one round of SFT on the selected answers.

For example, after setting `MODEL`, `TRAIN_DATA_ROOT` and `OUTPUT_DIR` to your
pretrain checkpoint, prepared dataset root and output directory:

```bash
python scripts/train/train_sft.py \
  --model "$MODEL" --model_type qwen3_vl \
  --stage_config configs/sft/action_warm_up_sft.yaml --data_root "$TRAIN_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" --gradient_accumulation_steps 1 \
  --attn_impl flash_attention_2 --padding_free --deepspeed zero2 \
  --save_strategy epoch
```

For multiple GPUs, replace `python scripts/train/train_sft.py` with
`NPROC_PER_NODE=8 bash scripts/launch/sft.sh` and pass the same CLI options.
For multiple nodes, also set `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.
SFT uses ms-swift 4.x. No platform-specific launcher or legacy CLI bypass is used.
See [training setup](training.md) for the dependency snapshot and launch examples.

Sources in the workspace knowledge base:
`bbox-grounding-main-experiment.md`
(Stage12 training parameters) and
`bbox-grounding-rl-reproduction-20260622.md`
(R1 and the 2026-09-24 data audit).

## Model initialization and resume

`--model` initializes a fresh run. In a recipe, full fine-tuning passes the
previous stage's weights to the next stage; LoRA keeps the base model and passes
the previous adapter through swift's `adapters` argument. Each new stage gets
a fresh optimizer and scheduler. Checkpoints come from the current `sft_main`
result; the script never recursively selects an older run by step number.

`--resume_from_checkpoint` restores an interrupted run. For a multi-stage recipe,
also pass `--resume_stage NAME`: earlier stages are skipped, and the resume
parameter is applied only to that stage. Keep its original training settings;
`epochs` is the total target, not additional epochs. Subsequent stages initialize
from the resumed stage's output with fresh optimizer state.

Resume requires a checkpoint with weights and `trainer_state.json`; preserve all
optimizer/scheduler/RNG state required by the original backend on every node.
The cross-node weight transport does **not** copy ZeRO optimizer state and is
not a substitute for a full resumable checkpoint. A new progress
file describes stages completed in the current invocation.

Missing weights or shards fail explicitly. `--save_strategy no` is allowed only
when no following stage needs model artifacts.

## Images and checkpoint transfer

`--image_max_tokens` controls the per-image token limit, defaulting to
`IMAGE_MAX_TOKEN_NUM` or 1200. Reproduction runs must explicitly match the original
image setting (the recorded initial grounding run used 16384; the retired platform
launcher defaulted to 1175). The new launch scripts do not override it.

Released datasets use paths relative to `--data_root`, or the JSONL parent when
no root is supplied. A small swift adapter resolves paths before image loading.
See [released datasets](data.md). Images are loaded by the training framework.

At a multi-node stage transition, rank 0 temporarily serves the checkpoint over
HTTP (port 18900). Local rank 0 on each other node downloads missing model files;
all ranks observe transfer failures. Checkpoint paths must be identical across
nodes, and the rank-0 address (`MASTER_ADDR`) must be reachable on that port.
Shared complete checkpoints skip downloading. Single-node runs start no server.

## Validation

```bash
python scripts/train/train_sft.py --help
ruff check src scripts
bash -n scripts/launch/sft.sh
```

These checks do not start training or submit jobs.
