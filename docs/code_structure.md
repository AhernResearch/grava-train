# Code structure

Workflows and command-line options live in `scripts/`. Python modules in
`src/grava_train/` provide the components those workflows compose.

```text
src/grava_train/
├── data/
│   └── dataset_utils.py
├── inference/
│   ├── transformers_client.py
│   └── vllm_client.py
├── trajectory/
│   ├── coordinate_utils.py
│   ├── trajectory_codec.py
│   ├── trajectory_parser.py
│   └── metrics_openloop.py
├── eval/
│   ├── data_utils.py
│   ├── sim_client.py
│   ├── scored_results.py
│   └── scoring/
│       ├── sim_reward_calculator.py
│       └── scoring_pipeline.py
├── rewards/
├── training/
├── rl/
│   ├── selection.py
│   └── rejection_sampling.py
└── utils/
    ├── checkpoint_sync.py
    ├── checkpoint_utils.py
    └── jsonl_utils.py
```

`data/dataset_utils.py` reads released records and resolves ordinary image paths.
`training/data.py` adapts these records to ms-swift. `eval/data_utils.py` extracts
evaluation scene metadata and original GT. `utils/jsonl_utils.py` provides generic
JSONL operations without depending on any training or evaluation schema.

`trajectory/trajectory_parser.py` owns text extraction, ego-context parsing and the
explicit output-format selection. Training rewards and evaluation share these
functions and the metrics in `trajectory/metrics_openloop.py`.

`eval/sim_client.py` calls the scoring engine over HTTP without a server-package
dependency. `trajectory/coordinate_utils.py` converts GRAVA coordinates to nuPlan.
The scorer owns the prediction coordinate frame; the pipeline reads it there.
The simulation calculator handles individual and batched predictions; `scoring_pipeline.py` combines decoding,
scoring and open-loop metrics, sharing one decoded trajectory per prediction.
`scored_results.py` contains result records, result JSONL readers/writers and report
aggregation. Package initializers stay lightweight.

Training reward formulas and reasoning helpers live in `rewards/`. Existing reward
names and coefficients are unchanged. `utils/checkpoint_utils.py` handles file and
shard validation; `training/checkpoints.py` selects the checkpoint returned by a
training stage. HTTP transfer stays in `utils/checkpoint_sync.py`.

The evaluation entrypoints are:

- `scripts/eval/eval_trajectory.py --metric pdms`: one greedy prediction per case;
  the default scoring dataset is `navtest`.
- `scripts/eval/eval_trajectory.py --metric cds`: the same workflow with CDS scoring;
  the default scoring dataset is `internal`, and score aggregation requires valid
  samples as before.
- `scripts/eval/rollout.py`: multiple samples, inference-only / scoring-only modes
  and missing-index resume, shared by evaluation, Active RL and rejection sampling.

Generation retry loops live in the eval and rollout scripts. Scoring components
parse and score a supplied output; they do not invoke the model.

SFT, GRPO and Active RL selection remain in their existing external scripts.
`scripts/rl/rejection_sampling.py` calls rollout and exports SFT targets for
self-distillation. `rl/rejection_sampling.py` selects the best complete answer
and constructs a training row, using the supervised reference at zero reward.

`inference/vllm_client.py` builds image requests and performs one inference call.
Concurrency and repeated sampling belong to the external scripts.
`inference/transformers_client.py` provides local single/batched inference with
one worker per selected device. Its async calls offload generation to a thread
and serialize access to each worker. This backend is not wired into the public
eval CLI; it remains available as a component.
