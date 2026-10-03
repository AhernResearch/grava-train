# Reward experiments

Public trajectory names are `executable_planner` (planner JSON) and
`direct_waypoints` (numeric waypoint text, including `{"trajectory": [...]}`
answers with optional heading). New dataset exports use `executable_planner`
in `trajectory_codec.name`; validation also accepts the historical `hybrid_v6`
metadata in existing training files. Planner parameter strings are unchanged.

The NAVSIM training default is `--reward_funcs pdms --reward_weights 1.0`.
Other rewards remain available to reproduce and compare driving reward designs.
Selecting several names computes their weighted combination through ms-swift.
Weights are experiment settings, not defaults for the paper's main result.

| Name | Signal |
| --- | --- |
| `pdms` | Official NAVSIM PDMS from the sim-engine scoring endpoint. |
| `cds` | Closed-loop Driving Score (CDS), as defined in the paper. |
| `progress` | Ego progress from the continuous RL evaluator. |
| `fde_progress` | Open-loop fidelity, `exp(-FDE / tau)`. |
| `pdms_fde` | Collision and drivable-area gates multiplied by FDE fidelity. |
| `safety` | Obstacle and boundary clearance, with collision penalties; discrete mode uses binary NC/DAC/TTC. |
| `comfort` | History comfort. |
| `ttc` | Time-to-collision score. |
| `safety_gate` | Binary safety gate from the discrete RL evaluator. |
| `gated_progress` | Progress in meters × NC × DAC × TTC × HC (discrete mode). |
| `pdms_monitor` | Official PDMS for logging; use weight zero to keep it out of optimization. |
| `nc` | Binary no-at-fault-collision reward. |
| `boundary` | Boundary clearance, saturated at 0.5 m. |
| `boundary_margin` | Signed boundary margin, scaled between -0.2 m and 0.5 m. |
| `chain_bonus` | Correctness-conditioned reasoning-mode bonus. |
| `gra_format` | Whether the final answer decodes as a supported trajectory. |

Decision samples retain exact/soft label matching where the reward supports them.
For trajectory samples, sim-engine rewards require `log_name` and `scene_token`
in `solution`. Executable Planner decoding also uses ego state from `solution.context`.
Open-loop fidelity rewards consume `solution.gt_trajectory`; its coordinates must
match the predicted coordinates. These fields come from the prepared dataset. Missing labels, scene fields and
service metrics raise errors; malformed generated trajectories keep their format
failure reward. Completion and solution batch lengths must match.

## CDS naming

CDS is the paper's **Closed-loop Driving Score**:

`CDS = Safety × (0.45 × KOC + 0.35 × Progress + 0.20 × Comfort)`.

Use `--reward_funcs cds --reward_weights 1.0` in GRPO, or
`scripts/eval/eval_trajectory.py --metric cds` for evaluation. Scored records store the scalar as
`cds`; evaluation summaries use the `cds` report section. The default output
files are `cds_report.json` and `cds_details.jsonl`. `key_action_score` in the
engine's submetrics corresponds to KOC.

`eval/sim_client.py` is the protocol boundary: the existing scoring engine still
uses its historical endpoint and score field. `DrivingScoreClient` exposes
`score` / `score_batch` with `scoring_mode="cds"` and renames the response field once, preserving
all numeric values. Missing required scores and service errors propagate; the
The client reads required metrics directly, without filling missing fields. The service itself is unchanged. Dataset
identifiers such as `internal` describe the data source, not the metric name.
Older exported score files must have their metric field renamed explicitly
before reading them with the new scored-record schema.

## Code organization

- `rewards/base.py`: shared trajectory/decision sample dispatch.
- `rewards/simulation.py`: simulator calls, extraction failure handling, and score caching.
- `rewards/driving.py`: driving reward formulas.
- `rewards/reasoning.py`: reasoning-mode and output-format rewards.
- `rewards/registry.py`: public reward names and ms-swift registration.
- `scripts/train/plugins/reward_plugin.py`: external plugin entrypoint.

`SIM_ENGINE_URL` and `SIM_ENGINE_DATASET` configure the scoring service.
`SIM_ENGINE_TRAJECTORY_FRAME` defaults to `grava` (x right, y forward).
`TRAJECTORY_DECODER` selects `direct_waypoints` or `executable_planner` explicitly;
both yield absolute waypoints. Planner decoding requires the ego state from the
rendered prompt. Outputs in the other format receive a format failure.
`REWARD_MODE=continuous|discrete` controls the `safety` formula.
`SAFETY_COLLISION_SIGNAL=overlap|penetration_distance` selects its collision signal.
Set these variables before loading the plugin.

## Output formats

Executable Planner JSON and direct numeric waypoint text are supported.
Planner parameter strings are ordinary text and require no tokenizer expansion.
Use `gra_format` to score the reasoning and answer structure.
