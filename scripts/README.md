# Scripts

| Directory | Purpose |
| --- | --- |
| `launch/` | Small, standard torchrun launchers for SFT and GRPO. |
| `train/` | Training workflows and the ms-swift reward registration plugin. |
| `rl/` | Active RL selection and rejection-sampled SFT targets. |
| `eval/` | Single-turn evaluation and rollout entrypoints. |

Use `launch/sft.sh` or `launch/grpo.sh` with the same CLI options as the
corresponding Python entrypoint. Set torchrun topology through `NNODES`,
`NODE_RANK`, `NPROC_PER_NODE`, `MASTER_ADDR` and `MASTER_PORT`.

Training and evaluation read ordinary front-camera image files. They support
Executable Planner JSON and direct waypoint answers. There is no archive reader,
image prefetch, automatic backup or cloud-specific setup. SFT retains HTTP
checkpoint transfer for multi-node stage transitions.

SFT stage names are `pretrain` and `action_warm_up_sft`. Driving metrics are PDMS and
CDS (Closed-loop Driving Score); use `eval/eval_trajectory.py --metric cds` for CDS evaluation and
`--reward_funcs cds` for its training reward.

See [training setup](../docs/training.md), [SFT components](../docs/sft.md),
[reward experiments](../docs/rewards.md) and [released datasets](../docs/data.md).

See [Active RL rounds](../docs/active_rl.md) for greedy scan, sampling and selection.
See [self-distillation](../docs/self_distillation.md) for one rejection-sampling
round with `rl/rejection_sampling.py`, followed by one SFT round.
