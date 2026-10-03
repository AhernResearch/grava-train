"""ms-swift plugin entrypoint; reward implementations live in grava_train.rewards."""

from grava_train.rewards.registry import register_rewards

register_rewards()
