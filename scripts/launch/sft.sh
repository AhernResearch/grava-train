#!/usr/bin/env bash
# Pass all SFT options as CLI arguments; set torchrun topology in the environment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec torchrun \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --nproc_per_node="${NPROC_PER_NODE:-1}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    "$SCRIPT_DIR/../train/train_sft.py" "$@"
