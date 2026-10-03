"""
Cross-node checkpoint synchronization for multi-node training.

When using DeepSpeed ZeRO-2 with multi-node training, only the rank 0 node
saves the full checkpoint (config.json, safetensors, tokenizer, etc.). Other
nodes only save rng_state files. This causes stage transitions to fail because
ms-swift's AutoConfig.from_pretrained cannot find config.json on non-rank-0 nodes.

Solution: After stage training completes, rank 0 starts a temporary HTTP file
server and other nodes download the missing files via HTTP (local network).

Usage in train_sft.py:
    from grava_train.utils.checkpoint_sync import sync_checkpoint_across_nodes
    sync_checkpoint_across_nodes(checkpoint_path)
"""

import json
import os
import socket
import threading
import traceback
import urllib.request
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist

from grava_train.utils.checkpoint_utils import validate_checkpoint

# Files that rank 0 saves but other nodes don't (DeepSpeed ZeRO-2)
# Other nodes only have rng_state_*.pth files
_SKIP_PATTERNS = {"rng_state_", "global_step"}

_SYNC_PORT = 18900


class _QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses request logging."""

    def log_message(self, format, *args):
        pass


class _FileServer:
    """Temporary HTTP file server for a directory."""

    def __init__(self, directory: str, port: int = _SYNC_PORT):
        handler = partial(_QuietHandler, directory=directory)
        self._server = HTTPServer(("0.0.0.0", port), handler)
        self._server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()


def _get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "0")))


def _get_nnodes() -> int:
    if "NNODES" in os.environ:
        return int(os.environ["NNODES"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return world_size // local_world_size


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_master_addr() -> str:
    return os.environ.get("MASTER_ADDR", "127.0.0.1")


def _ensure_dist_initialized():
    """Initialize torch.distributed if not already done.

    torchrun sets env vars (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    but does not call init_process_group — the training framework does that
    later in sft_main. We need dist for broadcast, so init early if needed.
    """
    if dist.is_initialized():
        return
    local_rank = _get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def broadcast_string(s: str, src: int = 0) -> str:
    """Broadcast text with the active process group's device (CPU for Gloo)."""
    if not dist.is_initialized():
        return s
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    if dist.get_rank() == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device=device)
    else:
        length = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(length, src=src)

    if dist.get_rank() == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device=device)
    else:
        tensor = torch.zeros(length.item(), dtype=torch.uint8, device=device)
    dist.broadcast(tensor, src=src)

    return bytes(tensor.cpu().tolist()).decode("utf-8")


def _list_sync_files(checkpoint_dir: str) -> List[str]:
    """List files that need to be synced (excluding rng_state and global_step)."""
    files = []
    ckpt = Path(checkpoint_dir)
    for f in ckpt.iterdir():
        if f.is_dir():
            continue
        if any(f.name.startswith(pat) for pat in _SKIP_PATTERNS):
            continue
        files.append(f.name)
    return sorted(files)


def _needs_sync(checkpoint_dir: str) -> bool:
    """A config alone does not imply that all model or adapter weights arrived."""

    try:
        validate_checkpoint(checkpoint_dir)
    except FileNotFoundError:
        return True
    return False


def _download_files(base_url: str, file_list: List[str], dest_dir: str) -> None:
    """Download files from HTTP server to local directory."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    total = len(file_list)

    for i, filename in enumerate(file_list):
        url = f"{base_url}/{filename}"
        local_path = dest / filename
        if local_path.is_file() and local_path.stat().st_size > 0:
            continue
        temporary = local_path.with_name(local_path.name + ".download")
        urllib.request.urlretrieve(url, str(temporary))
        temporary.replace(local_path)
        size_mb = local_path.stat().st_size / (1024 * 1024)
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"  [{i + 1}/{total}] {filename} ({size_mb:.1f} MB)")


def sync_checkpoint_across_nodes(checkpoint_path: str, port: int = _SYNC_PORT) -> None:
    """
    Synchronize checkpoint files from rank 0 node to all other nodes.

    Uses standard torchrun environment variables. No-op for single-node training.

    Args:
        checkpoint_path: Path to the checkpoint directory (same on all nodes).
        port: HTTP server port for file transfer.
    """
    nnodes = _get_nnodes()
    if nnodes <= 1:
        return

    # Ensure dist is initialized before using broadcast/barrier.
    # torchrun sets env vars but init_process_group happens later in sft_main.
    _ensure_dist_initialized()

    node_rank = _get_node_rank()
    local_rank = _get_local_rank()
    master_addr = _get_master_addr()

    server = None
    manifest = {}
    if local_rank == 0 and node_rank == 0:
        try:
            manifest["files"] = _list_sync_files(checkpoint_path)
            server = _FileServer(checkpoint_path, port)
            server.start()
        except Exception:
            manifest["error"] = traceback.format_exc()

    manifest = json.loads(broadcast_string(json.dumps(manifest)))
    if "error" in manifest:
        raise RuntimeError(f"Checkpoint server failed: {manifest['error']}")

    try:
        error = None
        if local_rank == 0 and node_rank != 0:
            try:
                if _needs_sync(checkpoint_path):
                    _download_files(
                        f"http://{master_addr}:{port}", manifest["files"], checkpoint_path
                    )

                validate_checkpoint(checkpoint_path)
            except Exception:
                error = f"Node {node_rank}:\n{traceback.format_exc()}"
        # All ranks observe a failed transfer, instead of waiting forever at a barrier.
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        failures = [message for message in errors if message]
        if failures:
            raise RuntimeError("Checkpoint sync failed: " + "; ".join(failures))
    finally:
        if server is not None:
            server.stop()
