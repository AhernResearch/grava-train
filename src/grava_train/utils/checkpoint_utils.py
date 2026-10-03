"""Validate model artifacts without loading their tensors or scanning old runs."""

import json
from pathlib import Path


def checkpoint_weight_files(path: str | Path) -> list[Path]:
    """Find full-model or adapter weights, including every shard named by an index."""
    directory = Path(path)
    adapter = (directory / "adapter_config.json").is_file()
    config = directory / ("adapter_config.json" if adapter else "config.json")
    config.stat()
    names = (
        ("adapter_model.safetensors", "adapter_model.bin")
        if adapter
        else (
            "model.safetensors",
            "pytorch_model.bin",
        )
    )
    for name in names:
        weight = directory / name
        index = directory / f"{name}.index.json"
        if weight.is_file() and weight.stat().st_size > 0:
            return [weight]
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not weight_map:
                raise ValueError(f"Empty or invalid weight index: {index}")
            if any(Path(s).name != s for s in weight_map.values()):
                raise ValueError(f"Invalid shard filename in {index}")
            shards = sorted(set(weight_map.values()))
            files = [directory / s for s in shards]
            for shard in files:
                if shard.stat().st_size == 0:
                    raise ValueError(f"Empty checkpoint shard: {shard}")
            return [index, *files]
    raise FileNotFoundError(f"No model weights in {directory}")


def validate_checkpoint(path: str, *, resume: bool = False) -> str:
    """Require loadable weight files; resume additionally needs trainer state."""
    checkpoint_weight_files(path)
    if resume:
        (Path(path) / "trainer_state.json").stat()
    return str(Path(path).resolve())
