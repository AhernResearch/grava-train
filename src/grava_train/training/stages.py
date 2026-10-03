"""Parse explicit SFT stages without loading a model or dataset."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StageSpec:
    """One training call; unset overrides inherit the command-line defaults."""

    name: str
    dataset: tuple[str, ...]
    epochs: float | None = None
    lr: float | None = None
    batch_size: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(
                "Stage names must contain only letters, digits, underscores or hyphens"
            )
        for field in ("epochs", "lr", "batch_size", "max_length"):
            value = getattr(self, field)
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{self.name}: {field} must be a positive finite number")
            if field in ("batch_size", "max_length") and not isinstance(value, int):
                raise ValueError(f"{self.name}: {field} must be an integer")


def load_stages(
    *, dataset: str | None = None, stage_config: str | None = None, data_dir: str = "."
) -> tuple[list[StageSpec], tuple[str, ...]]:
    """Normalize a dataset or YAML/JSON recipe and check every input file exists.

    Relative dataset paths are resolved against data_dir, never guessed from a stage name.
    Recipe lists and {stages: [...], special_tokens: [...]} use the existing config schema.
    """
    raw = [{"name": "sft", "dataset": dataset}]
    if stage_config:
        source = stage_config.strip()
        if source.startswith(("[", "{")):
            raw = json.loads(source)
        else:
            path = Path(source)
            text = path.read_text()
            if path.suffix.lower() in (".yaml", ".yml"):
                raw = yaml.safe_load(text)
            else:
                raw = json.loads(text)

    tokens = []
    if isinstance(raw, dict):
        unknown = raw.keys() - {"stages", "special_tokens"}
        if unknown:
            raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
        tokens = raw.get("special_tokens") or []
        raw = raw["stages"]
    if not isinstance(tokens, list) or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("special_tokens must be a list of non-empty strings")
    if not isinstance(raw, list) or not raw:
        raise ValueError("stage_config must contain a non-empty list of stages")

    stages = []
    for item in raw:
        values = dict(item)
        paths = values.pop("dataset")
        if isinstance(paths, str):
            paths = [p.strip() for p in paths.split(",")]
        resolved = []
        for value in paths:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(data_dir).expanduser() / path
            with path.open("rb"):
                resolved.append(str(path.resolve()))
        # PyYAML can parse scientific notation such as 1e-5 as a string.
        if isinstance(values.get("lr"), str):
            values["lr"] = float(values["lr"])
        stages.append(StageSpec(dataset=tuple(resolved), **values))
    if len({s.name for s in stages}) != len(stages):
        raise ValueError("Stage names must be unique")
    return stages, tuple(tokens)


def resume_stage_index(
    stages: list[StageSpec], checkpoint: str | None, stage_name: str | None
) -> int:
    """Select the interrupted stage; earlier stages are not rerun."""
    if not checkpoint:
        if stage_name:
            raise ValueError("resume_stage requires resume_from_checkpoint")
        return 0
    if stage_name is None:
        if len(stages) != 1:
            raise ValueError("Multi-stage resume requires --resume_stage NAME")
        return 0
    return [stage.name for stage in stages].index(stage_name)
