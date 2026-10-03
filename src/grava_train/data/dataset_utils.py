"""Read released JSONL records and resolve ordinary image files."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def resolve_image_paths(sample: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Return a record with absolute image paths, without changing its labels."""
    return {**sample, "images": [str((data_root / path).resolve()) for path in sample["images"]]}


def read_jsonl(path: str | Path, data_root: str | Path | None = None) -> Iterator[dict]:
    """Relative images use the explicit data root or the JSONL's parent directory."""
    path = Path(path).expanduser()
    if data_root is not None:
        path = Path(data_root).expanduser() / path
    path = path.resolve()
    root = Path(data_root).expanduser().resolve() if data_root is not None else path.parent
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield resolve_image_paths(json.loads(line), root)
