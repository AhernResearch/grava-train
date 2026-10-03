"""Minimal ms-swift adapter for released messages/images JSONL datasets."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

from grava_train.data.dataset_utils import resolve_image_paths


class ReleasedDatasetPreprocessor(RowPreprocessor):
    """Resolve image paths before swift casts them to its image representation."""

    def __init__(self, data_root: Path):
        super().__init__()
        self.data_root = data_root

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any]:
        return resolve_image_paths(row, self.data_root)


def register_released_datasets(
    paths: Iterable[str], data_root: str | Path | None = None,
) -> None:
    """Register all input files before the first swift dataset load."""
    for value in dict.fromkeys(paths):
        path = Path(value).expanduser().resolve(strict=True)
        root = Path(data_root).expanduser().resolve() if data_root is not None else path.parent
        register_dataset(
            DatasetMeta(
                dataset_path=str(path),
                preprocess_func=ReleasedDatasetPreprocessor(root),
            ),
            exist_ok=True,
        )
