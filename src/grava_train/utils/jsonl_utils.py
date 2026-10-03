"""Read, write and validate UTF-8 JSONL files."""

import json
from pathlib import Path
from typing import Dict, Iterator, List


def save_jsonl(samples: List[Dict], path: Path) -> None:
    """Write records one per line, replacing the destination."""
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def load_jsonl(path: Path, flatten_lists: bool = False) -> List[Dict]:
    """Load non-empty lines; optionally flatten records stored as lists."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if flatten_lists and isinstance(obj, list):
                samples.extend(obj)
            else:
                samples.append(obj)
    return samples


def load_jsonl_as_lookup(path: Path, key: str = "id") -> Dict[str, Dict]:
    """Index records by a non-empty key; later duplicates replace earlier entries."""
    lookup = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = obj.get(key, "")
            if k:
                lookup[k] = obj
    return lookup


def append_jsonl(samples: List[Dict], path: Path) -> None:
    """Append records without reading existing content. Callers coordinate concurrent writers."""
    with open(path, 'a', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield records from non-empty lines without loading the entire file."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_samples(path: Path) -> int:
    """Count non-empty lines without parsing JSON."""
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def validate_jsonl(path: Path) -> tuple[bool, str]:
    """Check JSON syntax and report blank lines. File errors propagate."""
    empty_lines = []
    with path.open(encoding='utf-8') as stream:
        for i, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                empty_lines.append(i)
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                return False, f"第 {i} 行 JSON 解析失败: {e}"

    if empty_lines:
        return True, f"警告：存在 {len(empty_lines)} 个空行（行号：{empty_lines[:5]}...）"

    return True, "格式正确"
