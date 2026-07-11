"""Hashing helpers for reproducible runs and manifest tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(data: Any) -> str:
    """Return the SHA-256 digest of canonical JSON data."""

    normalized = _to_jsonable(data)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_hash(manifest: str | Path | Iterable[Mapping[str, Any]]) -> str:
    """Return a stable hash for a JSONL manifest path or manifest rows."""

    if isinstance(manifest, str | Path):
        rows = _read_jsonl_for_hash(Path(manifest))
        return json_sha256(rows)
    return json_sha256(list(manifest))


def _read_jsonl_for_hash(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                msg = f"Expected JSON object in {path} at line {line_number}"
                raise ValueError(msg)
            rows.append(row)
    return rows


def _to_jsonable(data: Any) -> Any:
    if is_dataclass(data) and not isinstance(data, type):
        return _to_jsonable(asdict(data))
    if isinstance(data, Mapping):
        return {str(key): _to_jsonable(value) for key, value in data.items()}
    if isinstance(data, tuple | list):
        return [_to_jsonable(item) for item in data]
    return data
