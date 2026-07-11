"""Small file IO helpers used across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries.

    Blank lines are ignored. Parse errors include the failing line number so
    malformed submissions or manifests are easy to diagnose.
    """

    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {jsonl_path} at line {line_number}: {exc}"
                raise ValueError(msg) from exc
            if not isinstance(row, dict):
                msg = f"Expected JSON object in {jsonl_path} at line {line_number}"
                raise ValueError(msg)
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""

    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary."""

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Expected YAML mapping at top level in {yaml_path}"
        raise ValueError(msg)
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dictionary as pretty JSON."""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
