"""Audit official Track 1 submission JSONL files before upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from humomni.infer.make_submission import DEFAULT_ANSWER_CHOICES
from humomni.utils.io import read_jsonl, write_json

ALLOWED_FIELDS = frozenset({"question_id", "answer"})
LEAKAGE_TERMS = ("score", "label", "path", "goodpara", "badpara", "gold")


def audit_submission_file(
    *,
    submission_path: str | Path,
    expected_count: int | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a submission file and return a structured report."""

    try:
        rows = read_jsonl(submission_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "errors": [f"invalid jsonl: {exc}"],
            "num_rows": 0,
            "num_unique_question_ids": 0,
        }

    manifest_rows = read_jsonl(manifest_path) if manifest_path is not None else None
    return audit_submission_rows(
        rows,
        expected_count=expected_count,
        manifest_rows=manifest_rows,
    )


def audit_submission_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    manifest_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate official submission rows."""

    errors: list[str] = []
    manifest_by_question_id = _manifest_by_question_id(manifest_rows or [])
    if expected_count is None and manifest_rows is not None:
        expected_count = len(manifest_rows)

    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"row count mismatch: expected {expected_count}, got {len(rows)}")

    seen_question_ids: set[Any] = set()
    for row_index, row in enumerate(rows):
        row_keys = set(row)
        extra_fields = row_keys - ALLOWED_FIELDS
        missing_fields = ALLOWED_FIELDS - row_keys
        if extra_fields:
            errors.append(f"row {row_index}: unexpected fields {sorted(extra_fields)}")
        if missing_fields:
            errors.append(f"row {row_index}: missing fields {sorted(missing_fields)}")

        leakage_hits = _leakage_hits(row)
        if leakage_hits:
            errors.append(f"row {row_index}: leakage fields {leakage_hits}")

        question_id = row.get("question_id")
        if not _valid_question_id_type(question_id, manifest_by_question_id):
            errors.append(f"row {row_index}: question_id must be an integer or manifest string id")
        elif question_id in seen_question_ids:
            errors.append(f"duplicate question_id: {question_id}")
        else:
            seen_question_ids.add(question_id)

        answer = row.get("answer")
        if not isinstance(answer, str) or not answer:
            errors.append(f"row {row_index}: answer must be a non-empty string")
            continue
        legal_answers = _legal_answers(question_id, manifest_by_question_id)
        if answer not in legal_answers:
            errors.append(f"row {row_index}: illegal answer {answer!r}")

    if manifest_rows is not None:
        manifest_ids = set(manifest_by_question_id)
        missing_ids = manifest_ids - seen_question_ids
        extra_ids = seen_question_ids - manifest_ids
        if missing_ids:
            errors.append(f"missing question_id values: {sorted(missing_ids)[:8]}")
        if extra_ids:
            errors.append(f"unknown question_id values: {sorted(extra_ids)[:8]}")

    return {
        "passed": not errors,
        "errors": errors,
        "num_rows": len(rows),
        "num_unique_question_ids": len(seen_question_ids),
    }


def _manifest_by_question_id(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[Any, Mapping[str, Any]]:
    result: dict[Any, Mapping[str, Any]] = {}
    for row in manifest_rows:
        question_id = row.get("question_id")
        if question_id in result:
            raise ValueError(f"duplicate question_id in manifest: {question_id}")
        result[question_id] = row
    return result


def _legal_answers(
    question_id: Any,
    manifest_by_question_id: Mapping[Any, Mapping[str, Any]],
) -> set[str]:
    manifest_row = manifest_by_question_id.get(question_id)
    if isinstance(manifest_row, Mapping):
        candidate_audio_paths = manifest_row.get("candidate_audio_paths")
        if isinstance(candidate_audio_paths, Mapping) and candidate_audio_paths:
            return {str(key) for key in candidate_audio_paths}
    return set(DEFAULT_ANSWER_CHOICES)


def _valid_question_id_type(
    question_id: Any,
    manifest_by_question_id: Mapping[Any, Mapping[str, Any]],
) -> bool:
    if isinstance(question_id, bool):
        return False
    if isinstance(question_id, int):
        return True
    return isinstance(question_id, str) and bool(question_id) and question_id in manifest_by_question_id


def _leakage_hits(value: Any, *, path: str = "row") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_text not in ALLOWED_FIELDS and any(
                term in key_lower for term in LEAKAGE_TERMS
            ):
                hits.append(f"{path}.{key_text}")
            hits.extend(_leakage_hits(child, path=f"{path}.{key_text}"))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            hits.extend(_leakage_hits(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "goodpara" in lowered or "badpara" in lowered:
            hits.append(path)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Submission JSONL path.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Expected number of submission rows.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest JSONL used for row count and legal answer keys.",
    )
    parser.add_argument(
        "--report",
        default="audit_report.json",
        help="Audit report JSON path.",
    )
    args = parser.parse_args()

    report = audit_submission_file(
        submission_path=args.submission,
        expected_count=args.expected_count,
        manifest_path=args.manifest,
    )
    write_json(args.report, report)
    print(report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
