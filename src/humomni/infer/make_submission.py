"""Create official Track 1 submission JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from humomni.utils.io import read_jsonl, write_jsonl

DEFAULT_ANSWER_CHOICES = ("A", "B", "C", "D")
SCORE_FIELDS = ("candidate_scores", "scores")


def make_submission_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_rows: Sequence[Mapping[str, Any]] | None = None,
    default_answer_choices: Sequence[str] = DEFAULT_ANSWER_CHOICES,
) -> list[dict[str, Any]]:
    """Convert rich prediction rows to official two-field submission rows."""

    manifest_by_question_id = _manifest_by_question_id(manifest_rows or [])
    output_rows: list[dict[str, Any]] = []
    seen_question_ids: set[Any] = set()

    for row_index, row in enumerate(prediction_rows):
        question_id = row.get("question_id")
        if question_id in seen_question_ids:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen_question_ids.add(question_id)

        candidate_keys = _candidate_keys(
            row,
            manifest_by_question_id.get(question_id),
            default_answer_choices=default_answer_choices,
        )
        answer = row.get("answer")
        if answer is None:
            answer = _answer_from_scores(row)
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"row {row_index}: answer is required")
        if answer not in candidate_keys:
            raise ValueError(
                f"row {row_index}: answer {answer!r} is not a legal candidate key"
            )
        output_rows.append({"question_id": question_id, "answer": answer})
    return output_rows


def make_submission_file(
    *,
    predictions_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    prediction_rows = read_jsonl(predictions_path)
    manifest_rows = read_jsonl(manifest_path) if manifest_path is not None else None
    submission_rows = make_submission_rows(
        prediction_rows,
        manifest_rows=manifest_rows,
    )
    write_jsonl(output_path, submission_rows)
    return submission_rows


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


def _candidate_keys(
    prediction_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any] | None,
    *,
    default_answer_choices: Sequence[str],
) -> set[str]:
    for container in (manifest_row, prediction_row):
        if not isinstance(container, Mapping):
            continue
        candidate_audio_paths = container.get("candidate_audio_paths")
        if isinstance(candidate_audio_paths, Mapping) and candidate_audio_paths:
            return {str(key) for key in candidate_audio_paths}
    for score_field in SCORE_FIELDS:
        scores = prediction_row.get(score_field)
        if isinstance(scores, Mapping) and scores:
            return {str(key) for key in scores}
    return set(default_answer_choices)


def _answer_from_scores(row: Mapping[str, Any]) -> str | None:
    for score_field in SCORE_FIELDS:
        scores = row.get(score_field)
        if isinstance(scores, Mapping) and scores:
            return max(scores, key=lambda key: float(scores[key]))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="Input predictions_with_scores JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="submission.jsonl",
        help="Official submission JSONL output path.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest JSONL used to validate legal candidate keys.",
    )
    args = parser.parse_args()

    rows = make_submission_file(
        predictions_path=args.predictions,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print({"output": args.output, "num_rows": len(rows)})


if __name__ == "__main__":
    main()
