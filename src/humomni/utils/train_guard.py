"""Guards that keep Phase1 test data out of supervised training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from humomni.utils.io import read_jsonl


FORBIDDEN_LEAKAGE_FIELDS = {"gold", "answer", "goodPara", "badPara", "is_gold_candidate"}
TEST_SPLIT_MARKERS = {"phase1_test", "phase2_test", "test", "phase1-test", "phase2-test"}
TEST_PATH_MARKERS = {
    "phase1_test",
    "phase2_test",
    "phase1-test",
    "phase2-test",
    "feat-phase1-test",
    "feat-phase2-test",
    "feat-phase1-provided-text",
    "feat-phase2-provided-text",
    "feat-phase1-full-provided-text",
    "feat-phase2-full-provided-text",
    "phase1_full_test",
    "phase2_full_test",
}
SUPERVISED_LABEL_FIELDS = {"label"}
PHASE1_SPECIALIST_FORBIDDEN_FEATURE_TERMS = {
    "candidate_text_difference",
    "candidate_transcript_ngram",
    "candidate_asr_semantic",
    "candidate_text_ngram",
    "candidate_text_diff",
    "path",
    "hash",
    "audio_id",
}
SUBMISSION_FORBIDDEN_FIELDS = {
    "score",
    "scores",
    "candidate_scores",
    "path",
    "audio_path",
    "label",
    "gold",
    "goodPara",
    "badPara",
    "is_gold_candidate",
}


def assert_training_inputs_safe(
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
    paths: Sequence[str | Path] = (),
    mode: str,
    allow_valid_labels: bool = True,
) -> None:
    """Reject supervised training inputs that point at Phase1 test or leak labels."""

    supervised = mode == "supervised_train"
    for path in paths:
        _check_path(path, supervised=supervised)
    if rows is not None:
        _check_rows(rows, supervised=supervised, allow_valid_labels=allow_valid_labels)


def assert_jsonl_training_inputs_safe(
    *,
    paths: Sequence[str | Path],
    mode: str,
    allow_valid_labels: bool = True,
) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        _check_path(path, supervised=mode == "supervised_train")
        if Path(path).exists() and Path(path).suffix == ".jsonl":
            rows.extend(read_jsonl(path))
    assert_training_inputs_safe(
        rows=rows,
        paths=(),
        mode=mode,
        allow_valid_labels=allow_valid_labels,
    )


def assert_asr_does_not_override_provided_text(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject provided-text rows whose semantic text was overwritten by ASR."""

    for index, row in enumerate(rows):
        if row.get("semantic_text_source") != "json_provided":
            continue
        if row.get("semantic_text_from_asr") is True or row.get("asr_used") is True:
            raise ValueError(f"row {index}: ASR cannot override JSON-provided semantic text")
        if row.get("use_asr_for_semantic_text") is True:
            raise ValueError(f"row {index}: use_asr_for_semantic_text must be false")


def assert_phase1_specialist_features_safe(feature_names: Sequence[str]) -> None:
    """Reject text-difference, path/hash, and audio-id features in Phase1 specialist."""

    unsafe = [
        name
        for name in feature_names
        if any(term in str(name).lower() for term in PHASE1_SPECIALIST_FORBIDDEN_FEATURE_TERMS)
    ]
    if unsafe:
        raise ValueError(f"phase1 audio-delivery specialist forbidden features: {unsafe[:12]}")


def assert_submission_rows_safe(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject submissions containing anything except question_id and answer."""

    for index, row in enumerate(rows):
        keys = set(row)
        if keys != {"question_id", "answer"}:
            raise ValueError(f"submission row {index}: expected only question_id and answer, got {sorted(keys)}")
        forbidden = SUBMISSION_FORBIDDEN_FIELDS & keys
        if forbidden:
            raise ValueError(f"submission row {index}: forbidden fields {sorted(forbidden)}")


def _check_path(path: str | Path, *, supervised: bool) -> None:
    text = str(path).lower()
    if supervised and any(marker in text for marker in TEST_PATH_MARKERS):
        raise ValueError(f"training guard blocked supervised training path containing test split: {path}")


def _check_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    supervised: bool,
    allow_valid_labels: bool,
) -> None:
    for index, row in enumerate(rows):
        split = str(row.get("split", "")).lower()
        if supervised and any(marker in split for marker in TEST_SPLIT_MARKERS):
            raise ValueError(f"training guard blocked supervised training row from test split={split} at row {index}")
        if any(marker in split for marker in TEST_SPLIT_MARKERS):
            forbidden = (FORBIDDEN_LEAKAGE_FIELDS | SUPERVISED_LABEL_FIELDS) & set(row)
            if forbidden:
                raise ValueError(f"training guard blocked test leakage fields at row {index}: {sorted(forbidden)}")
        if not supervised:
            forbidden = (FORBIDDEN_LEAKAGE_FIELDS | SUPERVISED_LABEL_FIELDS) & set(row)
            if forbidden:
                raise ValueError(f"training guard blocked leakage fields outside supervised training at row {index}: {sorted(forbidden)}")
        if supervised and not allow_valid_labels and split == "valid" and "label" in row:
            raise ValueError("training guard blocked valid labels in training input")
