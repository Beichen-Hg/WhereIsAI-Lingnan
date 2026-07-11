"""Build no-ASR feature table for full Phase1 provided-text inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from humomni.features.audio_ssl_extract import validate_emotion_embedding_rows
from humomni.features.build_feature_table import build_feature_table_rows
from humomni.features.prosody_extract import validate_prosody_cache_rows
from humomni.utils.hashing import file_sha256, json_sha256
from humomni.utils.io import read_jsonl, write_json, write_jsonl
from humomni.utils.train_guard import assert_asr_does_not_override_provided_text, assert_training_inputs_safe


DEFAULT_MANIFEST = Path("artifacts/manifests/phase1_test_full_provided_text.jsonl")
DEFAULT_FEATURE_ROOT = Path("artifacts/features/feat-phase1-full-provided-text/phase1_test")
DEFAULT_PROSODY = DEFAULT_FEATURE_ROOT / "prosody.jsonl"
DEFAULT_EMOTION2VEC = Path("artifacts/features/feat-e2v-plus-large/phase1_full_test/emotion2vec.jsonl")
DEFAULT_FEATURE_TABLE = DEFAULT_FEATURE_ROOT / "feature_table.jsonl"
DEFAULT_REPORT = DEFAULT_FEATURE_ROOT / "feature_table_report.json"


def build_phase1_full_feature_table(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    prosody_path: str | Path = DEFAULT_PROSODY,
    emotion2vec_path: str | Path = DEFAULT_EMOTION2VEC,
    output_path: str | Path = DEFAULT_FEATURE_TABLE,
    report_path: str | Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    manifest_rows = read_jsonl(manifest_path)
    prosody_rows = read_jsonl(prosody_path)
    emotion_rows = read_jsonl(emotion2vec_path)
    validate_prosody_cache_rows(prosody_rows, mode="test")
    validate_emotion_embedding_rows(emotion_rows, mode="test")
    rows = build_feature_table_rows(
        manifest_rows=manifest_rows,
        mode="test",
        feature_metadata={"feature_version": "feat-phase1-full-provided-text", "split": "phase1_test"},
    )
    assert_training_inputs_safe(rows=rows, mode="infer")
    assert_asr_does_not_override_provided_text(rows)
    expected_rows = sum(_candidate_count(row) for row in manifest_rows)
    source_counts = Counter(str(row.get("source_id", "unknown")) for row in manifest_rows)
    task_counts = Counter(str(row.get("task_type", "unknown")) for row in manifest_rows)
    candidate_counts = Counter(str(_candidate_count(row)) for row in manifest_rows)
    feature_candidate_counts = Counter(str(row.get("source_id", "unknown")) for row in rows)
    candidate_texts_identical = _candidate_texts_identical(rows)
    report = {
        "passed": len(rows) == expected_rows and candidate_texts_identical and not _leakage_hits(rows),
        "manifest_rows": len(manifest_rows),
        "feature_table_rows": len(rows),
        "expected_feature_table_rows": expected_rows,
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "option_count_distribution": dict(candidate_counts),
        "feature_rows_by_source": dict(feature_candidate_counts),
        "candidate_texts_identical": candidate_texts_identical,
        "semantic_text_source": "json_provided",
        "asr_used": False,
        "semantic_text_from_asr": False,
        "leakage_hits": _leakage_hits(rows)[:20],
        "input_hashes": {
            "manifest": file_sha256(manifest_path),
            "prosody": file_sha256(prosody_path),
            "emotion2vec": file_sha256(emotion2vec_path),
        },
        "feature_hash": json_sha256(rows),
        "output_path": Path(output_path).as_posix(),
    }
    if not report["passed"]:
        raise ValueError(f"full Phase1 feature-table audit failed: {report}")
    write_jsonl(output_path, rows)
    write_json(report_path, report)
    return report


def _candidate_count(row: Mapping[str, Any]) -> int:
    candidates = row.get("candidate_audio_paths", {})
    return len(candidates) if isinstance(candidates, Mapping) else 0


def _candidate_texts_identical(rows: list[dict[str, Any]]) -> bool:
    by_qid: dict[Any, set[str]] = {}
    for row in rows:
        by_qid.setdefault(row.get("question_id"), set()).add(str(row.get("candidate_transcript", "")))
    return all(len(values) == 1 for values in by_qid.values())


def _leakage_hits(value: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            key_lower = str(key).lower()
            if key_lower in {"label", "gold", "answer", "goodpara", "badpara", "is_gold_candidate"}:
                hits.append(child_path)
            hits.extend(_leakage_hits(child, path=child_path))
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
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--prosody", default=DEFAULT_PROSODY.as_posix())
    parser.add_argument("--emotion2vec", default=DEFAULT_EMOTION2VEC.as_posix())
    parser.add_argument("--output", default=DEFAULT_FEATURE_TABLE.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    args = parser.parse_args()
    report = build_phase1_full_feature_table(
        manifest_path=args.manifest,
        prosody_path=args.prosody,
        emotion2vec_path=args.emotion2vec,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
