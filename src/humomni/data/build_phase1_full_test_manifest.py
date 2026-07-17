"""Build full Phase1 provided-text manifests for GigaSpeech, MELD, and EmoV-DB.

All official Phase1 releases provide the semantic text directly in JSON:
``utterance`` for the user text and ``response`` for every candidate response.
Candidate options differ by audio delivery only, so candidate transcripts are
intentionally identical within each question and ASR is disabled.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from humomni.utils.hashing import file_sha256, json_sha256
from humomni.utils.io import write_json, write_jsonl


FORBIDDEN_KEYS = {"label", "gold", "answer", "goodPara", "badPara", "is_gold_candidate"}
SEMANTIC_TEXT_SOURCE = "json_provided"


@dataclass(frozen=True)
class Phase1ReleaseSpec:
    source_id: str
    task_type: str
    release_json: Path
    data_root: Path
    expected_rows: int
    expected_groups: int
    expected_option_count: int
    question_id_field: str = "question_id"
    split: str = "phase1_test"


DEFAULT_SPECS = (
    Phase1ReleaseSpec(
        source_id="gigaspeech",
        task_type="context_variant",
        release_json=Path(
            "data/raw/empathyeval/phase1-test_multi-context_gigaspeech/"
            "phase1-test_gigaspeech_release.json"
        ),
        data_root=Path("data/raw/empathyeval/phase1-test_multi-context_gigaspeech"),
        expected_rows=200,
        expected_groups=100,
        expected_option_count=2,
    ),
    Phase1ReleaseSpec(
        source_id="meld",
        task_type="context_variant",
        release_json=Path(
            "data/raw/empathyeval/phase1-test_multi-context_meld/"
            "phase1-test_meld_release.json"
        ),
        data_root=Path("data/raw/empathyeval/phase1-test_multi-context_meld"),
        expected_rows=210,
        expected_groups=70,
        expected_option_count=3,
    ),
    Phase1ReleaseSpec(
        source_id="emovdb",
        task_type="tone_variant",
        release_json=Path(
            "data/raw/empathyeval/phase1-test_multi-emotion_emovdb/"
            "phase1-test_emovdb_release.json"
        ),
        data_root=Path("data/raw/empathyeval/phase1-test_multi-emotion_emovdb"),
        expected_rows=120,
        expected_groups=30,
        expected_option_count=2,
    ),
)


def build_phase1_full_provided_text_manifest(
    *,
    output_manifest: str | Path = "artifacts/manifests/phase1_test_full_provided_text.jsonl",
    output_report: str | Path = "artifacts/manifests/phase1_test_full_provided_text_report.json",
    specs: Sequence[Phase1ReleaseSpec] = DEFAULT_SPECS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_reports: dict[str, dict[str, Any]] = {}
    missing_text: list[dict[str, Any]] = []
    audio_missing: list[dict[str, Any]] = []
    non_identical_candidates: list[dict[str, Any]] = []

    for spec in specs:
        source_rows, source_report = _build_source_rows(spec)
        source_reports[spec.source_id] = source_report
        rows.extend(source_rows)
        missing_text.extend(source_report["missing_text_examples"])
        audio_missing.extend(source_report["missing_audio_examples"])
        non_identical_candidates.extend(source_report["non_identical_candidate_examples"])

    qids = [row["question_id"] for row in rows]
    duplicate_qids = sorted(str(qid) for qid, count in Counter(qids).items() if count > 1)
    leakage_hits = _leakage_hits(rows)[:50]
    source_counts = dict(Counter(str(row["source_id"]) for row in rows))
    task_counts = dict(Counter(str(row["task_type"]) for row in rows))
    option_counts = dict(Counter(str(len(row["candidate_audio_paths"])) for row in rows))
    group_counts = {
        source: len({row["group_id"] for row in rows if row["source_id"] == source})
        for source in source_counts
    }
    manifest_hash_payload = {
        spec.source_id: {
            "release_json": spec.release_json.as_posix(),
            "release_json_hash": file_sha256(spec.release_json) if spec.release_json.exists() else None,
        }
        for spec in specs
    }

    expected_total_rows = sum(spec.expected_rows for spec in specs)
    expected_total_groups = sum(spec.expected_groups for spec in specs)
    expected_source_counts = {spec.source_id: spec.expected_rows for spec in specs}
    expected_group_counts = {spec.source_id: spec.expected_groups for spec in specs}
    expected_option_counts = dict(
        Counter(str(spec.expected_option_count) for spec in specs for _ in range(spec.expected_rows))
    )
    passed = (
        len(rows) == expected_total_rows
        and len({row["group_id"] for row in rows}) == expected_total_groups
        and source_counts == expected_source_counts
        and group_counts == expected_group_counts
        and option_counts == expected_option_counts
        and not duplicate_qids
        and not missing_text
        and not audio_missing
        and not non_identical_candidates
        and not leakage_hits
    )
    report = {
        "passed": passed,
        "num_rows": len(rows),
        "num_groups": len({row["group_id"] for row in rows}),
        "source_counts": source_counts,
        "task_counts": task_counts,
        "group_counts_by_source": group_counts,
        "option_count_distribution": option_counts,
        "expected_num_rows": expected_total_rows,
        "expected_num_groups": expected_total_groups,
        "expected_source_counts": expected_source_counts,
        "expected_group_counts_by_source": expected_group_counts,
        "expected_option_count_distribution": expected_option_counts,
        "missing_text_count": len(missing_text),
        "audio_missing_count": len(audio_missing),
        "duplicate_question_id_count": len(duplicate_qids),
        "candidate_transcripts_identical": not non_identical_candidates,
        "semantic_text_source": SEMANTIC_TEXT_SOURCE,
        "use_asr_for_semantic_text": False,
        "leakage_hits": leakage_hits,
        "missing_text_examples": missing_text[:20],
        "missing_audio_examples": audio_missing[:20],
        "duplicate_question_ids": duplicate_qids[:20],
        "non_identical_candidate_examples": non_identical_candidates[:20],
        "source_reports": source_reports,
        "release_jsons": manifest_hash_payload,
        "release_json_hash": json_sha256(manifest_hash_payload),
        "output_manifest": Path(output_manifest).as_posix(),
    }
    if not passed:
        raise ValueError(f"full Phase1 provided-text manifest audit failed: {report}")

    write_jsonl(output_manifest, rows)
    write_json(output_report, report)
    return report


def _build_source_rows(spec: Phase1ReleaseSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spec.release_json.exists():
        raise FileNotFoundError(f"Phase1 release JSON not found: {spec.release_json}")
    records = json.loads(spec.release_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{spec.release_json}: release JSON must contain a list")

    rows: list[dict[str, Any]] = []
    missing_text: list[dict[str, Any]] = []
    audio_missing: list[dict[str, Any]] = []
    non_identical_candidates: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{spec.source_id} release row {index} is not an object")
        _assert_no_forbidden_keys(record, path=f"{spec.source_id}[{index}]")
        question_id = _required_string(
            record,
            spec.question_id_field,
            path=f"{spec.source_id}[{index}]",
        )
        context = str(record.get("context", ""))
        utterance = str(record.get("utterance", "")).strip()
        response = str(record.get("response", "")).strip()
        if not utterance or not response:
            missing_text.append({"row_index": index, "question_id": question_id, "source_id": spec.source_id})

        utterance_audio_path = _resolve_release_audio_path(spec.data_root, record.get("utterance_audio"))
        options = record.get("options")
        if not isinstance(options, Mapping) or not options:
            raise ValueError(f"{spec.source_id} row {question_id}: options must be a non-empty mapping")
        candidate_audio_paths: dict[str, str] = {}
        for raw_key, raw_path in sorted(options.items(), key=lambda item: _option_key(str(item[0]))):
            candidate_id = _candidate_id_from_option_key(str(raw_key))
            candidate_audio_paths[candidate_id] = _resolve_release_audio_path(spec.data_root, raw_path)
        if len(candidate_audio_paths) != spec.expected_option_count:
            raise ValueError(
                f"{spec.source_id} row {question_id}: expected {spec.expected_option_count} options, "
                f"got {sorted(candidate_audio_paths)}"
            )
        expected_candidates = _expected_candidate_ids(spec.expected_option_count)
        if tuple(candidate_audio_paths) != expected_candidates:
            raise ValueError(
                f"{spec.source_id} row {question_id}: expected candidate ids {expected_candidates}, "
                f"got {tuple(candidate_audio_paths)}"
            )
        candidate_transcripts = {candidate_id: response for candidate_id in candidate_audio_paths}
        if len(set(candidate_transcripts.values())) != 1:
            non_identical_candidates.append({"question_id": question_id, "source_id": spec.source_id})

        row = {
            "question_id": question_id,
            "group_id": _group_id_from_question_id(question_id, spec.source_id),
            "task_type": spec.task_type,
            "source_id": spec.source_id,
            "context": context,
            "user_transcript": utterance,
            "response_text": response,
            "candidate_transcripts": candidate_transcripts,
            "utterance_audio_path": utterance_audio_path,
            "candidate_audio_paths": candidate_audio_paths,
            "candidate_labels": list(candidate_audio_paths),
            "provided_utterance_text": True,
            "provided_response_text": True,
            "semantic_text_source": SEMANTIC_TEXT_SOURCE,
            "use_asr_for_semantic_text": False,
            "split": spec.split,
        }
        _assert_no_forbidden_keys(row, path=f"manifest.{question_id}")
        rows.append(row)

    for row in rows:
        for field, path in _iter_audio_refs(row):
            path_text = str(path)
            if _is_ignored_macos_path(path_text):
                audio_missing.append(
                    {"question_id": row["question_id"], "source_id": spec.source_id, "field": field, "reason": "ignored_macos_artifact"}
                )
            elif not Path(path_text).exists():
                audio_missing.append({"question_id": row["question_id"], "source_id": spec.source_id, "field": field})

    duplicate_qids = sorted(str(qid) for qid, count in Counter(row["question_id"] for row in rows).items() if count > 1)
    report = {
        "source_id": spec.source_id,
        "task_type": spec.task_type,
        "num_rows": len(rows),
        "num_groups": len({row["group_id"] for row in rows}),
        "expected_rows": spec.expected_rows,
        "expected_groups": spec.expected_groups,
        "expected_option_count": spec.expected_option_count,
        "question_id_field": spec.question_id_field,
        "split": spec.split,
        "option_count_distribution": dict(Counter(str(len(row["candidate_audio_paths"])) for row in rows)),
        "missing_text_count": len(missing_text),
        "audio_missing_count": len(audio_missing),
        "duplicate_question_id_count": len(duplicate_qids),
        "candidate_transcripts_identical": not non_identical_candidates,
        "missing_text_examples": missing_text[:20],
        "missing_audio_examples": audio_missing[:20],
        "duplicate_question_ids": duplicate_qids[:20],
        "non_identical_candidate_examples": non_identical_candidates[:20],
        "release_json": spec.release_json.as_posix(),
        "release_json_hash": file_sha256(spec.release_json),
        "data_root": spec.data_root.as_posix(),
    }
    report["passed"] = (
        report["num_rows"] == spec.expected_rows
        and report["num_groups"] == spec.expected_groups
        and not missing_text
        and not audio_missing
        and not duplicate_qids
        and not non_identical_candidates
    )
    return rows, report


def _required_string(record: Mapping[str, Any], field: str, *, path: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: missing non-empty {field}")
    return value


def _candidate_id_from_option_key(option_key: str) -> str:
    if not option_key.startswith("opt-"):
        raise ValueError(f"unexpected option key: {option_key}")
    candidate_id = option_key.replace("opt-", "", 1)
    if not re.fullmatch(r"[A-Z]", candidate_id):
        raise ValueError(f"unexpected option candidate id: {option_key}")
    return candidate_id


def _option_key(option_key: str) -> tuple[int, str]:
    candidate_id = option_key.replace("opt-", "", 1)
    if len(candidate_id) == 1 and candidate_id.isalpha():
        return (ord(candidate_id.upper()) - ord("A"), option_key)
    return (999, option_key)


def _expected_candidate_ids(count: int) -> tuple[str, ...]:
    if count <= 0 or count > 26:
        raise ValueError(f"unsupported candidate count: {count}")
    return tuple(chr(ord("A") + index) for index in range(count))


def _resolve_release_audio_path(data_root: Path, raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"invalid release audio path: {raw_path!r}")
    normalized = raw_path[2:] if raw_path.startswith("./") else raw_path
    if _is_ignored_macos_path(normalized):
        raise ValueError(f"ignored macOS artifact path is not a valid audio reference: {raw_path}")
    return (data_root / normalized).as_posix()


def _group_id_from_question_id(question_id: str, source_id: str) -> str:
    if source_id == "gigaspeech":
        match = re.match(r"^gigaspeech_(\d+)_\d+$", question_id)
        if match:
            return f"gigaspeech_{match.group(1)}"
        match = re.match(r"^(\d+)_\d+$", question_id)
        if match:
            return f"gigaspeech_{match.group(1)}"
    if source_id == "meld":
        match = re.match(r"^meld_(\d+)_\d+$", question_id)
        if match:
            return f"meld_{match.group(1)}"
    if source_id == "emovdb":
        match = re.match(r"^emovdb_([^_]+)_e\d+$", question_id)
        if match:
            return f"emovdb_{match.group(1)}"
    return question_id


def _iter_audio_refs(row: Mapping[str, Any]) -> list[tuple[str, Any]]:
    refs = [("utterance_audio_path", row.get("utterance_audio_path"))]
    candidates = row.get("candidate_audio_paths", {})
    if isinstance(candidates, Mapping):
        refs.extend((f"candidate_audio_paths.{candidate_id}", path) for candidate_id, path in candidates.items())
    return refs


def _is_ignored_macos_path(path: str) -> bool:
    lowered = path.lower()
    return "__macosx" in lowered or lowered.endswith(".ds_store") or "/._" in lowered


def _assert_no_forbidden_keys(row: Mapping[str, Any], *, path: str) -> None:
    forbidden = FORBIDDEN_KEYS & set(row)
    if forbidden:
        raise ValueError(f"{path}: forbidden label/gold fields present: {sorted(forbidden)}")


def _leakage_hits(value: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            key_lower = str(key).lower()
            if str(key) in FORBIDDEN_KEYS or key_lower in {item.lower() for item in FORBIDDEN_KEYS}:
                hits.append(child_path)
            hits.extend(_leakage_hits(child, path=child_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            hits.extend(_leakage_hits(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "goodpara" in lowered or "badpara" in lowered or "__macosx" in lowered or lowered.endswith(".ds_store"):
            hits.append(path)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-manifest", default="artifacts/manifests/phase1_test_full_provided_text.jsonl")
    parser.add_argument(
        "--output-report",
        default="artifacts/manifests/phase1_test_full_provided_text_report.json",
    )
    args = parser.parse_args()
    report = build_phase1_full_provided_text_manifest(
        output_manifest=args.output_manifest,
        output_report=args.output_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
