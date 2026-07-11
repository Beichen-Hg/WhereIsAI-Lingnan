"""Build the final provided-text candidate feature table.

The final Phase 1 method uses official JSON text directly. Candidate response
text is intentionally identical within a question, so this module never uses
ASR, teacher labels, audio-path metadata, or candidate-text differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from humomni.utils.io import read_jsonl, write_jsonl


STRICT_MODES = frozenset({"infer", "test"})
FORBIDDEN_FIELDS = {"label", "gold", "answer", "goodPara", "badPara", "is_gold_candidate"}


def build_feature_table_rows(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    mode: str,
    feature_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand a final provided-text manifest into one row per audio candidate."""

    normalized_mode = mode.lower()
    if normalized_mode not in STRICT_MODES:
        raise ValueError("final feature tables are only valid for infer/test mode")
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        if not _uses_provided_text(manifest_row):
            raise ValueError("final feature tables require json_provided text with ASR disabled")
        candidate_paths = manifest_row.get("candidate_audio_paths")
        candidate_transcripts = manifest_row.get("candidate_transcripts")
        if not isinstance(candidate_paths, Mapping) or not candidate_paths:
            raise ValueError("manifest row must include candidate_audio_paths")
        if not isinstance(candidate_transcripts, Mapping):
            raise ValueError("provided-text manifest row must include candidate_transcripts")
        for candidate_id in sorted(str(candidate_id) for candidate_id in candidate_paths):
            if candidate_id not in candidate_transcripts:
                raise ValueError(f"provided-text manifest row missing candidate transcript for {candidate_id}")
            row: dict[str, Any] = {
                "question_id": manifest_row["question_id"],
                "group_id": manifest_row["group_id"],
                "task_type": manifest_row.get("task_type", "unknown"),
                "source_id": manifest_row.get("source_id", "unknown"),
                "candidate_id": candidate_id,
                "split": manifest_row.get("split", normalized_mode),
                "context": str(manifest_row.get("context", "")),
                "user_transcript": str(manifest_row.get("user_transcript", "")),
                "candidate_transcript": str(candidate_transcripts[candidate_id]),
                "response_text": str(manifest_row.get("response_text", candidate_transcripts[candidate_id])),
                "provided_utterance_text": True,
                "provided_response_text": True,
                "semantic_text_source": "json_provided",
                "use_asr_for_semantic_text": False,
                "asr_used": False,
                "semantic_text_from_asr": False,
            }
            if feature_metadata:
                row["feature_metadata"] = dict(feature_metadata)
            rows.append(row)
    _assert_final_rows_safe(rows)
    return rows


def build_feature_table_file(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    mode: str,
    feature_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = build_feature_table_rows(
        manifest_rows=read_jsonl(manifest_path),
        mode=mode,
        feature_metadata=feature_metadata,
    )
    write_jsonl(output_path, rows)
    return rows


def _uses_provided_text(row: Mapping[str, Any]) -> bool:
    return (
        row.get("semantic_text_source") == "json_provided"
        and row.get("use_asr_for_semantic_text") is False
        and bool(row.get("provided_utterance_text"))
        and bool(row.get("provided_response_text"))
    )


def _assert_final_rows_safe(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        forbidden = FORBIDDEN_FIELDS & set(row)
        if forbidden:
            raise ValueError(f"final feature table contains forbidden fields: {sorted(forbidden)}")
        if row.get("asr_used") is not False or row.get("semantic_text_source") != "json_provided":
            raise ValueError("final feature table must use provided text with ASR disabled")
    payload = json.dumps(rows, ensure_ascii=False).lower()
    if "goodpara" in payload or "badpara" in payload:
        raise ValueError("final feature table contains raw label-like tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="test", choices=sorted(STRICT_MODES))
    args = parser.parse_args()
    rows = build_feature_table_file(
        manifest_path=args.manifest,
        output_path=args.output,
        mode=args.mode,
    )
    print({"output": args.output, "rows": len(rows), "asr_used": False})


if __name__ == "__main__":
    main()
