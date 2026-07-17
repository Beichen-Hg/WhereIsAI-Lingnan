"""Run the frozen audio-delivery scorer on blind EmpathyEval Phase 2 data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humomni.infer.audit_submission import audit_submission_rows
from humomni.infer.phase1_full_audio_delivery_system import (
    DEFAULT_AUDIO_MODEL_DIR,
    predict_full_phase1_audio_delivery,
)
from humomni.utils.hashing import file_sha256
from humomni.utils.io import read_jsonl, write_json, write_jsonl
from humomni.utils.train_guard import (
    assert_asr_does_not_override_provided_text,
    assert_submission_rows_safe,
    assert_training_inputs_safe,
)

SYSTEM_ID = "humomni_phase2_full_audio_delivery_v1"
DEFAULT_MANIFEST = Path("artifacts/manifests/phase2_test_full_provided_text.jsonl")
DEFAULT_FEATURE_ROOT = Path("artifacts/features/feat-phase2-full-provided-text/phase2_test")
DEFAULT_FEATURE_TABLE = DEFAULT_FEATURE_ROOT / "feature_table.jsonl"
DEFAULT_PROSODY = DEFAULT_FEATURE_ROOT / "prosody.jsonl"
DEFAULT_EMOTION2VEC = Path("artifacts/features/feat-e2v-plus-large/phase2_full_test/emotion2vec.jsonl")
DEFAULT_WAVLM = Path("artifacts/features/feat-wavlm-chunked/phase2_full_test/wavlm.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/submissions/phase2_test_full_audio_delivery/audio_delivery_all_labeled_clean_no_quality_v1")

EXPECTED_SOURCE_COUNTS = {"gigaspeech": 226, "meld": 168, "emovdb": 148}
EXPECTED_GROUP_COUNTS = {"gigaspeech": 113, "meld": 56, "emovdb": 37}
EXPECTED_TASK_COUNTS = {"context_variant": 394, "tone_variant": 148}
EXPECTED_OPTION_COUNTS = {"2": 374, "3": 168}


def run_phase2_full_audio_delivery_system(
    *,
    model_dir: str | Path = DEFAULT_AUDIO_MODEL_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    feature_table_path: str | Path = DEFAULT_FEATURE_TABLE,
    prosody_path: str | Path = DEFAULT_PROSODY,
    emotion2vec_path: str | Path = DEFAULT_EMOTION2VEC,
    wavlm_path: str | Path = DEFAULT_WAVLM,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
    audit_output: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a submission from Phase 2 inputs using the frozen Phase 1 scorer."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dependency_path = Path(audit_output) if audit_output is not None else output / "dependency_audit_report.json"
    dependency_report = audit_phase2_full_audio_delivery_dependencies(
        model_dir=model_dir,
        manifest_path=manifest_path,
        feature_table_path=feature_table_path,
        prosody_path=prosody_path,
        emotion2vec_path=emotion2vec_path,
        wavlm_path=wavlm_path,
        output_path=dependency_path,
    )
    if not dependency_report["passed"]:
        raise RuntimeError(f"Phase 2 dependency audit failed: {dependency_report}")

    manifest_rows = read_jsonl(manifest_path)
    feature_rows = read_jsonl(feature_table_path)
    predictions = predict_full_phase1_audio_delivery(
        model_dir=model_dir,
        manifest_rows=manifest_rows,
        feature_table_rows=feature_rows,
        prosody_rows=read_jsonl(prosody_path),
        emotion_rows=read_jsonl(emotion2vec_path),
        wavlm_rows=read_jsonl(wavlm_path),
        device=device,
    )
    for prediction in predictions:
        prediction["system_id"] = SYSTEM_ID
    write_jsonl(output / "predictions_with_scores.jsonl", predictions)

    submission_rows = [
        {"question_id": row["question_id"], "answer": row["answer"]}
        for row in predictions
    ]
    assert_submission_rows_safe(submission_rows)
    write_jsonl(output / "submission.jsonl", submission_rows)
    submission_audit = _submission_audit(submission_rows, manifest_rows)
    write_json(output / "submission_audit_report.json", submission_audit)
    if not submission_audit["passed"]:
        raise RuntimeError(f"Phase 2 submission audit failed: {submission_audit}")

    inference_audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_id": SYSTEM_ID,
        "primary_model": "audio_delivery_pairwise_all_labeled_clean_no_quality_v1",
        "prediction_rows": len(predictions),
        "answer_distribution": dict(Counter(row["answer"] for row in predictions)),
        "model_hash": file_sha256(Path(model_dir) / "model.pt"),
        "input_hashes": {
            "manifest": file_sha256(manifest_path),
            "feature_table": file_sha256(feature_table_path),
            "prosody": file_sha256(prosody_path),
            "emotion2vec": file_sha256(emotion2vec_path),
            "wavlm": file_sha256(wavlm_path),
        },
        "dependency_audit": dependency_report,
        "submission_audit": submission_audit,
        "semantic_text_source": "json_provided",
        "asr_used": False,
        "text_modules_used_for_candidate_decision": False,
        "fusion_used": False,
        "fallback_used": False,
        "no_train_on_test": True,
        "no_teacher_on_test": True,
    }
    write_json(output / "inference_audit_report.json", inference_audit)
    return {
        "system_id": SYSTEM_ID,
        "output_dir": output.as_posix(),
        "submission_path": (output / "submission.jsonl").as_posix(),
        "prediction_rows": len(predictions),
        "answer_distribution": submission_audit["answer_distribution"],
        "submission_audit_passed": submission_audit["passed"],
        "dependency_audit_passed": dependency_report["passed"],
    }


def audit_phase2_full_audio_delivery_dependencies(
    *,
    model_dir: str | Path = DEFAULT_AUDIO_MODEL_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    feature_table_path: str | Path = DEFAULT_FEATURE_TABLE,
    prosody_path: str | Path = DEFAULT_PROSODY,
    emotion2vec_path: str | Path = DEFAULT_EMOTION2VEC,
    wavlm_path: str | Path = DEFAULT_WAVLM,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Phase 2 inputs without loading or inferring any answer labels."""

    paths = {
        "model_pt": Path(model_dir) / "model.pt",
        "model_config": Path(model_dir) / "config.yaml",
        "manifest": Path(manifest_path),
        "feature_table": Path(feature_table_path),
        "prosody": Path(prosody_path),
        "emotion2vec": Path(emotion2vec_path),
        "wavlm": Path(wavlm_path),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    if not missing:
        manifest_rows = read_jsonl(paths["manifest"])
        feature_rows = read_jsonl(paths["feature_table"])
        prosody_rows = read_jsonl(paths["prosody"])
        emotion_rows = read_jsonl(paths["emotion2vec"])
        wavlm_rows = read_jsonl(paths["wavlm"])
        assert_training_inputs_safe(rows=manifest_rows, mode="infer")
        assert_training_inputs_safe(rows=feature_rows, mode="infer")
        assert_asr_does_not_override_provided_text(feature_rows)

        source_counts = Counter(str(row.get("source_id", "unknown")) for row in manifest_rows)
        task_counts = Counter(str(row.get("task_type", "unknown")) for row in manifest_rows)
        group_counts = {
            source: len({row.get("group_id") for row in manifest_rows if row.get("source_id") == source})
            for source in EXPECTED_SOURCE_COUNTS
        }
        option_counts = Counter(str(len(_candidate_ids(row))) for row in manifest_rows)
        qids = [row.get("question_id") for row in manifest_rows]
        expected_feature_rows = sum(len(_candidate_ids(row)) for row in manifest_rows)
        feature_pairs = {
            (row.get("question_id"), str(row.get("candidate_id")))
            for row in feature_rows
        }
        expected_pairs = {
            (row.get("question_id"), candidate_id)
            for row in manifest_rows
            for candidate_id in _candidate_ids(row)
        }
        audio_paths = [
            path
            for row in manifest_rows
            for path in [row.get("utterance_audio_path"), *dict(row.get("candidate_audio_paths", {})).values()]
        ]
        prosody_empty = _empty_audio_feature_count(
            manifest_rows,
            prosody_rows,
            user_key="user_prosody",
            candidate_key="candidate_prosodies",
        )
        emotion_empty = _empty_audio_feature_count(
            manifest_rows,
            emotion_rows,
            user_key="user_emotion_embedding",
            candidate_key="candidate_emotion_embeddings",
        )
        wavlm_empty = _empty_audio_feature_count(
            manifest_rows,
            wavlm_rows,
            user_key="user_wavlm_embedding",
            candidate_key="candidate_wavlm_embeddings",
        )
        checks = {
            "manifest_row_count_is_542": len(manifest_rows) == 542,
            "manifest_question_ids_are_unique": len(qids) == len(set(qids)) and all(isinstance(qid, str) and qid for qid in qids),
            "source_counts_match_release": dict(source_counts) == EXPECTED_SOURCE_COUNTS,
            "group_counts_match_release": group_counts == EXPECTED_GROUP_COUNTS,
            "task_counts_match_release": dict(task_counts) == EXPECTED_TASK_COUNTS,
            "option_counts_match_release": dict(option_counts) == EXPECTED_OPTION_COUNTS,
            "manifest_audio_paths_exist": all(isinstance(path, str) and Path(path).is_file() for path in audio_paths),
            "manifest_has_no_label_terms": not _leakage_hits(manifest_rows),
            "feature_table_row_count_matches_candidates": len(feature_rows) == expected_feature_rows,
            "feature_table_covers_every_candidate": feature_pairs == expected_pairs,
            "prosody_row_count_matches_manifest": len(prosody_rows) == len(manifest_rows),
            "emotion2vec_row_count_matches_manifest": len(emotion_rows) == len(manifest_rows),
            "wavlm_row_count_matches_manifest": len(wavlm_rows) == len(manifest_rows),
            "prosody_has_no_empty_audio_features": prosody_empty == 0,
            "emotion2vec_has_no_empty_audio_embeddings": emotion_empty == 0,
            "wavlm_has_no_empty_audio_embeddings": wavlm_empty == 0,
            "feature_rows_have_no_label_terms": not _leakage_hits(feature_rows),
        }
        details = {
            "manifest_rows": len(manifest_rows),
            "feature_table_rows": len(feature_rows),
            "expected_feature_table_rows": expected_feature_rows,
            "prosody_rows": len(prosody_rows),
            "emotion2vec_rows": len(emotion_rows),
            "wavlm_rows": len(wavlm_rows),
            "source_counts": dict(source_counts),
            "group_counts": group_counts,
            "task_counts": dict(task_counts),
            "option_counts": dict(option_counts),
            "empty_audio_feature_counts": {
                "prosody": prosody_empty,
                "emotion2vec": emotion_empty,
                "wavlm": wavlm_empty,
            },
            "feature_hashes": {
                name: file_sha256(path)
                for name, path in paths.items()
                if name not in {"model_pt", "model_config"}
            },
        }
    report = {
        "system_id": SYSTEM_ID,
        "passed": not missing and all(checks.values()),
        "missing": missing,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "details": details,
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "no_train_on_test": True,
        "no_teacher_on_test": True,
        "no_asr_used": True,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def _candidate_ids(row: Mapping[str, Any]) -> list[str]:
    candidates = row.get("candidate_audio_paths", {})
    if not isinstance(candidates, Mapping):
        return []
    return sorted(str(candidate_id) for candidate_id in candidates)


def _empty_audio_feature_count(
    manifest_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    user_key: str,
    candidate_key: str,
) -> int:
    """Count missing user/candidate vectors or dictionaries in a feature cache."""

    by_question_id = {row.get("question_id"): row for row in feature_rows}
    empty = 0
    for manifest_row in manifest_rows:
        feature_row = by_question_id.get(manifest_row.get("question_id"), {})
        if not feature_row.get(user_key):
            empty += 1
        candidate_values = feature_row.get(candidate_key, {})
        if not isinstance(candidate_values, Mapping):
            empty += len(_candidate_ids(manifest_row))
            continue
        for candidate_id in _candidate_ids(manifest_row):
            if not candidate_values.get(candidate_id):
                empty += 1
    return empty


def _submission_audit(
    submission_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = audit_submission_rows(submission_rows, manifest_rows=manifest_rows)
    question_id_order_matches_manifest = [
        row.get("question_id") for row in submission_rows
    ] == [row.get("question_id") for row in manifest_rows]
    if not question_id_order_matches_manifest:
        report["errors"].append("question_id order does not match manifest")
        report["passed"] = False
    manifest_by_qid = {row["question_id"]: row for row in manifest_rows}
    report.update(
        {
            "row_count": len(submission_rows),
            "unique_question_id_count": len({row["question_id"] for row in submission_rows}),
            "question_id_order_matches_manifest": question_id_order_matches_manifest,
            "answer_distribution": dict(Counter(str(row["answer"]) for row in submission_rows)),
            "answer_distribution_by_source": {
                source: dict(
                    Counter(
                        str(row["answer"])
                        for row in submission_rows
                        if manifest_by_qid[row["question_id"]].get("source_id") == source
                    )
                )
                for source in EXPECTED_SOURCE_COUNTS
            },
            "source_counts": dict(Counter(str(row.get("source_id")) for row in manifest_rows)),
            "task_counts": dict(Counter(str(row.get("task_type")) for row in manifest_rows)),
            "only_question_id_and_answer": all(set(row) == {"question_id", "answer"} for row in submission_rows),
            "no_label_or_gold_in_submission": all(
                not {"label", "gold", "goodPara", "badPara"} & set(row)
                for row in submission_rows
            ),
            "no_score_or_path_in_submission": all(
                not {"score", "scores", "candidate_scores", "path", "audio_path"} & set(row)
                for row in submission_rows
            ),
        }
    )
    return report


def _leakage_hits(value: Any) -> list[str]:
    serialized = json.dumps(value, ensure_ascii=False).lower()
    return [term for term in ("goodpara", "badpara", '"label"', '"gold"', '"answer"') if term in serialized]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_AUDIO_MODEL_DIR.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE.as_posix())
    parser.add_argument("--prosody", default=DEFAULT_PROSODY.as_posix())
    parser.add_argument("--emotion2vec", default=DEFAULT_EMOTION2VEC.as_posix())
    parser.add_argument("--wavlm", default=DEFAULT_WAVLM.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--audit-output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        report = audit_phase2_full_audio_delivery_dependencies(
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            feature_table_path=args.feature_table,
            prosody_path=args.prosody,
            emotion2vec_path=args.emotion2vec,
            wavlm_path=args.wavlm,
            output_path=args.audit_output,
        )
    else:
        report = run_phase2_full_audio_delivery_system(
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            feature_table_path=args.feature_table,
            prosody_path=args.prosody,
            emotion2vec_path=args.emotion2vec,
            wavlm_path=args.wavlm,
            output_dir=args.output_dir,
            audit_output=args.audit_output,
            device=args.device,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
