"""Full Phase1 provided-text audio-delivery ranking system.

This extends the clean GigaSpeech-only route to all official Phase1 sources:

* GigaSpeech context_variant, A/B
* MELD context_variant, A/B/C
* EmoV-DB tone_variant, A/B

The system never uses ASR, text reranking, teacher calls, training, or labels on
Phase1 test. Candidate transcripts are identical within a question and the
decision comes from audio-delivery features.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from humomni.infer.audit_submission import audit_submission_rows
from humomni.infer.final_audio_delivery_features import (
    candidate_feature_map,
    matrix,
    read_optional_jsonl,
    pairwise_features,
    resolve_torch_device,
)
from humomni.models.audio_delivery_pairwise import AudioDeliveryPairwiseMLP
from humomni.utils.hashing import file_sha256, json_sha256
from humomni.utils.io import read_jsonl, write_json, write_jsonl
from humomni.utils.train_guard import (
    assert_asr_does_not_override_provided_text,
    assert_phase1_specialist_features_safe,
    assert_submission_rows_safe,
    assert_training_inputs_safe,
)


SYSTEM_ID = "humomni_phase1_full_audio_delivery_v1"
DEFAULT_AUDIO_MODEL_DIR = Path("artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1")
DEFAULT_MANIFEST = Path("artifacts/manifests/phase1_test_full_provided_text.jsonl")
DEFAULT_FEATURE_ROOT = Path("artifacts/features/feat-phase1-full-provided-text/phase1_test")
DEFAULT_FEATURE_TABLE = DEFAULT_FEATURE_ROOT / "feature_table.jsonl"
DEFAULT_PROSODY = DEFAULT_FEATURE_ROOT / "prosody.jsonl"
DEFAULT_EMOTION2VEC = Path("artifacts/features/feat-e2v-plus-large/phase1_full_test/emotion2vec.jsonl")
DEFAULT_WAVLM = Path("artifacts/features/feat-wavlm-chunked/phase1_full_test/wavlm.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/reproduced") / "humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1"
DEFAULT_AUDIT_OUTPUT = DEFAULT_OUTPUT_DIR / "dependency_audit_report.json"
FORBIDDEN_FIELDS = {"label", "gold", "goodPara", "badPara", "is_gold_candidate"}


def run_phase1_full_audio_delivery_system(
    *,
    model_dir: str | Path = DEFAULT_AUDIO_MODEL_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    feature_table_path: str | Path = DEFAULT_FEATURE_TABLE,
    prosody_path: str | Path = DEFAULT_PROSODY,
    emotion2vec_path: str | Path = DEFAULT_EMOTION2VEC,
    wavlm_path: str | Path = DEFAULT_WAVLM,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
    audit_output: str | Path | None = DEFAULT_AUDIT_OUTPUT,
) -> dict[str, Any]:
    dependency_report = audit_phase1_full_audio_delivery_dependencies(
        model_dir=model_dir,
        manifest_path=manifest_path,
        feature_table_path=feature_table_path,
        prosody_path=prosody_path,
        emotion2vec_path=emotion2vec_path,
        wavlm_path=wavlm_path,
        output_path=audit_output,
    )
    if not dependency_report["passed"]:
        raise RuntimeError(f"full Phase1 audio-delivery dependency audit failed: {dependency_report}")

    manifest_rows = read_jsonl(manifest_path)
    feature_rows = read_jsonl(feature_table_path)
    predictions = predict_full_phase1_audio_delivery(
        model_dir=model_dir,
        manifest_rows=manifest_rows,
        feature_table_rows=feature_rows,
        prosody_rows=read_jsonl(prosody_path),
        emotion_rows=read_jsonl(emotion2vec_path),
        wavlm_rows=read_optional_jsonl(wavlm_path),
        device=device,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions_with_scores.jsonl", predictions)

    submission_rows = [{"question_id": row["question_id"], "answer": row["answer"]} for row in predictions]
    assert_submission_rows_safe(submission_rows)
    write_jsonl(output / "submission.jsonl", submission_rows)
    submission_audit = _submission_audit(submission_rows, manifest_rows)
    write_json(output / "submission_audit_report.json", submission_audit)
    if not submission_audit["passed"]:
        raise RuntimeError(f"full Phase1 audio-delivery submission audit failed: {submission_audit}")

    inference_audit = _inference_audit(
        predictions=predictions,
        submission_audit=submission_audit,
        dependency_report=dependency_report,
        model_dir=Path(model_dir),
        manifest_path=Path(manifest_path),
        feature_table_path=Path(feature_table_path),
        prosody_path=Path(prosody_path),
        emotion2vec_path=Path(emotion2vec_path),
        wavlm_path=Path(wavlm_path),
    )
    write_json(output / "inference_audit_report.json", inference_audit)
    return {
        "system_id": SYSTEM_ID,
        "output_dir": output.as_posix(),
        "submission_path": (output / "submission.jsonl").as_posix(),
        "prediction_rows": len(predictions),
        "answer_distribution": submission_audit["answer_distribution"],
        "answer_distribution_by_source": submission_audit["answer_distribution_by_source"],
        "submission_audit_passed": submission_audit["passed"],
        "dependency_audit_passed": dependency_report["passed"],
        "no_asr": True,
        "no_train_on_test": True,
        "no_teacher_on_test": True,
        "fusion_used": False,
    }


def predict_full_phase1_audio_delivery(
    *,
    model_dir: str | Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    feature_table_rows: Sequence[Mapping[str, Any]],
    prosody_rows: Sequence[Mapping[str, Any]],
    emotion_rows: Sequence[Mapping[str, Any]],
    wavlm_rows: Sequence[Mapping[str, Any]],
    device: str = "auto",
) -> list[dict[str, Any]]:
    assert_training_inputs_safe(rows=feature_table_rows, mode="infer")
    assert_asr_does_not_override_provided_text(feature_table_rows)
    checkpoint = torch.load(Path(model_dir) / "model.pt", map_location="cpu")
    feature_names = list(checkpoint["feature_names"])
    assert_phase1_specialist_features_safe(feature_names)
    torch_device = resolve_torch_device(device)
    model = AudioDeliveryPairwiseMLP(
        input_dim=len(feature_names),
        hidden_dims=checkpoint.get("hidden_dims", [256, 128]),
        dropout=float(checkpoint.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch_device)
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0

    feature_by_qid: dict[Any, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in feature_table_rows:
        feature_by_qid[row["question_id"]][str(row.get("candidate_id"))] = row
    prosody_by_qid = {row["question_id"]: row for row in prosody_rows}
    emotion_by_qid = {row["question_id"]: row for row in emotion_rows}
    wavlm_by_qid = {row["question_id"]: row for row in wavlm_rows}

    predictions: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        qid = manifest_row["question_id"]
        candidates = _candidate_ids(manifest_row)
        by_candidate = feature_by_qid.get(qid, {})
        missing = [candidate_id for candidate_id in candidates if candidate_id not in by_candidate]
        if missing:
            raise ValueError(f"feature table missing candidates for {qid}: {missing}")
        if len(candidates) < 2:
            raise ValueError(f"question {qid}: at least two candidates required")

        aggregate_scores = {candidate_id: 0.0 for candidate_id in candidates}
        pairwise_scores: dict[str, float] = {}
        pair_rows: list[dict[str, Any]] = []
        pair_keys: list[tuple[str, str]] = []
        for left, right in itertools.combinations(candidates, 2):
            left_features = candidate_feature_map(
                base_row=by_candidate[left],
                candidate_id=left,
                prosody_row=prosody_by_qid.get(qid, {}),
                emotion_row=emotion_by_qid.get(qid, {}),
                wavlm_row=wavlm_by_qid.get(qid, {}),
            )
            right_features = candidate_feature_map(
                base_row=by_candidate[right],
                candidate_id=right,
                prosody_row=prosody_by_qid.get(qid, {}),
                emotion_row=emotion_by_qid.get(qid, {}),
                wavlm_row=wavlm_by_qid.get(qid, {}),
            )
            pair_rows.append({"features": pairwise_features(left_features, right_features)})
            pair_keys.append((left, right))
        probs = _predict_pair_probs(model, feature_names, mean, std, pair_rows, torch_device)
        for (left, right), prob in zip(pair_keys, probs, strict=True):
            p_left = float(prob)
            aggregate_scores[left] += p_left
            aggregate_scores[right] += 1.0 - p_left
            pairwise_scores[f"{left}>{right}"] = p_left
            pairwise_scores[f"{right}>{left}"] = 1.0 - p_left
        denom = max(len(candidates) - 1, 1)
        candidate_scores = {
            candidate_id: float(score / denom)
            for candidate_id, score in aggregate_scores.items()
        }
        answer = max(candidates, key=lambda candidate_id: (candidate_scores[candidate_id], -candidates.index(candidate_id)))
        predictions.append(
            {
                "question_id": qid,
                "group_id": manifest_row.get("group_id"),
                "task_type": manifest_row.get("task_type", "unknown"),
                "source_id": manifest_row.get("source_id", "unknown"),
                "candidate_scores": candidate_scores,
                "pairwise_scores": pairwise_scores,
                "answer": answer,
                "margin": _margin(candidate_scores),
                "num_candidates": len(candidates),
                "candidate_labels": candidates,
                "system_id": SYSTEM_ID,
                "primary_model": "audio_delivery_pairwise_all_labeled_clean_no_quality_v1",
                "semantic_text_source": "json_provided",
                "asr_used": False,
                "text_modules_used_for_candidate_decision": False,
                "fusion_used": False,
                "fallback_used": False,
            }
        )
    return predictions


def audit_phase1_full_audio_delivery_dependencies(
    *,
    model_dir: str | Path = DEFAULT_AUDIO_MODEL_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    feature_table_path: str | Path = DEFAULT_FEATURE_TABLE,
    prosody_path: str | Path = DEFAULT_PROSODY,
    emotion2vec_path: str | Path = DEFAULT_EMOTION2VEC,
    wavlm_path: str | Path = DEFAULT_WAVLM,
    output_path: str | Path | None = DEFAULT_AUDIT_OUTPUT,
) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "model_pt": Path(model_dir) / "model.pt",
        "model_config": Path(model_dir) / "config.yaml",
        "manifest": Path(manifest_path),
        "feature_table": Path(feature_table_path),
        "prosody": Path(prosody_path),
        "emotion2vec_plus_large": Path(emotion2vec_path),
    }
    paths["wavlm"] = Path(wavlm_path)
    missing = [name for name, path in paths.items() if not path.exists()]
    checks: dict[str, Any] = {}
    model_info: dict[str, Any] = {}
    feature_reports: dict[str, Any] = {}

    if not missing:
        manifest_rows = read_jsonl(paths["manifest"])
        feature_rows = read_jsonl(paths["feature_table"])
        prosody_rows = read_jsonl(paths["prosody"])
        emotion_rows = read_jsonl(paths["emotion2vec_plus_large"])
        wavlm_rows = read_jsonl(paths["wavlm"]) if "wavlm" in paths else []
        model_info = _inspect_model(paths["model_pt"], paths["model_config"])
        expected_feature_rows = sum(len(_candidate_ids(row)) for row in manifest_rows)
        source_counts = dict(Counter(str(row.get("source_id", "unknown")) for row in manifest_rows))
        task_counts = dict(Counter(str(row.get("task_type", "unknown")) for row in manifest_rows))
        option_counts = dict(Counter(str(len(_candidate_ids(row))) for row in manifest_rows))
        checks.update(_check_manifest(manifest_rows))
        checks.update(_check_feature_table(feature_rows, manifest_rows))
        checks["manifest_row_count_is_530"] = len(manifest_rows) == 530
        checks["feature_table_row_count_matches_manifest_candidates"] = len(feature_rows) == expected_feature_rows
        checks["prosody_row_count_matches_manifest"] = len(prosody_rows) == len(manifest_rows)
        checks["emotion2vec_row_count_matches_manifest"] = len(emotion_rows) == len(manifest_rows)
        checks["wavlm_row_count_matches_manifest"] = len(wavlm_rows) == len(manifest_rows)
        checks["contains_gigaspeech_meld_emovdb"] = source_counts == {
            "gigaspeech": 200,
            "meld": 210,
            "emovdb": 120,
        }
        checks["contains_context_and_tone"] = task_counts == {
            "context_variant": 410,
            "tone_variant": 120,
        }
        checks["supports_two_and_three_candidate_questions"] = option_counts == {"2": 320, "3": 210}
        checks["checkpoint_schema_matches_config"] = (
            model_info.get("feature_schema_hash") == model_info.get("config_feature_schema_hash")
        )
        supplied_feature_groups = _supplied_feature_groups(paths)
        checks["required_audio_feature_groups_present"] = all(
            model_info.get("feature_groups", {}).get(group, 0) > 0
            for group in ("prosody", "emotion2vec", "wavlm")
        )
        checks["checkpoint_uses_only_final_feature_groups"] = set(model_info.get("feature_groups", {})) <= {
            "prosody",
            "emotion2vec",
            "wavlm",
            "task",
            "textmeta",
        }
        checks["checkpoint_feature_groups_have_inputs"] = _missing_model_input_groups(
            model_info.get("feature_groups", {}),
            supplied_feature_groups,
        ) == []
        feature_reports = {
            "manifest_rows": len(manifest_rows),
            "feature_table_rows": len(feature_rows),
            "expected_feature_table_rows": expected_feature_rows,
            "prosody_rows": len(prosody_rows),
            "emotion2vec_rows": len(emotion_rows),
            "wavlm_rows": len(wavlm_rows),
            "source_counts": source_counts,
            "task_counts": task_counts,
            "option_count_distribution": option_counts,
            "feature_hashes": {
                name: file_sha256(path)
                for name, path in paths.items()
                if name not in {"model_pt", "model_config"}
            },
            "supplied_feature_groups": sorted(supplied_feature_groups),
        }
        model_info["missing_input_feature_groups"] = _missing_model_input_groups(
            model_info.get("feature_groups", {}),
            supplied_feature_groups,
        )

    failed_checks = [name for name, passed in checks.items() if passed is not True]
    report = {
        "system_id": SYSTEM_ID,
        "passed": not missing and not failed_checks,
        "missing": missing,
        "failed_checks": failed_checks,
        "model_info": model_info,
        "feature_reports": feature_reports,
        "checks": checks,
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "no_train_on_test": True,
        "no_teacher_on_test": True,
        "no_asr_used": True,
        "fusion_used": False,
        "fallback_enabled": False,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def _predict_pair_probs(
    model: AudioDeliveryPairwiseMLP,
    feature_names: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> np.ndarray:
    x = matrix(rows, feature_names)
    x = (x - mean) / std
    with torch.inference_mode():
        probs = torch.sigmoid(model(torch.from_numpy(x).float().to(device))).detach().cpu().numpy()
    return probs


def _candidate_ids(row: Mapping[str, Any]) -> list[str]:
    candidates = row.get("candidate_audio_paths", {})
    if not isinstance(candidates, Mapping) or not candidates:
        labels = row.get("candidate_labels", [])
        if isinstance(labels, list | tuple) and labels:
            return [str(item) for item in labels]
        raise ValueError(f"row {row.get('question_id')}: missing candidate_audio_paths")
    return sorted((str(key) for key in candidates), key=lambda value: (len(value), value))


def _margin(scores: Mapping[str, float]) -> float:
    values = sorted((float(value) for value in scores.values()), reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def _inspect_model(model_path: Path, config_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu")
    feature_names = list(checkpoint.get("feature_names", []))
    assert_phase1_specialist_features_safe(feature_names)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "model_path": model_path.as_posix(),
        "config_path": config_path.as_posix(),
        "model": config.get("model"),
        "feature_count": len(feature_names),
        "config_feature_count": config.get("feature_count"),
        "feature_schema_hash": json_sha256(feature_names),
        "config_feature_schema_hash": config.get("feature_schema_hash"),
        "feature_groups": dict(Counter(_feature_group(name) for name in feature_names)),
        "hidden_dims": checkpoint.get("hidden_dims", config.get("hidden_dims")),
        "dropout": float(checkpoint.get("dropout", config.get("dropout", 0.0))),
        "forbid_candidate_text_difference": bool(
            checkpoint.get("config", {}).get("forbid_candidate_text_difference", False)
        ),
        "no_phase1_test_used": bool(config.get("no_phase1_test_used", False)),
    }


def _check_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    no_leakage = all(not (FORBIDDEN_FIELDS & set(row)) for row in rows)
    provided_text = all(
        row.get("semantic_text_source") == "json_provided"
        and row.get("use_asr_for_semantic_text") is False
        and row.get("provided_utterance_text") is True
        and row.get("provided_response_text") is True
        for row in rows
    )
    same_text = True
    audio_refs_exist = True
    for row in rows:
        response = row.get("response_text")
        transcripts = row.get("candidate_transcripts", {})
        if not isinstance(transcripts, Mapping):
            same_text = False
        else:
            for candidate_id in _candidate_ids(row):
                if transcripts.get(candidate_id) != response:
                    same_text = False
        for path in _manifest_audio_paths(row):
            if not Path(str(path)).exists():
                audio_refs_exist = False
    return {
        "manifest_no_label_leakage": no_leakage,
        "manifest_uses_json_provided_text": provided_text,
        "manifest_candidate_texts_identical": same_text,
        "manifest_audio_refs_exist": audio_refs_exist,
    }


def _check_feature_table(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    assert_training_inputs_safe(rows=rows, mode="infer")
    assert_asr_does_not_override_provided_text(rows)
    manifest_candidates = {row["question_id"]: set(_candidate_ids(row)) for row in manifest_rows}
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("question_id")].append(row)
    candidate_sets_match = True
    candidate_texts_identical = True
    for qid, expected_candidates in manifest_candidates.items():
        question_rows = grouped.get(qid, [])
        by_candidate = {str(row.get("candidate_id")): row for row in question_rows}
        if set(by_candidate) != expected_candidates:
            candidate_sets_match = False
            continue
        transcripts = {str(row.get("candidate_transcript", "")) for row in question_rows}
        if len(transcripts) != 1:
            candidate_texts_identical = False
    return {
        "feature_table_infer_safe": True,
        "feature_table_no_asr_override": True,
        "feature_table_candidate_sets_match_manifest": candidate_sets_match,
        "feature_table_candidate_texts_identical": candidate_texts_identical,
        "feature_table_no_label_leakage": all(not (FORBIDDEN_FIELDS & set(row)) for row in rows),
    }


def _submission_audit(
    submission_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_report = audit_submission_rows(submission_rows, expected_count=len(manifest_rows), manifest_rows=manifest_rows)
    qids = [row.get("question_id") for row in submission_rows]
    manifest_qids = [row["question_id"] for row in manifest_rows]
    manifest_by_qid = {row["question_id"]: row for row in manifest_rows}
    answer_by_qid = {row.get("question_id"): row.get("answer") for row in submission_rows}
    distribution_by_source: dict[str, dict[str, int]] = {}
    for source_id in sorted({str(row.get("source_id", "unknown")) for row in manifest_rows}):
        answers = [
            answer_by_qid.get(row["question_id"])
            for row in manifest_rows
            if str(row.get("source_id", "unknown")) == source_id
        ]
        distribution_by_source[source_id] = dict(Counter(str(answer) for answer in answers))
    legal_by_source = {
        source_id: sorted(
            {
                candidate_id
                for row in manifest_rows
                if str(row.get("source_id", "unknown")) == source_id
                for candidate_id in _candidate_ids(row)
            }
        )
        for source_id in distribution_by_source
    }
    report = {
        **base_report,
        "row_count": len(submission_rows),
        "unique_question_id_count": len(set(qids)),
        "question_id_order_matches_manifest": qids == manifest_qids,
        "only_question_id_and_answer": all(set(row) == {"question_id", "answer"} for row in submission_rows),
        "answer_distribution": dict(Counter(row.get("answer") for row in submission_rows)),
        "answer_distribution_by_source": distribution_by_source,
        "legal_answers_by_source": legal_by_source,
        "source_counts": dict(Counter(str(row.get("source_id", "unknown")) for row in manifest_rows)),
        "task_counts": dict(Counter(str(row.get("task_type", "unknown")) for row in manifest_rows)),
        "no_score_or_path_in_submission": all(not ({"score", "path"} & set(row)) for row in submission_rows),
        "no_label_or_gold_in_submission": all(not (FORBIDDEN_FIELDS & set(row)) for row in submission_rows),
    }
    report["passed"] = bool(
        base_report["passed"]
        and qids == manifest_qids
        and report["only_question_id_and_answer"]
        and report["no_score_or_path_in_submission"]
        and report["no_label_or_gold_in_submission"]
        and all(
            str(answer_by_qid.get(qid)) in set(_candidate_ids(manifest_by_qid[qid]))
            for qid in qids
        )
    )
    return report


def _inference_audit(
    *,
    predictions: Sequence[Mapping[str, Any]],
    submission_audit: Mapping[str, Any],
    dependency_report: Mapping[str, Any],
    model_dir: Path,
    manifest_path: Path,
    feature_table_path: Path,
    prosody_path: Path,
    emotion2vec_path: Path,
    wavlm_path: Path,
) -> dict[str, Any]:
    model_paths = {
        "audio_delivery_pairwise_model": (model_dir / "model.pt").as_posix(),
        "audio_delivery_pairwise_config": (model_dir / "config.yaml").as_posix(),
    }
    feature_paths = {
        "manifest": manifest_path,
        "feature_table": feature_table_path,
        "prosody": prosody_path,
        "emotion2vec_plus_large": emotion2vec_path,
    }
    feature_paths["wavlm"] = wavlm_path
    feature_hashes = {name: file_sha256(path) for name, path in feature_paths.items()}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_id": SYSTEM_ID,
        "primary_model": "audio_delivery_pairwise_v1",
        "prediction_rows": len(predictions),
        "answer_distribution": dict(Counter(row.get("answer") for row in predictions)),
        "answer_distribution_by_source": dict(submission_audit.get("answer_distribution_by_source", {})),
        "source_counts": dict(submission_audit.get("source_counts", {})),
        "task_counts": dict(submission_audit.get("task_counts", {})),
        "feature_hash": json_sha256(feature_hashes),
        "feature_hashes": feature_hashes,
        "model_artifact_paths": model_paths,
        "model_artifact_hashes": {name: file_sha256(Path(path)) for name, path in model_paths.items()},
        "dependency_audit": dict(dependency_report),
        "submission_audit": dict(submission_audit),
        "semantic_text_source": "json_provided",
        "use_asr_for_semantic_text": False,
        "asr_used": False,
        "candidate_texts_identical": True,
        "text_modules_used_for_candidate_decision": False,
        "fusion_used": False,
        "teacher_used": False,
        "fallback_enabled": False,
        "no_fallback": True,
        "no_train_on_test": True,
        "no_teacher_on_test": True,
        "no_asr": True,
    }


def _manifest_audio_paths(row: Mapping[str, Any]) -> list[Any]:
    paths = [row.get("utterance_audio_path")]
    candidates = row.get("candidate_audio_paths", {})
    if isinstance(candidates, Mapping):
        paths.extend(candidates.values())
    return paths


def _feature_group(name: str) -> str:
    lowered = name.lower()
    for group in ("wavlm", "emotion2vec", "prosody", "task", "textmeta"):
        if group in lowered:
            return group
    return "other"


def _supplied_feature_groups(paths: Mapping[str, Path]) -> set[str]:
    del paths
    return {"task", "textmeta", "prosody", "emotion2vec", "wavlm"}


def _missing_model_input_groups(
    model_feature_groups: Mapping[str, int],
    supplied_feature_groups: set[str],
) -> list[str]:
    optional_internal_groups = {"other", "quality", "pause"}
    required = {
        group
        for group, count in model_feature_groups.items()
        if int(count) > 0 and group not in optional_internal_groups
    }
    return sorted(required - supplied_feature_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_AUDIO_MODEL_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE.as_posix())
    parser.add_argument("--prosody", default=DEFAULT_PROSODY.as_posix())
    parser.add_argument("--emotion2vec", default=DEFAULT_EMOTION2VEC.as_posix())
    parser.add_argument("--wavlm", default=DEFAULT_WAVLM.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT_OUTPUT.as_posix())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if args.audit_only:
        report = audit_phase1_full_audio_delivery_dependencies(
            model_dir=args.model_dir,
            manifest_path=args.manifest,
            feature_table_path=args.feature_table,
            prosody_path=args.prosody,
            emotion2vec_path=args.emotion2vec,
            wavlm_path=args.wavlm,
            output_path=args.audit_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return

    report = run_phase1_full_audio_delivery_system(
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
