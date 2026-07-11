from __future__ import annotations

import json
from pathlib import Path

import torch

from humomni.data.build_phase1_full_test_manifest import (
    Phase1ReleaseSpec,
    build_phase1_full_provided_text_manifest,
)
from humomni.features.build_feature_table import build_feature_table_rows
from humomni.infer.audit_submission import audit_submission_rows
from humomni.infer.phase1_full_audio_delivery_system import (
    audit_phase1_full_audio_delivery_dependencies,
    predict_full_phase1_audio_delivery,
)
from humomni.models.audio_delivery_pairwise import AudioDeliveryPairwiseMLP
from humomni.utils.hashing import json_sha256
from humomni.utils.io import read_jsonl, write_json, write_jsonl
from humomni.utils.train_guard import assert_training_inputs_safe


def test_full_phase1_manifest_supports_gigaspeech_meld_emovdb(tmp_path: Path) -> None:
    specs = [
        _write_release(tmp_path, "gigaspeech", "context_variant", 2, 2, "phase1-test_gigaspeech"),
        _write_release(tmp_path, "meld", "context_variant", 3, 3, "phase1-test_meld"),
        _write_release(tmp_path, "emovdb", "tone_variant", 2, 2, "phase1-test_emovdb_"),
    ]
    output = tmp_path / "full_manifest.jsonl"
    report_path = tmp_path / "report.json"

    report = build_phase1_full_provided_text_manifest(
        output_manifest=output,
        output_report=report_path,
        specs=specs,
    )
    rows = read_jsonl(output)

    assert report["passed"] is True
    assert report["num_rows"] == 7
    assert report["source_counts"] == {"gigaspeech": 2, "meld": 3, "emovdb": 2}
    assert report["task_counts"] == {"context_variant": 5, "tone_variant": 2}
    assert report["option_count_distribution"] == {"2": 4, "3": 3}
    meld_row = next(row for row in rows if row["source_id"] == "meld")
    assert meld_row["candidate_labels"] == ["A", "B", "C"]
    assert meld_row["candidate_transcripts"] == {
        "A": meld_row["response_text"],
        "B": meld_row["response_text"],
        "C": meld_row["response_text"],
    }
    assert meld_row["semantic_text_source"] == "json_provided"
    assert meld_row["use_asr_for_semantic_text"] is False
    assert not ({"label", "gold", "answer", "goodPara", "badPara", "is_gold_candidate"} & set(meld_row))


def test_feature_table_expands_three_candidate_provided_text_rows() -> None:
    manifest = [
        {
            "question_id": "meld_183_1",
            "group_id": "meld_183",
            "task_type": "context_variant",
            "source_id": "meld",
            "context": "context",
            "user_transcript": "user",
            "response_text": "same response",
            "candidate_transcripts": {"A": "same response", "B": "same response", "C": "same response"},
            "utterance_audio_path": "u.wav",
            "candidate_audio_paths": {"A": "a.wav", "B": "b.wav", "C": "c.wav"},
            "provided_utterance_text": True,
            "provided_response_text": True,
            "semantic_text_source": "json_provided",
            "use_asr_for_semantic_text": False,
            "split": "phase1_test",
        }
    ]

    rows = build_feature_table_rows(manifest_rows=manifest, mode="test")

    assert len(rows) == 3
    assert {row["candidate_id"] for row in rows} == {"A", "B", "C"}
    assert {row["candidate_transcript"] for row in rows} == {"same response"}
    assert all(row["asr_used"] is False for row in rows)


def test_three_candidate_pairwise_tournament_can_select_c(tmp_path: Path) -> None:
    model_dir = _write_linear_pairwise_checkpoint(tmp_path)
    manifest = [
        {
            "question_id": "meld_183_1",
            "group_id": "meld_183",
            "task_type": "context_variant",
            "source_id": "meld",
            "context": "context",
            "user_transcript": "user",
            "response_text": "same response",
            "candidate_transcripts": {"A": "same response", "B": "same response", "C": "same response"},
            "utterance_audio_path": "u.wav",
            "candidate_audio_paths": {"A": "a.wav", "B": "b.wav", "C": "c.wav"},
            "provided_utterance_text": True,
            "provided_response_text": True,
            "semantic_text_source": "json_provided",
            "use_asr_for_semantic_text": False,
            "split": "phase1_test",
        }
    ]
    feature_rows = build_feature_table_rows(manifest_rows=manifest, mode="test")
    prosody = [
        {
            "question_id": "meld_183_1",
            "group_id": "meld_183",
            "split": "phase1_test",
            "user_prosody": {"rms_mean": 0.0},
            "candidate_prosodies": {
                "A": {"rms_mean": 0.1},
                "B": {"rms_mean": 0.2},
                "C": {"rms_mean": 0.9},
            },
            "prosody_meta": {"backend": "unit"},
            "user_audio_hash": "u",
            "candidate_audio_hashes": {"A": "a", "B": "b", "C": "c"},
        }
    ]
    emotion = [
        {
            "question_id": "meld_183_1",
            "group_id": "meld_183",
            "split": "phase1_test",
            "user_emotion_embedding": [0.0],
            "candidate_emotion_embeddings": {"A": [0.0], "B": [0.0], "C": [0.0]},
            "embedding_meta": {"model_id": "unit"},
            "user_audio_hash": "u",
            "candidate_audio_hashes": {"A": "a", "B": "b", "C": "c"},
        }
    ]

    predictions = predict_full_phase1_audio_delivery(
        model_dir=model_dir,
        manifest_rows=manifest,
        feature_table_rows=feature_rows,
        prosody_rows=prosody,
        emotion_rows=emotion,
        wavlm_rows=[],
        device="cpu",
    )

    assert len(predictions) == 1
    assert predictions[0]["answer"] == "C"
    assert set(predictions[0]["candidate_scores"]) == {"A", "B", "C"}
    assert "C>A" in predictions[0]["pairwise_scores"]
    assert predictions[0]["asr_used"] is False


def test_submission_audit_allows_manifest_defined_c_answer() -> None:
    manifest = [{"question_id": "meld_183_1", "candidate_audio_paths": {"A": "a.wav", "B": "b.wav", "C": "c.wav"}}]
    report = audit_submission_rows([{"question_id": "meld_183_1", "answer": "C"}], manifest_rows=manifest)
    assert report["passed"] is True


def test_train_guard_blocks_full_phase1_feature_path() -> None:
    try:
        assert_training_inputs_safe(
            paths=["artifacts/features/feat-phase1-full-provided-text/phase1_test/feature_table.jsonl"],
            mode="supervised_train",
        )
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("expected full Phase1 feature path to be blocked")


def test_dependency_audit_rejects_nonfinal_checkpoint_feature_groups(tmp_path: Path) -> None:
    manifest_path, feature_path, prosody_path, emotion_path, wavlm_path = _write_minimal_full_phase1_inputs(tmp_path)
    model_dir = _write_checkpoint_requiring_nonfinal_feature(tmp_path)

    report = audit_phase1_full_audio_delivery_dependencies(
        model_dir=model_dir,
        manifest_path=manifest_path,
        feature_table_path=feature_path,
        prosody_path=prosody_path,
        emotion2vec_path=emotion_path,
        wavlm_path=wavlm_path,
        output_path=None,
    )

    assert report["passed"] is False
    assert "checkpoint_uses_only_final_feature_groups" in report["failed_checks"]


def _write_release(
    tmp_path: Path,
    source_id: str,
    task_type: str,
    options: int,
    rows: int,
    audio_dir_name: str,
) -> Phase1ReleaseSpec:
    root = tmp_path / source_id
    utterance_dir = root / audio_dir_name
    option_dir = root / f"{audio_dir_name}_options"
    if source_id == "meld":
        option_dir = root / "phase1-test_meld_options"
    if source_id == "emovdb":
        option_dir = root / "phase1-test_emovdb_options_"
    utterance_dir.mkdir(parents=True)
    option_dir.mkdir(parents=True)
    release_rows = []
    for index in range(rows):
        qid = _qid(source_id, index)
        user_audio = utterance_dir / f"{qid}.wav"
        user_audio.write_bytes(b"")
        option_payload = {}
        for candidate_index in range(options):
            label = chr(ord("A") + candidate_index)
            candidate_audio = option_dir / f"{qid}_opt-{label}.wav"
            candidate_audio.write_bytes(b"")
            option_payload[f"opt-{label}"] = f"./{option_dir.name}/{candidate_audio.name}"
        release_rows.append(
            {
                "question_id": qid,
                "utterance": f"utterance {qid}",
                "utterance_audio": f"./{utterance_dir.name}/{user_audio.name}",
                "context": f"context {qid}",
                "response": f"response {qid}",
                "options": option_payload,
            }
        )
    release = root / f"{source_id}_release.json"
    release.write_text(json.dumps(release_rows), encoding="utf-8")
    return Phase1ReleaseSpec(
        source_id=source_id,
        task_type=task_type,
        release_json=release,
        data_root=root,
        expected_rows=rows,
        expected_groups=rows if source_id != "gigaspeech" else rows,
        expected_option_count=options,
    )


def _qid(source_id: str, index: int) -> str:
    if source_id == "gigaspeech":
        return f"gigaspeech_{index}_1"
    if source_id == "meld":
        return f"meld_{index}_1"
    return f"emovdb_{index:04d}_e1"


def _write_linear_pairwise_checkpoint(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    feature_name = "diff_prosody_candidate_prosody_rms_mean"
    model = AudioDeliveryPairwiseMLP(1, hidden_dims=(), dropout=0.0)
    with torch.no_grad():
        model.network[0].weight.fill_(8.0)
        model.network[0].bias.zero_()
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_names": [feature_name],
        "mean": [0.0],
        "std": [1.0],
        "hidden_dims": [],
        "dropout": 0.0,
        "config": {"forbid_candidate_text_difference": True, "no_phase1_test_used": True},
    }
    torch.save(checkpoint, model_dir / "model.pt")
    write_json(
        model_dir / "config.yaml",
        {
            "model": "AudioDeliveryPairwiseMLP",
            "feature_count": 1,
            "feature_schema_hash": json_sha256([feature_name]),
            "hidden_dims": [],
            "dropout": 0.0,
            "no_phase1_test_used": True,
        },
    )
    return model_dir


def _write_checkpoint_requiring_nonfinal_feature(tmp_path: Path) -> Path:
    model_dir = tmp_path / "nonfinal_feature_model"
    model_dir.mkdir()
    feature_names = [
        "diff_prosody_candidate_prosody_rms_mean",
        "diff_emotion2vec_000",
        "diff_wavlm_000",
        "diff_unsupported_legacy_feature",
    ]
    model = AudioDeliveryPairwiseMLP(len(feature_names), hidden_dims=(), dropout=0.0)
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "mean": [0.0] * len(feature_names),
        "std": [1.0] * len(feature_names),
        "hidden_dims": [],
        "dropout": 0.0,
        "config": {"forbid_candidate_text_difference": True, "no_phase1_test_used": True},
    }
    torch.save(checkpoint, model_dir / "model.pt")
    write_json(
        model_dir / "config.yaml",
        {
            "model": "AudioDeliveryPairwiseMLP",
            "feature_count": len(feature_names),
            "feature_schema_hash": json_sha256(feature_names),
            "hidden_dims": [],
            "dropout": 0.0,
            "no_phase1_test_used": True,
        },
    )
    return model_dir


def _write_minimal_full_phase1_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    user_audio = tmp_path / "u.wav"
    a_audio = tmp_path / "a.wav"
    b_audio = tmp_path / "b.wav"
    for path in (user_audio, a_audio, b_audio):
        path.write_bytes(b"")
    manifest = [
        {
            "question_id": "gigaspeech_0_1",
            "group_id": "gigaspeech_0",
            "task_type": "context_variant",
            "source_id": "gigaspeech",
            "context": "context",
            "user_transcript": "utterance",
            "response_text": "same response",
            "candidate_transcripts": {"A": "same response", "B": "same response"},
            "utterance_audio_path": user_audio.as_posix(),
            "candidate_audio_paths": {"A": a_audio.as_posix(), "B": b_audio.as_posix()},
            "provided_utterance_text": True,
            "provided_response_text": True,
            "semantic_text_source": "json_provided",
            "use_asr_for_semantic_text": False,
            "split": "phase1_test",
        }
    ]
    feature_rows = build_feature_table_rows(manifest_rows=manifest, mode="test")
    prosody = [
        {
            "question_id": "gigaspeech_0_1",
            "group_id": "gigaspeech_0",
            "split": "phase1_test",
            "user_prosody": {"rms_mean": 0.0},
            "candidate_prosodies": {"A": {"rms_mean": 0.1}, "B": {"rms_mean": 0.2}},
        }
    ]
    emotion = [
        {
            "question_id": "gigaspeech_0_1",
            "group_id": "gigaspeech_0",
            "split": "phase1_test",
            "user_emotion_embedding": [0.0],
            "candidate_emotion_embeddings": {"A": [0.1], "B": [0.2]},
        }
    ]
    wavlm = [
        {
            "question_id": "gigaspeech_0_1",
            "group_id": "gigaspeech_0",
            "split": "phase1_test",
            "user_wavlm_embedding": [0.0],
            "candidate_wavlm_embeddings": {"A": [0.1], "B": [0.2]},
        }
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    feature_path = tmp_path / "feature_table.jsonl"
    prosody_path = tmp_path / "prosody.jsonl"
    emotion_path = tmp_path / "emotion.jsonl"
    wavlm_path = tmp_path / "wavlm.jsonl"
    write_jsonl(manifest_path, manifest)
    write_jsonl(feature_path, feature_rows)
    write_jsonl(prosody_path, prosody)
    write_jsonl(emotion_path, emotion)
    write_jsonl(wavlm_path, wavlm)
    return manifest_path, feature_path, prosody_path, emotion_path, wavlm_path
