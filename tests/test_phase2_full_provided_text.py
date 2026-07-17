from __future__ import annotations

import json
from pathlib import Path

from humomni.data.build_phase1_full_test_manifest import Phase1ReleaseSpec
from humomni.data.build_phase2_full_test_manifest import (
    build_phase2_full_provided_text_manifest,
)
from humomni.utils.io import read_jsonl
from humomni.utils.train_guard import assert_training_inputs_safe


def test_phase2_manifest_normalizes_data_index_and_blocks_training(tmp_path: Path) -> None:
    root = tmp_path / "phase2-gigaspeech"
    utterance_dir = root / "phase2-test_gigaspeech"
    option_dir = root / "phase2-test_gigaspeech_options"
    utterance_dir.mkdir(parents=True)
    option_dir.mkdir()
    (utterance_dir / "u.wav").write_bytes(b"")
    (option_dir / "4_1_opt-A.wav").write_bytes(b"")
    (option_dir / "4_1_opt-B.wav").write_bytes(b"")
    release = root / "release.json"
    release.write_text(
        json.dumps(
            [
                {
                    "data_index": "4_1",
                    "context": "context",
                    "utterance": "user text",
                    "utterance_audio": "./phase2-test_gigaspeech/u.wav",
                    "response": "response text",
                    "options": {
                        "opt-A": "./phase2-test_gigaspeech_options/4_1_opt-A.wav",
                        "opt-B": "./phase2-test_gigaspeech_options/4_1_opt-B.wav",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    spec = Phase1ReleaseSpec(
        source_id="gigaspeech",
        task_type="context_variant",
        release_json=release,
        data_root=root,
        expected_rows=1,
        expected_groups=1,
        expected_option_count=2,
        question_id_field="data_index",
        split="phase2_test",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    report = build_phase2_full_provided_text_manifest(
        output_manifest=manifest_path,
        output_report=tmp_path / "report.json",
        specs=[spec],
    )

    rows = read_jsonl(manifest_path)
    assert report["passed"] is True
    assert rows[0]["question_id"] == "4_1"
    assert rows[0]["group_id"] == "gigaspeech_4"
    assert rows[0]["split"] == "phase2_test"
    try:
        assert_training_inputs_safe(rows=rows, mode="supervised_train")
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("expected Phase 2 manifest to be blocked from supervised training")
