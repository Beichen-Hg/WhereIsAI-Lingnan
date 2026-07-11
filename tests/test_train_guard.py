from __future__ import annotations

import pytest

from humomni.utils.train_guard import assert_training_inputs_safe


def test_train_guard_blocks_phase1_test_in_supervised_training() -> None:
    with pytest.raises(ValueError, match="test split"):
        assert_training_inputs_safe(
            rows=[{"question_id": "q1", "split": "phase1_test", "label": "A"}],
            mode="supervised_train",
        )


def test_train_guard_allows_train_valid_labels_in_supervised_training() -> None:
    assert_training_inputs_safe(
        rows=[
            {"question_id": 1, "split": "train", "label": "A"},
            {"question_id": 2, "split": "valid", "label": "B"},
        ],
        mode="supervised_train",
    )


def test_train_guard_blocks_labels_outside_supervised_training() -> None:
    with pytest.raises(ValueError, match="outside supervised"):
        assert_training_inputs_safe(
            rows=[{"question_id": 1, "split": "valid", "label": "A"}],
            mode="infer",
        )


def test_train_guard_blocks_path_marker() -> None:
    with pytest.raises(ValueError, match="test split"):
        assert_training_inputs_safe(
            paths=["artifacts/features/feat-phase1-test/phase1_test/feature_table.jsonl"],
            mode="supervised_train",
        )
