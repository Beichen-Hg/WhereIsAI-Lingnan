from __future__ import annotations

from humomni.infer.audit_submission import audit_submission_rows
from humomni.infer.make_submission import make_submission_rows


def test_make_submission_outputs_only_question_id_and_answer():
    predictions = [
        {
            "question_id": 0,
            "answer": "B",
            "candidate_scores": {"A": 0.2, "B": 0.8},
            "score": 0.8,
        }
    ]

    submission = make_submission_rows(predictions)

    assert submission == [{"question_id": 0, "answer": "B"}]


def test_submission_with_extra_fields_fails_audit():
    rows = [{"question_id": 0, "answer": "A", "score": 0.9}]

    report = audit_submission_rows(rows, expected_count=1)

    assert not report["passed"]
    assert any("unexpected fields" in error for error in report["errors"])


def test_submission_duplicate_question_id_fails_audit():
    rows = [
        {"question_id": 0, "answer": "A"},
        {"question_id": 0, "answer": "B"},
    ]

    report = audit_submission_rows(rows, expected_count=2)

    assert not report["passed"]
    assert any("duplicate question_id" in error for error in report["errors"])
