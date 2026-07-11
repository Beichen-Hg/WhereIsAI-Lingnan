from __future__ import annotations

import torch

from humomni.infer.final_audio_delivery_features import candidate_feature_map, pairwise_features
from humomni.models.audio_delivery_pairwise import AudioDeliveryPairwiseMLP, pairwise_bce_loss
from humomni.utils.train_guard import assert_phase1_specialist_features_safe


def _feature_rows() -> list[dict]:
    base = {
        "question_id": "q1",
        "group_id": "g1",
        "split": "train",
        "task_type": "context_variant",
        "source_id": "unit",
        "context": "context",
        "user_transcript": "user text",
        "label": "A",
    }
    return [
        {**base, "candidate_id": "A", "candidate_transcript": "same response"},
        {**base, "candidate_id": "B", "candidate_transcript": "different response not used as diff"},
    ]


def _prosody_rows() -> list[dict]:
    return [
        {
            "question_id": "q1",
            "group_id": "g1",
            "split": "train",
            "user_prosody": {"duration": 1.0, "rms_mean": 0.2},
            "candidate_prosodies": {
                "A": {"duration": 1.2, "rms_mean": 0.3},
                "B": {"duration": 0.9, "rms_mean": 0.1},
            },
        }
    ]


def _emotion_rows() -> list[dict]:
    return [
        {
            "question_id": "q1",
            "group_id": "g1",
            "split": "train",
            "user_emotion_embedding": [0.1, 0.2, 0.3],
            "candidate_emotion_embeddings": {"A": [0.2, 0.1, 0.4], "B": [0.0, 0.2, 0.1]},
        }
    ]


def test_pairwise_model_forward_and_loss() -> None:
    model = AudioDeliveryPairwiseMLP(5, hidden_dims=(4,), dropout=0.0)
    logits = model(torch.ones(3, 5))
    assert logits.shape == (3,)
    loss = pairwise_bce_loss(logits, torch.tensor([1.0, 0.0, 1.0]))
    assert float(loss.detach()) > 0.0
    weighted = pairwise_bce_loss(
        logits,
        torch.tensor([1.0, 0.0, 1.0]),
        sample_weight=torch.tensor([3.0, 1.0, 3.0]),
    )
    assert float(weighted.detach()) > 0.0


def test_final_pairwise_feature_builder_forbids_text_diff_terms() -> None:
    row_a = _feature_rows()[0]
    row_b = _feature_rows()[1]
    prosody = _prosody_rows()[0]
    emotion = _emotion_rows()[0]
    features_a = candidate_feature_map(
        base_row=row_a,
        candidate_id="A",
        prosody_row=prosody,
        emotion_row=emotion,
        wavlm_row={},
    )
    features_b = candidate_feature_map(
        base_row=row_b,
        candidate_id="B",
        prosody_row=prosody,
        emotion_row=emotion,
        wavlm_row={},
    )
    feature_names = list(pairwise_features(features_a, features_b))
    assert_phase1_specialist_features_safe(feature_names)
    assert not any("candidate_text_difference" in name for name in feature_names)
    assert any("prosody" in name for name in feature_names)
    assert any("emotion2vec" in name for name in feature_names)
