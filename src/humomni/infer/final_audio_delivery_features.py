"""Feature construction used exclusively by the final audio-delivery scorer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from humomni.utils.io import read_jsonl
from humomni.utils.train_guard import assert_phase1_specialist_features_safe


EMBEDDING_HEAD_DIMS = 32


def candidate_feature_map(
    *,
    base_row: Mapping[str, Any],
    candidate_id: str,
    prosody_row: Mapping[str, Any],
    emotion_row: Mapping[str, Any],
    wavlm_row: Mapping[str, Any],
) -> dict[str, float]:
    """Construct final-model features for one candidate without text comparisons."""

    features: dict[str, float] = {}
    task_type = str(base_row.get("task_type", "unknown"))
    for task in ("context_variant", "tone_variant", "unknown"):
        features[f"task_{task}"] = float(task_type == task)

    context = str(base_row.get("context", ""))
    response = str(base_row.get("candidate_transcript", ""))
    user_text = str(base_row.get("user_transcript", ""))
    features.update(
        {
            "textmeta_context_char_len": float(len(context)),
            "textmeta_context_word_len": float(len(context.split())),
            "textmeta_response_char_len": float(len(response)),
            "textmeta_response_word_len": float(len(response.split())),
            "textmeta_user_char_len": float(len(user_text)),
            "textmeta_user_word_len": float(len(user_text.split())),
        }
    )

    user_prosody, candidate_prosody = _resolve_dict_pair(
        prosody_row,
        candidate_id,
        "user_prosody",
        ("candidate_prosodies", "candidate_prosody"),
    )
    features.update(
        {f"prosody_{key}": value for key, value in _prosody_features(user_prosody, candidate_prosody).items()}
    )

    user_emotion, candidate_emotion = _resolve_embedding_pair(
        emotion_row,
        candidate_id,
        "user_emotion_embedding",
        ("candidate_emotion_embeddings", "candidate_audio_embeddings"),
    )
    features.update(
        {
            f"emotion2vec_{key}": value
            for key, value in _embedding_summary_features(user_emotion, candidate_emotion).items()
        }
    )

    user_wavlm, candidate_wavlm = _resolve_embedding_pair(
        wavlm_row,
        candidate_id,
        "user_wavlm_embedding",
        ("candidate_wavlm_embeddings",),
    )
    features.update(
        {
            f"wavlm_{key}": value
            for key, value in _embedding_summary_features(user_wavlm, candidate_wavlm).items()
        }
    )
    return {key: float(value) for key, value in features.items() if _safe_feature_name(key)}


def pairwise_features(left_features: Mapping[str, float], right_features: Mapping[str, float]) -> dict[str, float]:
    """Build the A-versus-B pairwise representation expected by the checkpoint."""

    output: dict[str, float] = {}
    for name in sorted(set(left_features) | set(right_features)):
        left = float(left_features.get(name, 0.0))
        right = float(right_features.get(name, 0.0))
        output[f"A_{name}"] = left
        output[f"B_{name}"] = right
        output[f"diff_{name}"] = left - right
        output[f"absdiff_{name}"] = abs(left - right)
        output[f"ratio_{name}"] = _safe_ratio(left, right)
    assert_phase1_specialist_features_safe(output)
    return output


def matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    assert_phase1_specialist_features_safe(feature_names)
    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float32,
    )


def read_optional_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    candidate = Path(path)
    return read_jsonl(candidate) if candidate.exists() else []


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return requested


def _resolve_dict_pair(
    row: Mapping[str, Any],
    candidate_id: str,
    user_key: str,
    candidate_keys: Sequence[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    user = row.get(user_key, {})
    candidates: Any = {}
    for key in candidate_keys:
        if key in row:
            candidates = row.get(key, {})
            break
    candidate = candidates.get(candidate_id, {}) if isinstance(candidates, Mapping) else {}
    return user if isinstance(user, Mapping) else {}, candidate if isinstance(candidate, Mapping) else {}


def _resolve_embedding_pair(
    row: Mapping[str, Any],
    candidate_id: str,
    user_key: str,
    candidate_keys: Sequence[str],
) -> tuple[list[float], list[float]]:
    user = _float_list(row.get(user_key, []))
    candidates: Any = {}
    for key in candidate_keys:
        if key in row:
            candidates = row.get(key, {})
            break
    candidate = candidates.get(candidate_id, []) if isinstance(candidates, Mapping) else []
    return user, _float_list(candidate)


def _prosody_features(user_prosody: Mapping[str, Any], candidate_prosody: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key in sorted(set(user_prosody) | set(candidate_prosody)):
        if _forbidden_name(key):
            continue
        user = _float(user_prosody.get(key))
        candidate = _float(candidate_prosody.get(key))
        safe_key = _safe_key(str(key))
        features[f"user_prosody_{safe_key}"] = user
        features[f"candidate_prosody_{safe_key}"] = candidate
        features[f"prosody_{safe_key}_diff"] = candidate - user
        features[f"prosody_{safe_key}_abs_diff"] = abs(candidate - user)
        features[f"prosody_{safe_key}_ratio"] = _safe_ratio(candidate, user)
    return features


def _embedding_summary_features(user_embedding: Sequence[Any], candidate_embedding: Sequence[Any]) -> dict[str, float]:
    user = _float_list(user_embedding)
    candidate = _float_list(candidate_embedding)
    dim = min(len(user), len(candidate))
    if dim <= 0:
        return {"emotion2vec_embedding_dim": 0.0, "emotion2vec_failed": 1.0}
    user = user[:dim]
    candidate = candidate[:dim]
    diff = [candidate_value - user_value for user_value, candidate_value in zip(user, candidate, strict=True)]
    abs_diff = [abs(value) for value in diff]
    features = {
        "emotion2vec_embedding_dim": float(dim),
        "emotion2vec_failed": 0.0,
        "emotion2vec_user_norm": _l2(user),
        "emotion2vec_candidate_norm": _l2(candidate),
        "emotion2vec_cosine": _cosine(user, candidate),
        "emotion2vec_diff_mean": _mean(diff),
        "emotion2vec_diff_std": _std(diff),
        "emotion2vec_abs_diff_mean": _mean(abs_diff),
        "emotion2vec_abs_diff_std": _std(abs_diff),
        "emotion2vec_abs_diff_max": max(abs_diff) if abs_diff else 0.0,
    }
    for index in range(dim):
        features[f"emotion2vec_user_{index}"] = float(user[index])
        features[f"emotion2vec_candidate_{index}"] = float(candidate[index])
        features[f"emotion2vec_diff_{index}"] = float(diff[index])
        features[f"emotion2vec_abs_diff_{index}"] = abs(float(diff[index]))
    return features


def _safe_feature_name(name: Any) -> bool:
    lowered = str(name).lower()
    return not any(
        term in lowered
        for term in (
            "path",
            "hash",
            "audio_id",
            "candidate_text",
            "candidate_asr_semantic",
            "candidate_transcript_ngram",
        )
    )


def _forbidden_name(name: Any) -> bool:
    lowered = str(name).lower()
    return any(term in lowered for term in ("path", "hash", "audio_id", "candidate_text"))


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() or character == "_" else "_" for character in value.lower())


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list | tuple):
        return []
    return [_float(value) for value in values[:EMBEDDING_HEAD_DIMS]]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-8 else 0.0


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = _l2(left) * _l2(right)
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True)) / denominator if denominator > 1e-8 else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
