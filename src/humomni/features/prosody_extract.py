"""Prosody feature extraction with split-safe cache and leakage checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from humomni.features.feature_store import (
    build_feature_store,
    cache_root_from_config,
    extractor_config_hash,
    extractor_settings,
    feature_version_from_config,
    metadata_for_row,
)
from humomni.utils.io import read_jsonl, read_yaml, write_json, write_jsonl

STRICT_INFER_KEYS = frozenset({"label", "gold", "answer", "is_gold_candidate"})
LEAKAGE_VALUE_TERMS = ("goodpara", "badpara")


class ProsodyExtractor(Protocol):
    extractor_name: str
    extractor_config: Mapping[str, Any]

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        """Extract prosody fields for a manifest row."""


class BasicLibrosaProsodyExtractor:
    """Compute lightweight prosody features from local audio files."""

    extractor_name = "librosa_basic_prosody"

    def __init__(
        self,
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        top_db: float = 30.0,
        **_: Any,
    ) -> None:
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.top_db = float(top_db)
        self.extractor_config = {
            "backend": "librosa_basic",
            "frame_length": int(frame_length),
            "hop_length": int(hop_length),
            "top_db": self.top_db,
        }
        self._version_info = _audio_dependency_versions()

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_audio_paths = manifest_row.get("candidate_audio_paths", {})
        if not isinstance(candidate_audio_paths, Mapping) or not candidate_audio_paths:
            raise ValueError("manifest row must include candidate_audio_paths")
        utterance_audio_path = manifest_row.get("utterance_audio_path")
        if not isinstance(utterance_audio_path, str):
            raise ValueError("manifest row must include utterance_audio_path")

        candidate_prosodies = {
            str(candidate_id): extract_audio_prosody(
                str(audio_path),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                top_db=self.top_db,
            )
            for candidate_id, audio_path in sorted(candidate_audio_paths.items())
        }
        return {
            "question_id": manifest_row["question_id"],
            "group_id": manifest_row.get("group_id"),
            "split": manifest_row.get("split"),
            "user_audio_hash": _safe_audio_id(utterance_audio_path),
            "candidate_audio_hashes": {
                str(candidate_id): _safe_audio_id(str(audio_path))
                for candidate_id, audio_path in sorted(candidate_audio_paths.items())
            },
            "user_prosody": extract_audio_prosody(
                utterance_audio_path,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                top_db=self.top_db,
            ),
            "candidate_prosodies": candidate_prosodies,
            # Backward-compatible alias for earlier feature-table code.
            "candidate_prosody": candidate_prosodies,
            "prosody_meta": {
                "backend": "librosa_basic",
                "version": self._version_info,
                "config_hash": extractor_config_hash(self.extractor_config),
            },
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "librosa_basic",
            "extractor_name": self.extractor_name,
            "version": self._version_info,
            "config_hash": extractor_config_hash(self.extractor_config),
        }


def extract_audio_prosody(
    audio_path: str,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 30.0,
) -> dict[str, float]:
    """Compute duration, energy, spectral and pause proxy features."""

    librosa, np, sf = _require_audio_dependencies()
    signal, sample_rate = sf.read(audio_path, always_2d=False)
    y = np.asarray(signal, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.size == 0:
        return {
            "duration": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "zero_crossing_rate": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "zero_crossing_rate_std": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_centroid_std": 0.0,
            "spectral_rolloff_mean": 0.0,
            "spectral_rolloff_std": 0.0,
            "tempo_proxy": 0.0,
            "speech_rate_proxy": 0.0,
            "silence_ratio": 1.0,
            "pause_ratio_proxy": 1.0,
        }

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
    )[0]
    try:
        tempo_values = librosa.beat.tempo(y=y, sr=sample_rate)
        tempo_proxy = float(tempo_values[0]) if len(tempo_values) else 0.0
    except Exception:
        tempo_proxy = 0.0

    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    pause_threshold = max(rms_mean * 0.1, 1e-8)
    pause_ratio = float(np.mean(rms < pause_threshold)) if rms.size else 1.0
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    return {
        "duration": float(y.size / sample_rate),
        "rms_mean": rms_mean,
        "rms_std": float(np.std(rms)) if rms.size else 0.0,
        "zero_crossing_rate": zcr_mean,
        "zero_crossing_rate_mean": zcr_mean,
        "zero_crossing_rate_std": float(np.std(zcr)) if zcr.size else 0.0,
        "spectral_centroid_mean": float(np.mean(centroid)) if centroid.size else 0.0,
        "spectral_centroid_std": float(np.std(centroid)) if centroid.size else 0.0,
        "spectral_rolloff_mean": float(np.mean(rolloff)) if rolloff.size else 0.0,
        "spectral_rolloff_std": float(np.std(rolloff)) if rolloff.size else 0.0,
        "tempo_proxy": tempo_proxy,
        "speech_rate_proxy": tempo_proxy,
        "silence_ratio": _silence_ratio(librosa, y=y, top_db=top_db),
        "pause_ratio_proxy": pause_ratio,
    }


def extract_prosody_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    extractor: ProsodyExtractor,
    *,
    store_metadata: Mapping[str, str],
    mode: str = "train",
) -> list[dict[str, Any]]:
    if mode.lower() in {"infer", "test"}:
        assert_no_infer_leakage(manifest_rows)
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        feature_row = extractor.extract_row(manifest_row)
        feature_row.update(store_metadata)
        feature_row["extractor_name"] = extractor.extractor_name
        rows.append(feature_row)
    return rows


def validate_prosody_cache_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> None:
    if mode.lower() in {"infer", "test"}:
        assert_no_infer_leakage(rows)
    for row_index, row in enumerate(rows):
        for field in (
            "question_id",
            "group_id",
            "split",
            "user_audio_hash",
            "candidate_audio_hashes",
            "user_prosody",
            "candidate_prosodies",
            "prosody_meta",
        ):
            if field not in row:
                raise ValueError(f"prosody row {row_index}: missing {field}")


def build_prosody_quality_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    extractor: ProsodyExtractor,
) -> dict[str, Any]:
    user_items = [row.get("user_prosody", {}) for row in rows]
    candidate_items = [
        prosody
        for row in rows
        for prosody in (
            row.get("candidate_prosodies")
            if isinstance(row.get("candidate_prosodies"), Mapping)
            else {}
        ).values()
    ]
    all_items = [
        item for item in [*user_items, *candidate_items] if isinstance(item, Mapping)
    ]
    durations = [_finite_float(item.get("duration")) for item in all_items]
    empty_features = sum(1 for item in all_items if not item)
    backend_counts: dict[str, int] = {}
    for row in rows:
        meta = row.get("prosody_meta", {})
        backend = str(meta.get("backend", "unknown")) if isinstance(meta, Mapping) else "unknown"
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    model_info = extractor.model_info() if hasattr(extractor, "model_info") else {}
    return {
        "total_rows": len(rows),
        "total_user_audios": len(user_items),
        "total_candidate_audios": len(candidate_items),
        "empty_feature_count": empty_features,
        "empty_feature_rate": empty_features / len(all_items) if all_items else 0.0,
        "duration": _length_stats(durations),
        "backend_distribution": dict(sorted(backend_counts.items())),
        "model_info": model_info,
    }


def assert_no_infer_leakage(rows: Any) -> None:
    hits = _leakage_hits(rows)
    if hits:
        raise ValueError(f"prosody input/cache contains leakage fields/terms: {hits[:8]}")


def _require_audio_dependencies() -> tuple[Any, Any, Any]:
    missing = [
        package
        for package in ("librosa", "numpy", "soundfile")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise ImportError(
            "Prosody extraction requires optional audio dependencies: "
            f"{', '.join(missing)}. Install them before running prosody extraction."
        )
    import librosa
    import numpy as np
    import soundfile as sf

    return librosa, np, sf


def _build_extractor(name: str, config: Mapping[str, Any]) -> ProsodyExtractor:
    normalized = name.lower()
    if normalized in {
        "basic",
        "basic_librosa",
        "basic_librosa_prosody",
        "librosa_basic",
        "librosa_basic_prosody",
    }:
        return BasicLibrosaProsodyExtractor(**dict(config))
    raise ValueError("the final method only supports librosa_basic prosody extraction")


def _stable_prosody_output_path(
    config: Mapping[str, Any],
    split: str,
    *,
    feature_version: str,
) -> Path:
    return cache_root_from_config(config) / feature_version / split / "prosody.jsonl"


def _stable_prosody_quality_report_path(
    config: Mapping[str, Any],
    split: str,
    *,
    feature_version: str,
) -> Path:
    return (
        cache_root_from_config(config)
        / feature_version
        / split
        / "prosody_quality_report.json"
    )


def _safe_audio_id(path: str) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]


def _audio_dependency_versions() -> dict[str, str]:
    try:
        librosa, np, sf = _require_audio_dependencies()
    except ImportError:
        return {}
    return {
        "librosa": str(getattr(librosa, "__version__", "unknown")),
        "numpy": str(getattr(np, "__version__", "unknown")),
        "soundfile": str(getattr(sf, "__version__", "unknown")),
    }


def _silence_ratio(librosa: Any, *, y: Any, top_db: float) -> float:
    if len(y) == 0:
        return 1.0
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
    except Exception:
        return 0.0
    voiced = sum(int(end) - int(start) for start, end in intervals)
    return float(max(0.0, min(1.0, 1.0 - voiced / max(len(y), 1))))


def _length_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values))) - 1)
    return {
        "mean": float(sum(sorted_values) / len(sorted_values)),
        "p95": float(sorted_values[p95_index]),
        "max": float(sorted_values[-1]),
    }


def _safe_feature_name(name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in name)


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _leakage_hits(value: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_lower = str(key).lower()
            child_path = f"{path}.{key}"
            if key_lower in STRICT_INFER_KEYS:
                hits.append(child_path)
            hits.extend(_leakage_hits(child, path=child_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            hits.extend(_leakage_hits(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(term in lowered for term in LEAKAGE_VALUE_TERMS):
            hits.append(path)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional local extractor config path.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL path.")
    parser.add_argument("--split", required=True, help="Split name for cache key.")
    parser.add_argument("--mode", default=None, help="Mode for leakage checks.")
    parser.add_argument("--extractor", default=None, help="Override extractor name.")
    parser.add_argument("--table-name", default="prosody", help="Output table name.")
    parser.add_argument("--feature-version", default=None, help="Override feature version.")
    parser.add_argument("--output", default=None, help="Override stable prosody JSONL output.")
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Override prosody quality report JSON path.",
    )
    args = parser.parse_args()

    config = read_yaml(args.config) if args.config and Path(args.config).exists() else {}
    _, configured_config = extractor_settings(config, "prosody") if config else ("librosa_basic", {})
    extractor_name = args.extractor or "librosa_basic"
    feature_version = args.feature_version or "feat-phase1-full-provided-text"
    extractor = _build_extractor(extractor_name, configured_config)
    mode = (args.mode or args.split).lower()
    manifest_rows = read_jsonl(args.manifest)
    store = build_feature_store(
        cache_root=cache_root_from_config(config),
        feature_version=feature_version,
        split=args.split,
        extractor_name=extractor.extractor_name,
        extractor_config=extractor.extractor_config,
        manifest_path=args.manifest,
    )
    rows = extract_prosody_rows(
        manifest_rows,
        extractor,
        store_metadata=metadata_for_row(store),
        mode=mode,
    )
    validate_prosody_cache_rows(rows, mode=mode)
    nested_output_path = store.write_table(rows, table_name=args.table_name)
    stable_output_path = Path(args.output) if args.output else _stable_prosody_output_path(
        config,
        args.split,
        feature_version=feature_version,
    )
    write_jsonl(stable_output_path, rows)
    quality_report_path = (
        Path(args.quality_report)
        if args.quality_report
        else _stable_prosody_quality_report_path(
            config,
            args.split,
            feature_version=feature_version,
        )
    )
    quality_report = build_prosody_quality_report(rows, extractor=extractor)
    quality_report["cache_key"] = metadata_for_row(store)
    quality_report["stable_prosody_path"] = stable_output_path.as_posix()
    quality_report["nested_prosody_path"] = nested_output_path.as_posix()
    write_json(quality_report_path, quality_report)
    print(
        {
            "output": stable_output_path.as_posix(),
            "nested_output": nested_output_path.as_posix(),
            "quality_report": quality_report_path.as_posix(),
            "num_rows": len(rows),
        }
    )


if __name__ == "__main__":
    main()
