"""Audio SSL and emotion2vec embedding extraction interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
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
from humomni.utils.hashing import json_sha256
from humomni.utils.io import read_jsonl, read_yaml, write_json, write_jsonl

STRICT_INFER_KEYS = frozenset({"label", "gold", "answer", "is_gold_candidate"})
LEAKAGE_VALUE_TERMS = ("goodpara", "badpara")


class AudioEmbeddingExtractor(Protocol):
    extractor_name: str
    extractor_config: Mapping[str, Any]

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        """Extract audio embedding fields for a manifest row."""


class DummyAudioEmbeddingExtractor:
    """Deterministic fixed-size embedding stub."""

    extractor_name = "dummy_audio_embedding"

    def __init__(self, *, embedding_dim: int = 8) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.extractor_config = {"embedding_dim": embedding_dim}

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_audio_paths = _candidate_audio_paths(manifest_row)
        user_seed = str(manifest_row.get("utterance_audio_path", manifest_row["question_id"]))
        user_embedding = _hash_embedding(user_seed, self.embedding_dim)
        candidate_embeddings = {
            candidate_id: _hash_embedding(str(audio_path), self.embedding_dim)
            for candidate_id, audio_path in sorted(candidate_audio_paths.items())
        }
        return {
            "question_id": manifest_row["question_id"],
            "group_id": manifest_row.get("group_id"),
            "split": manifest_row.get("split"),
            "user_audio_hash": _safe_audio_id(user_seed),
            "candidate_audio_hashes": {
                candidate_id: _safe_audio_id(str(audio_path))
                for candidate_id, audio_path in sorted(candidate_audio_paths.items())
            },
            "user_audio_embedding": user_embedding,
            "candidate_audio_embeddings": candidate_embeddings,
            "user_emotion_embedding": user_embedding,
            "candidate_emotion_embeddings": candidate_embeddings,
            "embedding_meta": {
                "model_id": "dummy",
                "local_snapshot_path": "dummy://local",
                "device": "none",
                "dtype": "float32",
                "embedding_dim": self.embedding_dim,
                "duration": 0.0,
                "empty_or_failed": False,
            },
        }


class Emotion2VecExtractor:
    """Frozen emotion2vec embedding extractor using a local HuggingFace snapshot."""

    extractor_name = "emotion2vec"

    def __init__(
        self,
        *,
        model_id: str = "emotion2vec/emotion2vec_plus_large",
        revision: str = "main",
        cache_dir: str = "artifacts/model_cache/audio_ssl",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        local_files_only: bool = True,
        allow_download: bool = False,
        local_snapshot_path: str | None = None,
        trust_remote_code: bool = True,
        backend: str = "auto",
        **extra_config: Any,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.device_spec = device
        self.dtype_spec = dtype
        self.batch_size = int(batch_size)
        self.local_files_only = bool(local_files_only)
        self.allow_download = bool(allow_download)
        self.trust_remote_code = bool(trust_remote_code)
        self.backend = backend
        self.local_snapshot_path = local_snapshot_path or resolve_hf_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only or not allow_download,
            allow_download=allow_download,
        )
        self.extractor_config = {
            "model_id": model_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "device": device,
            "dtype": dtype,
            "batch_size": self.batch_size,
            "local_files_only": self.local_files_only,
            "allow_download": self.allow_download,
            "local_snapshot_path": self.local_snapshot_path,
            "trust_remote_code": self.trust_remote_code,
            "backend": self.backend,
            "ignored_extra_config_keys": sorted(extra_config),
        }
        self._torch = None
        self._processor = None
        self._model = None
        self._funasr_model = None
        self._device = "cpu"
        self._dtype = "float32"
        self._sampling_rate = 16000
        self._load_model()

    def _load_model(self) -> None:
        if self.backend not in {"auto", "transformers", "funasr"}:
            raise ValueError("emotion2vec backend must be auto, transformers, or funasr")
        if self.backend in {"auto", "funasr"} and _snapshot_uses_funasr(self.local_snapshot_path):
            self._load_funasr_model()
            return
        self._load_transformers_model()

    def _load_funasr_model(self) -> None:
        _ensure_python_ffmpeg_on_path()
        try:
            from funasr import AutoModel
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "The local emotion2vec snapshot is a FunASR checkpoint, not a standard "
                "Transformers AutoModel. Install the optional Python packages "
                "`funasr` and `modelscope` in this environment, then rerun offline verify. "
                "No package installation is performed automatically."
            ) from exc
        try:
            import torch

            self._device = _resolve_device(self.device_spec, torch)
            self._dtype = _resolve_dtype_name(self.dtype_spec, self._device)
        except Exception:
            self._device = "cpu" if self.device_spec == "auto" else str(self.device_spec)
            self._dtype = "float32" if self.dtype_spec == "auto" else str(self.dtype_spec)
        try:
            self._funasr_model = AutoModel(
                model=self.local_snapshot_path,
                disable_update=True,
                device=self._device if self._device != "auto" else "cpu",
            )
        except TypeError:
            self._funasr_model = AutoModel(model=self.local_snapshot_path)
        self._sampling_rate = 16000

    def _load_transformers_model(self) -> None:
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
        except Exception as exc:  # pragma: no cover - exercised in integration
            raise ImportError(
                "emotion2vec extraction requires torch and transformers to be installed."
            ) from exc

        self._torch = torch
        self._device = _resolve_device(self.device_spec, torch)
        self._dtype = _resolve_dtype_name(self.dtype_spec, self._device)
        torch_dtype = _torch_dtype(self._dtype, torch)
        loader_kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        processor_error: Exception | None = None
        try:
            self._processor = AutoFeatureExtractor.from_pretrained(
                self.local_snapshot_path,
                **loader_kwargs,
            )
        except Exception as exc:
            processor_error = exc
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.local_snapshot_path,
                    **loader_kwargs,
                )
            except Exception as processor_exc:
                if _snapshot_uses_funasr(self.local_snapshot_path):
                    raise ImportError(
                        "The local emotion2vec snapshot is a FunASR checkpoint and cannot "
                        "be loaded by Transformers AutoProcessor/AutoModel. Use backend=funasr "
                        "and install optional packages `funasr` and `modelscope`."
                    ) from processor_exc
                raise RuntimeError(
                    "Failed to load emotion2vec processor/feature extractor from local cache "
                    f"{self.local_snapshot_path}. Original errors: {processor_error}; {processor_exc}"
                ) from processor_exc
        self._sampling_rate = int(
            getattr(self._processor, "sampling_rate", None)
            or getattr(self._processor, "feature_extractor", self._processor).__dict__.get(
                "sampling_rate", 16000
            )
            or 16000
        )
        try:
            self._model = AutoModel.from_pretrained(
                self.local_snapshot_path,
                torch_dtype=torch_dtype,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load emotion2vec model from local cache "
                f"{self.local_snapshot_path}. The snapshot may not be transformers-compatible."
            ) from exc
        self._model.to(self._device)
        self._model.eval()

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_audio_paths = _candidate_audio_paths(manifest_row)
        utterance_audio_path = manifest_row.get("utterance_audio_path")
        if not isinstance(utterance_audio_path, str):
            raise ValueError("manifest row must include utterance_audio_path")
        user_embedding, user_meta = self.extract_file(utterance_audio_path)
        candidate_embeddings: dict[str, list[float]] = {}
        candidate_meta: dict[str, dict[str, Any]] = {}
        for candidate_id, audio_path in sorted(candidate_audio_paths.items()):
            embedding, meta = self.extract_file(str(audio_path))
            candidate_embeddings[str(candidate_id)] = embedding
            candidate_meta[str(candidate_id)] = meta
        embedding_dim = len(user_embedding)
        return {
            "question_id": manifest_row["question_id"],
            "group_id": manifest_row.get("group_id"),
            "split": manifest_row.get("split"),
            "user_audio_hash": _safe_audio_id(utterance_audio_path),
            "candidate_audio_hashes": {
                str(candidate_id): _safe_audio_id(str(audio_path))
                for candidate_id, audio_path in sorted(candidate_audio_paths.items())
            },
            "user_emotion_embedding": user_embedding,
            "candidate_emotion_embeddings": candidate_embeddings,
            # Backward-compatible aliases used by feature_table and ToneExpert.
            "user_audio_embedding": user_embedding,
            "candidate_audio_embeddings": candidate_embeddings,
            "embedding_meta": {
                "model_id": self.model_id,
                "revision": self.revision,
                "local_snapshot_path": self.local_snapshot_path,
                "device": self._device,
                "dtype": self._dtype,
                "embedding_dim": embedding_dim,
                "duration": user_meta.get("duration", 0.0),
                "empty_or_failed": bool(user_meta.get("empty_or_failed", False)),
                "user_meta": user_meta,
                "candidate_meta": candidate_meta,
            },
        }

    def extract_file(self, audio_path: str) -> tuple[list[float], dict[str, Any]]:
        if self._funasr_model is not None:
            return self._extract_file_funasr(audio_path)
        torch = self._torch
        assert torch is not None
        try:
            signal, sample_rate = _read_audio(audio_path)
            duration = len(signal) / float(sample_rate) if sample_rate else 0.0
            if sample_rate != self._sampling_rate:
                signal = _resample(signal, sample_rate, self._sampling_rate)
                sample_rate = self._sampling_rate
            inputs = self._processor(
                signal,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            with torch.inference_mode():
                output = self._model(**inputs)
            pooled = _pool_model_output(output)
            vector = pooled.detach().float().cpu().numpy().reshape(-1)
            embedding = [float(value) for value in vector.tolist()]
            return embedding, {
                "duration": duration,
                "empty_or_failed": False,
                "embedding_dim": len(embedding),
            }
        except Exception as exc:
            return [], {
                "duration": 0.0,
                "empty_or_failed": True,
                "error": type(exc).__name__,
                "error_message": str(exc)[:240],
                "embedding_dim": 0,
            }

    def _extract_file_funasr(self, audio_path: str) -> tuple[list[float], dict[str, Any]]:
        try:
            signal, sample_rate = _read_audio(audio_path)
            duration = len(signal) / float(sample_rate) if sample_rate else 0.0
            with tempfile.TemporaryDirectory(prefix="emotion2vec_") as output_dir:
                model = self._funasr_model
                assert model is not None
                if hasattr(model, "generate"):
                    result = model.generate(
                        input=audio_path,
                        output_dir=output_dir,
                        granularity="utterance",
                        extract_embedding=True,
                    )
                else:
                    result = model(
                        input=audio_path,
                        output_dir=output_dir,
                        granularity="utterance",
                        extract_embedding=True,
                    )
                embedding = _embedding_from_funasr_result(result)
                if not embedding:
                    embedding = _embedding_from_output_dir(output_dir)
            return embedding, {
                "duration": duration,
                "empty_or_failed": not bool(embedding),
                "embedding_dim": len(embedding),
            }
        except Exception as exc:
            return [], {
                "duration": 0.0,
                "empty_or_failed": True,
                "error": type(exc).__name__,
                "error_message": str(exc)[:240],
                "embedding_dim": 0,
            }


class WavLMExtractor:
    """WavLM interface placeholder with no automatic download."""

    extractor_name = "wavlm"

    def __init__(self, *, model_path: str | None = None, local_files_only: bool = True) -> None:
        self.extractor_config = {
            "model_path": model_path,
            "local_files_only": local_files_only,
        }
        if model_path is None:
            raise RuntimeError(
                "WavLMExtractor requires an explicit local model path; "
                "automatic model download is disabled."
            )

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("WavLM extraction is an interface placeholder.")


def prepare_emotion2vec_cache(
    *,
    mode: str,
    model_id: str,
    revision: str,
    cache_dir: str,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    allow_download: bool = False,
    backend: str = "auto",
    manifest_path: str | Path = "artifacts/model_cache/audio_ssl/emotion2vec_plus_large_manifest.json",
) -> dict[str, Any]:
    """Download or offline-verify an emotion2vec snapshot."""

    normalized_mode = mode.lower()
    if normalized_mode not in {"download", "verify"}:
        raise ValueError("prepare_emotion2vec_cache mode must be download or verify")
    offline = normalized_mode == "verify"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        local_files_only = True
        allow_download = False
    snapshot = resolve_hf_snapshot(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_download=allow_download,
    )
    verified = False
    verify_error = None
    try:
        Emotion2VecExtractor(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            local_files_only=True,
            allow_download=False,
            local_snapshot_path=snapshot,
            backend=backend,
        )
        verified = True
    except Exception as exc:
        verify_error = f"{type(exc).__name__}: {exc}"
        if normalized_mode == "verify":
            raise RuntimeError(
                f"offline emotion2vec verification failed for {model_id}: {verify_error}"
            ) from exc
    record = {
        "model_id": model_id,
        "revision": revision,
        "local_snapshot_path": snapshot,
        "config_hash": json_sha256(
            {
                "model_id": model_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "device": device,
                "dtype": dtype,
                "backend": backend,
            }
        ),
        "operation": normalized_mode,
        "offline_verified": verified,
        "verify_error": verify_error,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, record)
    return record


def resolve_hf_snapshot(
    *,
    model_id: str,
    revision: str,
    cache_dir: str,
    local_files_only: bool,
    allow_download: bool,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise ImportError("huggingface_hub is required for emotion2vec cache prep") from exc
    if not allow_download:
        local_files_only = True
    try:
        return snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if local_files_only:
            raise FileNotFoundError(
                f"Local emotion2vec cache is missing or incomplete for {model_id}@{revision} "
                f"under {cache_dir}. Run MODE=download first."
            ) from exc
        raise


def extract_audio_embedding_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    extractor: AudioEmbeddingExtractor,
    *,
    store_metadata: Mapping[str, str] | None = None,
    mode: str = "train",
) -> list[dict[str, Any]]:
    if mode.lower() in {"infer", "test"}:
        assert_no_infer_leakage(manifest_rows)
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        feature_row = extractor.extract_row(manifest_row)
        if store_metadata:
            feature_row.update(store_metadata)
        feature_row["extractor_name"] = extractor.extractor_name
        rows.append(feature_row)
    if mode.lower() in {"infer", "test"}:
        assert_no_infer_leakage(rows)
    return rows


def validate_emotion_embedding_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str = "train",
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
            "user_emotion_embedding",
            "candidate_emotion_embeddings",
            "embedding_meta",
        ):
            if field not in row:
                raise ValueError(f"emotion embedding row {row_index}: missing {field}")
        if not isinstance(row["candidate_emotion_embeddings"], Mapping):
            raise ValueError(f"emotion embedding row {row_index}: candidate embeddings must be dict")


def build_emotion_quality_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    extractor_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    user_dims = []
    candidate_dims = []
    failed = 0
    total = 0
    for row in rows:
        user_embedding = row.get("user_emotion_embedding", [])
        user_dims.append(len(user_embedding) if isinstance(user_embedding, list) else 0)
        meta = row.get("embedding_meta", {})
        if isinstance(meta, Mapping) and meta.get("empty_or_failed"):
            failed += 1
        total += 1
        candidates = row.get("candidate_emotion_embeddings", {})
        if isinstance(candidates, Mapping):
            for embedding in candidates.values():
                candidate_dims.append(len(embedding) if isinstance(embedding, list) else 0)
                total += 1
                if not embedding:
                    failed += 1
    dims = [*user_dims, *candidate_dims]
    return {
        "num_rows": len(rows),
        "total_user_audios": len(user_dims),
        "total_candidate_audios": len(candidate_dims),
        "embedding_dim": max(dims) if dims else 0,
        "failed_rate": failed / total if total else 0.0,
        "empty_embedding_count": sum(1 for dim in dims if dim == 0),
        "extractor_info": dict(extractor_info or {}),
    }


def assert_no_infer_leakage(rows: Sequence[Mapping[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        hits = _leakage_hits(row)
        if hits:
            raise ValueError(f"infer row {row_index} contains leakage: {hits[:5]}")


def _candidate_audio_paths(manifest_row: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate_audio_paths = manifest_row.get("candidate_audio_paths", {})
    if not isinstance(candidate_audio_paths, Mapping) or not candidate_audio_paths:
        raise ValueError("manifest row must include candidate_audio_paths")
    return candidate_audio_paths


def _hash_embedding(seed: str, dim: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
        for byte in digest:
            values.append(round((byte / 255.0) * 2.0 - 1.0, 6))
            if len(values) == dim:
                break
        counter += 1
    return values


def _safe_audio_id(audio_path: str) -> str:
    return hashlib.sha256(str(audio_path).encode("utf-8")).hexdigest()


def _read_audio(audio_path: str) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    signal, sample_rate = sf.read(audio_path, always_2d=False)
    y = np.asarray(signal, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, int(sample_rate)


def _resample(signal: Any, source_rate: int, target_rate: int) -> Any:
    import librosa

    return librosa.resample(signal, orig_sr=source_rate, target_sr=target_rate)


def _resolve_device(device: str, torch: Any) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_dtype_name(dtype: str, device: str) -> str:
    if dtype == "auto":
        return "float16" if device.startswith("cuda") else "float32"
    return str(dtype)


def _torch_dtype(dtype: str, torch: Any) -> Any:
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _pool_model_output(output: Any) -> Any:
    tensor = None
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        tensor = output.last_hidden_state.mean(dim=1)
    elif hasattr(output, "logits") and output.logits is not None:
        tensor = output.logits
    elif isinstance(output, tuple) and output:
        tensor = output[0]
        if getattr(tensor, "ndim", 0) == 3:
            tensor = tensor.mean(dim=1)
    if tensor is None:
        raise RuntimeError("emotion2vec model output does not contain a usable tensor")
    if getattr(tensor, "ndim", 0) > 2:
        tensor = tensor.mean(dim=1)
    return tensor


def _snapshot_uses_funasr(snapshot_path: str | Path) -> bool:
    configuration_path = Path(snapshot_path) / "configuration.json"
    if not configuration_path.exists():
        return False
    try:
        data = json.loads(configuration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(data, Mapping):
        return False
    serialized = json.dumps(data).lower()
    return "funasr" in serialized or "emotion-recognition" in serialized


def _embedding_from_funasr_result(result: Any) -> list[float]:
    candidates = []
    if isinstance(result, Mapping):
        candidates.append(result)
    elif isinstance(result, list | tuple):
        candidates.extend(item for item in result if isinstance(item, Mapping))
    for item in candidates:
        for key in (
            "feats",
            "features",
            "embedding",
            "embeddings",
            "emotion_embedding",
            "utterance_embedding",
        ):
            if key in item:
                values = _flatten_numeric(item[key])
                if values:
                    return values
    return []


def _embedding_from_output_dir(output_dir: str | Path) -> list[float]:
    try:
        import numpy as np
    except Exception:
        return []
    output_path = Path(output_dir)
    for path in sorted(output_path.rglob("*.npy")):
        try:
            array = np.load(path)
        except Exception:
            continue
        values = _flatten_numeric(array)
        if values:
            return values
    return []


def _ensure_python_ffmpeg_on_path() -> None:
    """Expose imageio-ffmpeg's binary to libraries that shell out to ffmpeg."""

    try:
        import imageio_ffmpeg
    except Exception:
        return
    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return
    if not ffmpeg_path.exists():
        return
    link_dir = Path(tempfile.gettempdir()) / "humomni_ffmpeg_bin"
    link_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / "ffmpeg"
    if not link_path.exists():
        try:
            link_path.symlink_to(ffmpeg_path)
        except OSError:
            import shutil

            shutil.copy2(ffmpeg_path, link_path)
            link_path.chmod(0o755)
    os.environ["PATH"] = f"{link_dir.as_posix()}{os.pathsep}{os.environ.get('PATH', '')}"


def _flatten_numeric(value: Any) -> list[float]:
    try:
        import numpy as np

        array = np.asarray(value, dtype=np.float32).reshape(-1)
        return [float(item) for item in array.tolist()]
    except Exception:
        if isinstance(value, list | tuple):
            output: list[float] = []
            for item in value:
                output.extend(_flatten_numeric(item))
            return output
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []


def _build_extractor(name: str, config: Mapping[str, Any]) -> AudioEmbeddingExtractor:
    normalized = name.lower()
    if normalized in {"dummy", "dummy_audio_embedding"}:
        return DummyAudioEmbeddingExtractor(**dict(config))
    if normalized in {"emotion2vec", "emotion_2_vec"}:
        return Emotion2VecExtractor(**dict(config))
    if normalized in {"wavlm", "wav_lm"}:
        return WavLMExtractor(**dict(config))
    raise ValueError(f"unknown audio embedding extractor: {name}")


def _extractor_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "local_files_only": args.local_files_only,
        "allow_download": args.allow_download,
        "backend": args.backend,
    }


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
    parser.add_argument("--mode", default="extract", choices=["download", "verify", "extract"])
    parser.add_argument("--config", default=None, help="Optional local extractor config path.")
    parser.add_argument("--manifest", default=None, help="Input manifest JSONL path.")
    parser.add_argument("--split", default="train", help="Split name.")
    parser.add_argument("--feature-version", default="feat-e2v-plus-large")
    parser.add_argument("--extractor", default="emotion2vec", help="Override extractor name.")
    parser.add_argument("--table-name", default="emotion2vec", help="Output table name.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--model-id", default="emotion2vec/emotion2vec_plus_large")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default="artifacts/model_cache/audio_ssl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "transformers", "funasr"])
    parser.add_argument(
        "--cache-manifest",
        default="artifacts/model_cache/audio_ssl/emotion2vec_plus_large_manifest.json",
    )
    args = parser.parse_args()

    if args.mode in {"download", "verify"}:
        record = prepare_emotion2vec_cache(
            mode=args.mode,
            model_id=args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            allow_download=args.allow_download,
            backend=args.backend,
            manifest_path=args.cache_manifest,
        )
        print(record)
        return

    if args.manifest is None:
        raise ValueError("--manifest is required in extract mode")
    config = read_yaml(args.config) if args.config and Path(args.config).exists() else {}
    configured_name, configured_config = extractor_settings(config, "audio_ssl")
    extractor_name = args.extractor or configured_name
    extractor_config = dict(configured_config)
    if extractor_name.lower() in {"emotion2vec", "emotion_2_vec"}:
        extractor_config.update(_extractor_config_from_args(args))
    extractor = _build_extractor(extractor_name, extractor_config)
    manifest_rows = read_jsonl(args.manifest)
    if args.split in {"infer", "test"}:
        assert_no_infer_leakage(manifest_rows)
    store = build_feature_store(
        cache_root=cache_root_from_config(config),
        feature_version=args.feature_version or feature_version_from_config(config),
        split=args.split,
        extractor_name=extractor.extractor_name,
        extractor_config=extractor.extractor_config,
        manifest_path=args.manifest,
    )
    rows = extract_audio_embedding_rows(
        manifest_rows,
        extractor,
        store_metadata=metadata_for_row(store),
        mode=args.split,
    )
    validate_emotion_embedding_rows(rows, mode=args.split)
    output_path = (
        Path(args.output)
        if args.output
        else Path("artifacts/features") / args.feature_version / args.split / f"{args.table_name}.jsonl"
    )
    write_jsonl(output_path, rows)
    report = build_emotion_quality_report(rows, extractor_info=getattr(extractor, "extractor_config", {}))
    quality_path = (
        Path(args.quality_report)
        if args.quality_report
        else output_path.with_name(f"{args.table_name}_quality_report.json")
    )
    write_json(quality_path, report)
    print({"output": output_path.as_posix(), "quality_report": quality_path.as_posix(), "num_rows": len(rows)})


if __name__ == "__main__":
    main()
