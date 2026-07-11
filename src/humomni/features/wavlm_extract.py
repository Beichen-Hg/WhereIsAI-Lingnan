"""Frozen WavLM embedding extraction for audio-delivery experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from humomni.utils.hashing import json_sha256
from humomni.utils.io import read_jsonl, write_json, write_jsonl


STRICT_INFER_KEYS = frozenset({"label", "gold", "answer", "is_gold_candidate"})
LEAKAGE_VALUE_TERMS = ("goodpara", "badpara")


class WavLMLikeExtractor(Protocol):
    extractor_name: str
    extractor_config: Mapping[str, Any]

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        ...


class FrozenWavLMExtractor:
    extractor_name = "wavlm_frozen"

    def __init__(
        self,
        *,
        model_id: str = "microsoft/wavlm-base-plus",
        revision: str = "main",
        cache_dir: str = "artifacts/model_cache/wavlm",
        local_snapshot_path: str | None = None,
        local_files_only: bool = True,
        allow_download: bool = False,
        device: str = "auto",
        dtype: str = "auto",
        max_chunk_seconds: float = 15.0,
        trust_remote_code: bool = False,
        pooling_strategy: str = "last_mean",
        hidden_layers: Sequence[int] | str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.allow_download = bool(allow_download)
        self.local_snapshot_path = local_snapshot_path or resolve_hf_snapshot(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only or not allow_download,
            allow_download=allow_download,
        )
        self.device_spec = device
        self.dtype_spec = dtype
        self.max_chunk_seconds = float(max_chunk_seconds)
        self.trust_remote_code = bool(trust_remote_code)
        self.pooling_strategy = str(pooling_strategy)
        self.hidden_layers = _parse_hidden_layers(hidden_layers)
        if self.pooling_strategy not in {"last_mean", "hidden_mean", "hidden_concat", "hidden_mean_std"}:
            raise ValueError("WavLM pooling_strategy must be last_mean, hidden_mean, hidden_concat, or hidden_mean_std")
        self.extractor_config = {
            "model_id": model_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_snapshot_path": self.local_snapshot_path,
            "local_files_only": self.local_files_only,
            "allow_download": self.allow_download,
            "device": device,
            "dtype": dtype,
            "max_chunk_seconds": self.max_chunk_seconds,
            "trust_remote_code": self.trust_remote_code,
            "pooling_strategy": self.pooling_strategy,
            "hidden_layers": self.hidden_layers,
        }
        self._torch = None
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._dtype = "float32"
        self._sampling_rate = 16000
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
        except Exception as exc:  # pragma: no cover
            raise ImportError("WavLM extraction requires torch and transformers to be installed.") from exc

        self._torch = torch
        self._device = _resolve_device(self.device_spec, torch)
        self._dtype = _resolve_dtype_name(self.dtype_spec, self._device)
        torch_dtype = _torch_dtype(self._dtype, torch)
        loader_kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        try:
            self._processor = AutoFeatureExtractor.from_pretrained(self.local_snapshot_path, **loader_kwargs)
        except Exception:
            self._processor = AutoProcessor.from_pretrained(self.local_snapshot_path, **loader_kwargs)
        self._sampling_rate = int(getattr(self._processor, "sampling_rate", 16000) or 16000)
        self._model = AutoModel.from_pretrained(
            self.local_snapshot_path,
            torch_dtype=torch_dtype,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_audio_paths = _candidate_audio_paths(manifest_row)
        user_audio_path = manifest_row.get("utterance_audio_path")
        if not isinstance(user_audio_path, str):
            raise ValueError("manifest row must include utterance_audio_path")
        user_embedding, user_meta = self.extract_file(user_audio_path)
        candidate_embeddings: dict[str, list[float]] = {}
        candidate_meta: dict[str, dict[str, Any]] = {}
        for candidate_id, audio_path in sorted(candidate_audio_paths.items()):
            embedding, meta = self.extract_file(str(audio_path))
            candidate_embeddings[str(candidate_id)] = embedding
            candidate_meta[str(candidate_id)] = meta
        return _wavlm_row(
            manifest_row=manifest_row,
            user_audio_path=user_audio_path,
            candidate_audio_paths=candidate_audio_paths,
            user_embedding=user_embedding,
            candidate_embeddings=candidate_embeddings,
            meta={
                "model_id": self.model_id,
                "revision": self.revision,
                "local_snapshot_path": self.local_snapshot_path,
                "device": self._device,
                "dtype": self._dtype,
                "embedding_dim": len(user_embedding),
                "pooling_strategy": self.pooling_strategy,
                "hidden_layers": self.hidden_layers,
                "empty_or_failed": bool(user_meta.get("empty_or_failed", False)),
                "user_meta": user_meta,
                "candidate_meta": candidate_meta,
            },
            extractor_name=self.extractor_name,
        )

    def extract_file(self, audio_path: str) -> tuple[list[float], dict[str, Any]]:
        torch = self._torch
        assert torch is not None
        try:
            signal, sample_rate = _read_audio(audio_path)
            duration = len(signal) / float(sample_rate) if sample_rate else 0.0
            if sample_rate != self._sampling_rate:
                signal = _resample(signal, sample_rate, self._sampling_rate)
                sample_rate = self._sampling_rate
            vector = self._extract_signal_embedding(signal, sample_rate)
            embedding = [float(value) for value in vector.tolist()]
            return embedding, {"duration": duration, "embedding_dim": len(embedding), "empty_or_failed": False}
        except Exception as exc:
            return [], {
                "duration": 0.0,
                "embedding_dim": 0,
                "empty_or_failed": True,
                "error": type(exc).__name__,
                "error_message": str(exc)[:240],
            }

    def _extract_signal_embedding(self, signal: Any, sample_rate: int) -> Any:
        import numpy as np

        max_samples = int(max(1.0, self.max_chunk_seconds) * sample_rate)
        if self.max_chunk_seconds <= 0 or len(signal) <= max_samples:
            return self._forward_signal(signal, sample_rate)

        pooled_vectors = []
        weights = []
        for start in range(0, len(signal), max_samples):
            chunk = signal[start : start + max_samples]
            if len(chunk) == 0:
                continue
            pooled_vectors.append(self._forward_signal(chunk, sample_rate))
            weights.append(float(len(chunk)))
        if not pooled_vectors:
            return self._forward_signal(signal, sample_rate)
        stacked = np.stack(pooled_vectors, axis=0)
        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / max(float(weight_array.sum()), 1.0)
        return (stacked * weight_array[:, None]).sum(axis=0)

    def _forward_signal(self, signal: Any, sample_rate: int) -> Any:
        torch = self._torch
        assert torch is not None
        inputs = self._processor(signal, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        torch_dtype = _torch_dtype(self._dtype, torch)
        inputs = {
            key: (
                value.to(self._device, dtype=torch_dtype)
                if torch.is_floating_point(value)
                else value.to(self._device)
            )
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        with torch.inference_mode():
            output = self._model(
                **inputs,
                output_hidden_states=self.pooling_strategy != "last_mean",
            )
        pooled = _pool_model_output(
            output,
            pooling_strategy=self.pooling_strategy,
            hidden_layers=self.hidden_layers,
        )
        return pooled.detach().float().cpu().numpy().reshape(-1)


class DummyWavLMExtractor:
    extractor_name = "dummy_wavlm"

    def __init__(self, *, embedding_dim: int = 8) -> None:
        self.embedding_dim = int(embedding_dim)
        self.extractor_config = {"backend": "dummy", "embedding_dim": self.embedding_dim}

    def extract_row(self, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_audio_paths = _candidate_audio_paths(manifest_row)
        user_audio_path = str(manifest_row.get("utterance_audio_path", manifest_row["question_id"]))
        return _wavlm_row(
            manifest_row=manifest_row,
            user_audio_path=user_audio_path,
            candidate_audio_paths=candidate_audio_paths,
            user_embedding=_hash_embedding(user_audio_path, self.embedding_dim),
            candidate_embeddings={
                str(candidate_id): _hash_embedding(str(audio_path), self.embedding_dim)
                for candidate_id, audio_path in sorted(candidate_audio_paths.items())
            },
            meta={
                "model_id": "dummy",
                "local_snapshot_path": "dummy://local",
                "embedding_dim": self.embedding_dim,
                "empty_or_failed": False,
            },
            extractor_name=self.extractor_name,
        )


def prepare_wavlm_cache(
    *,
    mode: str,
    model_id: str,
    revision: str,
    cache_dir: str,
    local_files_only: bool = False,
    allow_download: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    pooling_strategy: str = "last_mean",
    hidden_layers: Sequence[int] | str | None = None,
    manifest_path: str | Path = "artifacts/model_cache/wavlm/manifest.json",
) -> dict[str, Any]:
    normalized_mode = mode.lower()
    if normalized_mode not in {"download", "verify"}:
        raise ValueError("WavLM cache mode must be download or verify")
    if normalized_mode == "verify":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        local_files_only = True
        allow_download = False
    snapshot = resolve_hf_snapshot(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only or not allow_download,
        allow_download=allow_download,
    )
    verified = False
    verify_error = None
    try:
        FrozenWavLMExtractor(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_snapshot_path=snapshot,
            local_files_only=True,
            allow_download=False,
            device=device,
            dtype=dtype,
            pooling_strategy=pooling_strategy,
            hidden_layers=hidden_layers,
        )
        verified = True
    except Exception as exc:
        verify_error = f"{type(exc).__name__}: {exc}"
        if normalized_mode == "verify":
            raise RuntimeError(f"offline WavLM verification failed for {model_id}: {verify_error}") from exc
    record = {
        "model_id": model_id,
        "revision": revision,
        "local_snapshot_path": snapshot,
        "config_hash": json_sha256(
            {
                "model_id": model_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "pooling_strategy": pooling_strategy,
                "hidden_layers": _parse_hidden_layers(hidden_layers),
            }
        ),
        "operation": normalized_mode,
        "offline_verified": verified,
        "verify_error": verify_error,
    }
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
        raise ImportError("huggingface_hub is required for WavLM cache preparation") from exc
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
                f"Local WavLM cache is missing or incomplete for {model_id}@{revision} under {cache_dir}. "
                "Run MODE=download first."
            ) from exc
        raise


def extract_wavlm_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    extractor: WavLMLikeExtractor,
    *,
    mode: str,
    feature_version: str = "feat-wavlm-chunked",
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    if mode.lower() in {"infer", "test", "phase1_test"}:
        assert_no_infer_leakage(manifest_rows)
    rows = []
    total_rows = len(manifest_rows)
    for row_index, manifest_row in enumerate(manifest_rows, start=1):
        row = extractor.extract_row(manifest_row)
        row["feature_version"] = feature_version
        row["extractor_name"] = extractor.extractor_name
        rows.append(row)
        if progress_every > 0 and (row_index % progress_every == 0 or row_index == total_rows):
            print(f"[wavlm_extract] {mode}: {row_index}/{total_rows}", file=sys.stderr, flush=True)
    if mode.lower() in {"infer", "test", "phase1_test"}:
        assert_no_infer_leakage(rows)
    return rows


def validate_wavlm_rows(rows: Sequence[Mapping[str, Any]], *, mode: str = "train") -> None:
    if mode.lower() in {"infer", "test", "phase1_test"}:
        assert_no_infer_leakage(rows)
    for index, row in enumerate(rows):
        for field in (
            "question_id",
            "group_id",
            "split",
            "user_audio_hash",
            "candidate_audio_hashes",
            "user_wavlm_embedding",
            "candidate_wavlm_embeddings",
            "wavlm_meta",
        ):
            if field not in row:
                raise ValueError(f"WavLM row {index}: missing {field}")
        if not isinstance(row["candidate_wavlm_embeddings"], Mapping):
            raise ValueError(f"WavLM row {index}: candidate_wavlm_embeddings must be a dict")


def build_wavlm_quality_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dims = []
    norms = []
    failed = 0
    total = 0
    for row in rows:
        vectors = [row.get("user_wavlm_embedding", [])]
        candidates = row.get("candidate_wavlm_embeddings", {})
        if isinstance(candidates, Mapping):
            vectors.extend(candidates.values())
        for vector in vectors:
            values = _float_list(vector)
            dims.append(len(values))
            norms.append(_l2(values))
            failed += int(not values)
            total += 1
    return {
        "num_rows": len(rows),
        "total_audio_items": total,
        "embedding_dim": max(dims) if dims else 0,
        "empty_embedding_count": sum(1 for dim in dims if dim == 0),
        "failed_rate": failed / total if total else 0.0,
        "norm_distribution": _stats(norms),
        "backend_distribution": _counter(str(row.get("extractor_name", "unknown")) for row in rows),
    }


def assert_no_infer_leakage(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        hits = _leakage_hits(row)
        if hits:
            raise ValueError(f"infer/test WavLM row {index} contains leakage: {hits[:5]}")


def _wavlm_row(
    *,
    manifest_row: Mapping[str, Any],
    user_audio_path: str,
    candidate_audio_paths: Mapping[str, Any],
    user_embedding: Sequence[float],
    candidate_embeddings: Mapping[str, Sequence[float]],
    meta: Mapping[str, Any],
    extractor_name: str,
) -> dict[str, Any]:
    return {
        "question_id": manifest_row["question_id"],
        "group_id": manifest_row.get("group_id"),
        "split": manifest_row.get("split"),
        "user_audio_hash": _safe_audio_id(user_audio_path),
        "candidate_audio_hashes": {
            str(candidate_id): _safe_audio_id(str(audio_path))
            for candidate_id, audio_path in sorted(candidate_audio_paths.items())
        },
        "user_wavlm_embedding": [float(value) for value in user_embedding],
        "candidate_wavlm_embeddings": {
            str(candidate_id): [float(value) for value in embedding]
            for candidate_id, embedding in candidate_embeddings.items()
        },
        "wavlm_meta": dict(meta),
        "extractor_name": extractor_name,
    }


def _build_extractor(args: argparse.Namespace) -> WavLMLikeExtractor:
    if args.backend == "dummy":
        return DummyWavLMExtractor(embedding_dim=args.dummy_dim)
    if args.backend != "wavlm":
        raise ValueError("WavLM backend must be wavlm or dummy")
    return FrozenWavLMExtractor(
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        allow_download=args.allow_download,
        device=args.device,
        dtype=args.dtype,
        max_chunk_seconds=args.max_chunk_seconds,
        pooling_strategy=args.pooling_strategy,
        hidden_layers=args.hidden_layers,
    )


def _candidate_audio_paths(row: Mapping[str, Any]) -> Mapping[str, Any]:
    paths = row.get("candidate_audio_paths", {})
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError("manifest row must include candidate_audio_paths")
    return paths


def _read_audio(audio_path: str) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    signal, sample_rate = sf.read(audio_path, always_2d=False)
    y = np.asarray(signal, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(sample_rate)


def _resample(signal: Any, source_rate: int, target_rate: int) -> Any:
    import librosa

    return librosa.resample(signal, orig_sr=source_rate, target_sr=target_rate)


def _pool_model_output(
    output: Any,
    *,
    pooling_strategy: str = "last_mean",
    hidden_layers: Sequence[int] | None = None,
) -> Any:
    if pooling_strategy != "last_mean":
        if not hasattr(output, "hidden_states") or output.hidden_states is None:
            raise RuntimeError("WavLM output does not contain hidden_states for multi-layer pooling")
        states = list(output.hidden_states)
        layer_indices = list(hidden_layers or [-1, -2, -3, -4])
        selected = []
        for layer_index in layer_indices:
            selected.append(states[layer_index].mean(dim=1))
        if pooling_strategy == "hidden_mean":
            return sum(selected) / float(len(selected))
        if pooling_strategy == "hidden_concat":
            import torch

            return torch.cat(selected, dim=-1)
        if pooling_strategy == "hidden_mean_std":
            import torch

            stacked = torch.stack(selected, dim=0)
            return torch.cat([stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)], dim=-1)
        raise RuntimeError(f"unsupported WavLM pooling_strategy={pooling_strategy}")
    tensor = None
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        tensor = output.last_hidden_state.mean(dim=1)
    elif hasattr(output, "extract_features") and output.extract_features is not None:
        tensor = output.extract_features.mean(dim=1)
    elif isinstance(output, tuple) and output:
        tensor = output[0]
        if getattr(tensor, "ndim", 0) == 3:
            tensor = tensor.mean(dim=1)
    if tensor is None:
        raise RuntimeError("WavLM output does not contain a usable tensor")
    return tensor


def _parse_hidden_layers(value: Sequence[int] | str | None) -> list[int]:
    if value is None or value == "":
        return [-1, -2, -3, -4]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


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


def _safe_audio_id(path: str) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


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


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list | tuple):
        return []
    return [_float(value) for value in values]


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


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
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--feature-version", default="feat-wavlm-chunked")
    parser.add_argument("--backend", default="wavlm", choices=["wavlm", "dummy"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--model-id", default="microsoft/wavlm-base-plus")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default="artifacts/model_cache/wavlm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-chunk-seconds", type=float, default=15.0)
    parser.add_argument("--pooling-strategy", default="last_mean", choices=["last_mean", "hidden_mean", "hidden_concat", "hidden_mean_std"])
    parser.add_argument("--hidden-layers", default="-1,-2,-3,-4")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-manifest", default="artifacts/model_cache/wavlm/manifest.json")
    parser.add_argument("--dummy-dim", type=int, default=8)
    args = parser.parse_args()

    if args.mode in {"download", "verify"}:
        record = prepare_wavlm_cache(
            mode=args.mode,
            model_id=args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            allow_download=args.allow_download,
            device=args.device,
            dtype=args.dtype,
            pooling_strategy=args.pooling_strategy,
            hidden_layers=args.hidden_layers,
            manifest_path=args.cache_manifest,
        )
        print(json.dumps(record, indent=2))
        return
    if args.manifest is None:
        raise ValueError("--manifest is required in extract mode")
    extractor = _build_extractor(args)
    rows = extract_wavlm_rows(
        read_jsonl(args.manifest),
        extractor,
        mode=args.split,
        feature_version=args.feature_version,
        progress_every=args.progress_every,
    )
    validate_wavlm_rows(rows, mode=args.split)
    output_path = Path(args.output or f"artifacts/features/{args.feature_version}/{args.split}/wavlm.jsonl")
    report_path = Path(args.quality_report or output_path.with_name("wavlm_quality_report.json"))
    write_jsonl(output_path, rows)
    report = build_wavlm_quality_report(rows)
    report["extractor_config"] = dict(extractor.extractor_config)
    write_json(report_path, report)
    print(json.dumps({"output": output_path.as_posix(), "quality_report": report_path.as_posix(), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
