"""Split-safe feature cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from humomni.utils.hashing import json_sha256, manifest_hash
from humomni.utils.io import read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class FeatureCacheKey:
    """A cache key that includes all split-safety fields."""

    feature_version: str
    split: str
    extractor_name: str
    extractor_config_hash: str
    input_manifest_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "feature_version": self.feature_version,
            "split": self.split,
            "extractor_name": self.extractor_name,
            "extractor_config_hash": self.extractor_config_hash,
            "input_manifest_hash": self.input_manifest_hash,
        }

    def as_cache_id(self) -> str:
        return "::".join(
            [
                self.feature_version,
                self.split,
                self.extractor_name,
                self.extractor_config_hash,
                self.input_manifest_hash,
            ]
        )


@dataclass(frozen=True)
class FeatureStore:
    """Path resolver and writer for cached feature artifacts."""

    cache_root: Path
    cache_key: FeatureCacheKey

    def cache_dir(self) -> Path:
        return (
            self.cache_root
            / self.cache_key.feature_version
            / self.cache_key.split
            / self.cache_key.extractor_name
            / self.cache_key.extractor_config_hash
            / self.cache_key.input_manifest_hash
        )

    def table_path(self, table_name: str = "features") -> Path:
        return self.cache_dir() / f"{table_name}.jsonl"

    def metadata_path(self, table_name: str = "features") -> Path:
        return self.cache_dir() / f"{table_name}.meta.json"

    def write_table(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        table_name: str = "features",
    ) -> Path:
        output_path = self.table_path(table_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, [dict(row) for row in rows])
        write_json(
            self.metadata_path(table_name),
            {
                "cache_key": self.cache_key.to_dict(),
                "cache_id": self.cache_key.as_cache_id(),
                "table_name": table_name,
                "num_rows": len(rows),
                "format": "jsonl",
            },
        )
        return output_path

    def read_table(self, *, table_name: str = "features") -> list[dict[str, Any]]:
        return read_jsonl(self.table_path(table_name))


def extractor_config_hash(extractor_config: Mapping[str, Any] | None) -> str:
    """Hash an extractor config using canonical JSON."""

    return json_sha256(dict(extractor_config or {}))


def build_cache_key(
    *,
    feature_version: str,
    split: str,
    extractor_name: str,
    extractor_config: Mapping[str, Any] | None,
    input_manifest_hash: str,
) -> FeatureCacheKey:
    """Build the required split-safe cache key."""

    if not feature_version:
        raise ValueError("feature_version is required for feature cache keys")
    if not split:
        raise ValueError("split is required for feature cache keys")
    if not extractor_name:
        raise ValueError("extractor_name is required for feature cache keys")
    if not input_manifest_hash:
        raise ValueError("input_manifest_hash is required for feature cache keys")
    return FeatureCacheKey(
        feature_version=feature_version,
        split=split,
        extractor_name=extractor_name,
        extractor_config_hash=extractor_config_hash(extractor_config),
        input_manifest_hash=input_manifest_hash,
    )


def build_feature_store(
    *,
    cache_root: str | Path,
    feature_version: str,
    split: str,
    extractor_name: str,
    extractor_config: Mapping[str, Any] | None,
    manifest_path: str | Path,
) -> FeatureStore:
    """Create a FeatureStore for a manifest-backed extraction run."""

    key = build_cache_key(
        feature_version=feature_version,
        split=split,
        extractor_name=extractor_name,
        extractor_config=extractor_config,
        input_manifest_hash=manifest_hash(manifest_path),
    )
    return FeatureStore(cache_root=Path(cache_root), cache_key=key)


def feature_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    features = config.get("features", {})
    if not isinstance(features, Mapping):
        raise ValueError("model config field 'features' must be a mapping")
    return features


def feature_version_from_config(config: Mapping[str, Any]) -> str:
    return str(feature_settings(config).get("feature_version", "feat-v0"))


def cache_root_from_config(config: Mapping[str, Any]) -> Path:
    return Path(str(feature_settings(config).get("cache_root", "artifacts/features")))


def extractor_settings(
    config: Mapping[str, Any],
    extractor_group: str,
) -> tuple[str, dict[str, Any]]:
    settings = feature_settings(config).get(extractor_group, {})
    if not isinstance(settings, Mapping):
        raise ValueError(f"features.{extractor_group} must be a mapping")
    extractor_name = str(settings.get("extractor_name", extractor_group))
    extractor_config = settings.get("extractor_config", {})
    if not isinstance(extractor_config, Mapping):
        raise ValueError(f"features.{extractor_group}.extractor_config must be a mapping")
    return extractor_name, dict(extractor_config)


def metadata_for_row(store: FeatureStore) -> dict[str, str]:
    """Return cache metadata fields to embed in feature rows."""

    return store.cache_key.to_dict()
