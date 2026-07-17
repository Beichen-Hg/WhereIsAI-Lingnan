"""Build the blind Phase 2 provided-text manifest for EmpathyEval Track 1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from humomni.data.build_phase1_full_test_manifest import (
    Phase1ReleaseSpec,
    build_phase1_full_provided_text_manifest,
)

PHASE2_SPECS = (
    Phase1ReleaseSpec(
        source_id="gigaspeech",
        task_type="context_variant",
        release_json=Path(
            "data/raw/empathyeval/phase2-test_multi-context_gigaspeech/"
            "phase2-test_gigaspeech_release.json"
        ),
        data_root=Path("data/raw/empathyeval/phase2-test_multi-context_gigaspeech"),
        expected_rows=226,
        expected_groups=113,
        expected_option_count=2,
        question_id_field="data_index",
        split="phase2_test",
    ),
    Phase1ReleaseSpec(
        source_id="meld",
        task_type="context_variant",
        release_json=Path(
            "data/raw/empathyeval/phase2-test_multi-context_meld/"
            "phase2-test_meld_release.json"
        ),
        data_root=Path("data/raw/empathyeval/phase2-test_multi-context_meld"),
        expected_rows=168,
        expected_groups=56,
        expected_option_count=3,
        question_id_field="data_index",
        split="phase2_test",
    ),
    Phase1ReleaseSpec(
        source_id="emovdb",
        task_type="tone_variant",
        release_json=Path(
            "data/raw/empathyeval/phase2-test_multi-emotion_emovdb/"
            "phase2-test_emovdb_release.json"
        ),
        data_root=Path("data/raw/empathyeval/phase2-test_multi-emotion_emovdb"),
        expected_rows=148,
        expected_groups=37,
        expected_option_count=2,
        split="phase2_test",
    ),
)


def build_phase2_full_provided_text_manifest(
    *,
    output_manifest: str | Path = "artifacts/manifests/phase2_test_full_provided_text.jsonl",
    output_report: str | Path = "artifacts/manifests/phase2_test_full_provided_text_report.json",
    specs: Sequence[Phase1ReleaseSpec] = PHASE2_SPECS,
) -> dict[str, Any]:
    """Build and audit the Phase 2 manifest without reading answer labels."""

    return build_phase1_full_provided_text_manifest(
        output_manifest=output_manifest,
        output_report=output_report,
        specs=specs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-manifest",
        default="artifacts/manifests/phase2_test_full_provided_text.jsonl",
    )
    parser.add_argument(
        "--output-report",
        default="artifacts/manifests/phase2_test_full_provided_text_report.json",
    )
    args = parser.parse_args()
    report = build_phase2_full_provided_text_manifest(
        output_manifest=args.output_manifest,
        output_report=args.output_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
