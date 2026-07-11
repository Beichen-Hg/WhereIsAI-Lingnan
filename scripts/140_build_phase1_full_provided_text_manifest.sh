#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-artifacts/manifests/phase1_test_full_provided_text.jsonl}"
OUTPUT_REPORT="${OUTPUT_REPORT:-artifacts/manifests/phase1_test_full_provided_text_report.json}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

"${PYTHON_BIN}" -m humomni.data.build_phase1_full_test_manifest \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --output-report "${OUTPUT_REPORT}" \
  "$@"
