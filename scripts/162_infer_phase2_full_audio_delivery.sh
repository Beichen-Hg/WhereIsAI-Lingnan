#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
MODEL_DIR="${MODEL_DIR:-artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1}"
MANIFEST="${MANIFEST:-artifacts/manifests/phase2_test_full_provided_text.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/features/feat-phase2-full-provided-text/phase2_test}"
FEATURE_TABLE="${FEATURE_TABLE:-${FEATURE_ROOT}/feature_table.jsonl}"
PROSODY="${PROSODY:-${FEATURE_ROOT}/prosody.jsonl}"
EMOTION2VEC="${EMOTION2VEC:-artifacts/features/feat-e2v-plus-large/phase2_full_test/emotion2vec.jsonl}"
WAVLM="${WAVLM:-artifacts/features/feat-wavlm-chunked/phase2_full_test/wavlm.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/submissions/phase2_test_full_audio_delivery/audio_delivery_all_labeled_clean_no_quality_v1}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

"${PYTHON_BIN}" -m humomni.infer.phase2_full_audio_delivery_system \
  --model-dir "${MODEL_DIR}" \
  --manifest "${MANIFEST}" \
  --feature-table "${FEATURE_TABLE}" \
  --prosody "${PROSODY}" \
  --emotion2vec "${EMOTION2VEC}" \
  --wavlm "${WAVLM}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
