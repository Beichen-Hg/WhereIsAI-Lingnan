#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
FINAL_ID="humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1"
MODEL_DIR="${MODEL_DIR:-artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1}"
MANIFEST="${MANIFEST:-artifacts/manifests/phase1_test_full_provided_text.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/features/feat-phase1-full-provided-text/phase1_test}"
EMOTION2VEC="${EMOTION2VEC:-artifacts/features/feat-e2v-plus-large/phase1_full_test/emotion2vec.jsonl}"
WAVLM="${WAVLM:-artifacts/features/feat-wavlm-chunked/phase1_full_test/wavlm.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reproduced/${FINAL_ID}}"
FROZEN_SUBMISSION="${FROZEN_SUBMISSION:-artifacts/final_candidates/${FINAL_ID}/submission.jsonl}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export OMP_NUM_THREADS

"${PYTHON_BIN}" -m humomni.infer.phase1_full_audio_delivery_system \
  --model-dir "${MODEL_DIR}" \
  --manifest "${MANIFEST}" \
  --feature-table "${FEATURE_ROOT}/feature_table.jsonl" \
  --prosody "${FEATURE_ROOT}/prosody.jsonl" \
  --emotion2vec "${EMOTION2VEC}" \
  --wavlm "${WAVLM}" \
  --output-dir "${OUTPUT_DIR}" \
  --audit-output "${OUTPUT_DIR}/dependency_audit_report.json" \
  --device "${DEVICE}"

cmp --silent "${OUTPUT_DIR}/submission.jsonl" "${FROZEN_SUBMISSION}"
printf 'Reproduced submission matches frozen final candidate: %s\n' "${FROZEN_SUBMISSION}"
