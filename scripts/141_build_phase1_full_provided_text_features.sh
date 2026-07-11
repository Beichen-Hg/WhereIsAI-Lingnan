#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES_ARG:-0}"
MANIFEST="${MANIFEST:-artifacts/manifests/phase1_test_full_provided_text.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/features/feat-phase1-full-provided-text/phase1_test}"
PROSODY="${PROSODY:-${FEATURE_ROOT}/prosody.jsonl}"
PROSODY_REPORT="${PROSODY_REPORT:-${FEATURE_ROOT}/prosody_quality_report.json}"
EMOTION2VEC="${EMOTION2VEC:-artifacts/features/feat-e2v-plus-large/phase1_full_test/emotion2vec.jsonl}"
EMOTION2VEC_REPORT="${EMOTION2VEC_REPORT:-artifacts/features/feat-e2v-plus-large/phase1_full_test/emotion2vec_quality_report.json}"
WAVLM="${WAVLM:-artifacts/features/feat-wavlm-chunked/phase1_full_test/wavlm.jsonl}"
WAVLM_REPORT="${WAVLM_REPORT:-artifacts/features/feat-wavlm-chunked/phase1_full_test/wavlm_quality_report.json}"
FEATURE_TABLE="${FEATURE_TABLE:-${FEATURE_ROOT}/feature_table.jsonl}"
FEATURE_TABLE_REPORT="${FEATURE_TABLE_REPORT:-${FEATURE_ROOT}/feature_table_report.json}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ARG}"

mkdir -p "${FEATURE_ROOT}" "$(dirname "${EMOTION2VEC}")" "$(dirname "${WAVLM}")" "$(dirname "${FEATURE_TABLE_REPORT}")"

if [[ "${FORCE_FEATURES}" == "1" || ! -f "${PROSODY}" ]]; then
  "${PYTHON_BIN}" -m humomni.features.prosody_extract \
    --manifest "${MANIFEST}" \
    --split phase1_test \
    --mode test \
    --feature-version feat-phase1-full-provided-text \
    --output "${PROSODY}" \
    --quality-report "${PROSODY_REPORT}"
else
  echo "{\"reuse\":\"${PROSODY}\"}"
fi

if [[ "${FORCE_FEATURES}" == "1" || ! -f "${EMOTION2VEC}" ]]; then
  "${PYTHON_BIN}" -m humomni.features.audio_ssl_extract \
    --mode extract \
    --manifest "${MANIFEST}" \
    --split phase1_full_test \
    --feature-version feat-e2v-plus-large \
    --extractor emotion2vec \
    --table-name emotion2vec \
    --output "${EMOTION2VEC}" \
    --quality-report "${EMOTION2VEC_REPORT}" \
    --model-id "${EMOTION_MODEL_ID:-emotion2vec/emotion2vec_plus_large}" \
    --revision "${EMOTION_REVISION:-main}" \
    --cache-dir "${EMOTION_CACHE_DIR:-artifacts/model_cache/audio_ssl}" \
    --device "${EMOTION_DEVICE:-auto}" \
    --dtype "${EMOTION_DTYPE:-auto}" \
    --backend "${EMOTION_BACKEND:-auto}" \
    --local-files-only
else
  echo "{\"reuse\":\"${EMOTION2VEC}\"}"
fi

if [[ "${FORCE_FEATURES}" == "1" || ! -f "${WAVLM}" ]]; then
  "${PYTHON_BIN}" -m humomni.features.wavlm_extract \
    --mode extract \
    --manifest "${MANIFEST}" \
    --split phase1_full_test \
    --feature-version feat-wavlm-chunked \
    --backend wavlm \
    --output "${WAVLM}" \
    --quality-report "${WAVLM_REPORT}" \
    --model-id "${WAVLM_MODEL_ID:-microsoft/wavlm-base-plus}" \
    --revision "${WAVLM_REVISION:-main}" \
    --cache-dir "${WAVLM_CACHE_DIR:-artifacts/model_cache/wavlm}" \
    --device "${WAVLM_DEVICE:-auto}" \
    --dtype "${WAVLM_DTYPE:-auto}" \
    --pooling-strategy "${WAVLM_POOLING_STRATEGY:-last_mean}" \
    --hidden-layers="${WAVLM_HIDDEN_LAYERS:--1,-2,-3,-4}" \
    --local-files-only
else
  echo "{\"reuse\":\"${WAVLM}\"}"
fi

"${PYTHON_BIN}" -m humomni.features.build_phase1_full_provided_text_features \
  --manifest "${MANIFEST}" \
  --prosody "${PROSODY}" \
  --emotion2vec "${EMOTION2VEC}" \
  --output "${FEATURE_TABLE}" \
  --report "${FEATURE_TABLE_REPORT}"
