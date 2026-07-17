#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES_ARG:-}"
MANIFEST="${MANIFEST:-artifacts/manifests/phase2_test_full_provided_text.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/features/feat-phase2-full-provided-text/phase2_test}"
PROSODY="${PROSODY:-${FEATURE_ROOT}/prosody.jsonl}"
PROSODY_REPORT="${PROSODY_REPORT:-${FEATURE_ROOT}/prosody_quality_report.json}"
EMOTION2VEC="${EMOTION2VEC:-artifacts/features/feat-e2v-plus-large/phase2_full_test/emotion2vec.jsonl}"
EMOTION2VEC_REPORT="${EMOTION2VEC_REPORT:-artifacts/features/feat-e2v-plus-large/phase2_full_test/emotion2vec_quality_report.json}"
WAVLM="${WAVLM:-artifacts/features/feat-wavlm-chunked/phase2_full_test/wavlm.jsonl}"
WAVLM_REPORT="${WAVLM_REPORT:-artifacts/features/feat-wavlm-chunked/phase2_full_test/wavlm_quality_report.json}"
FEATURE_TABLE="${FEATURE_TABLE:-${FEATURE_ROOT}/feature_table.jsonl}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/humomni-numba-cache}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ARG}"
export NUMBA_CACHE_DIR

mkdir -p "${FEATURE_ROOT}" "$(dirname "${EMOTION2VEC}")" "$(dirname "${WAVLM}")"

if [[ "${FORCE_FEATURES}" == "1" || ! -f "${PROSODY}" ]]; then
  "${PYTHON_BIN}" -m humomni.features.prosody_extract \
    --manifest "${MANIFEST}" \
    --split phase2_test \
    --mode test \
    --feature-version feat-phase2-full-provided-text \
    --output "${PROSODY}" \
    --quality-report "${PROSODY_REPORT}"
else
  printf '{"reuse":"%s"}\n' "${PROSODY}"
fi

if [[ "${FORCE_FEATURES}" == "1" || ! -f "${EMOTION2VEC}" ]]; then
  "${PYTHON_BIN}" -m humomni.features.audio_ssl_extract \
    --mode extract \
    --manifest "${MANIFEST}" \
    --split phase2_test \
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
  printf '{"reuse":"%s"}\n' "${EMOTION2VEC}"
fi

if [[ "${FORCE_FEATURES}" == "1" || ! -f "${WAVLM}" ]]; then
  "${PYTHON_BIN}" -m humomni.features.wavlm_extract \
    --mode extract \
    --manifest "${MANIFEST}" \
    --split phase2_test \
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
    --progress-every "${WAVLM_PROGRESS_EVERY:-25}" \
    --local-files-only
else
  printf '{"reuse":"%s"}\n' "${WAVLM}"
fi

"${PYTHON_BIN}" -m humomni.features.build_feature_table \
  --manifest "${MANIFEST}" \
  --output "${FEATURE_TABLE}" \
  --mode test
