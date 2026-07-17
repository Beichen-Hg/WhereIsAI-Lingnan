#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

FINAL_ID="humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1"
FINAL_CANDIDATE="artifacts/final_candidates/${FINAL_ID}"
FINAL_REFIT="artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1"
FINAL_SUBMISSION="artifacts/submissions/phase1_test_full_audio_delivery/audio_delivery_all_labeled_clean_no_quality_v1"

if [[ "${1:-}" != "--apply" ]]; then
  printf '%s\n' "Dry run only. Re-run with --apply to delete non-final generated artifacts."
  printf '%s\n' "Retained candidate: ${FINAL_CANDIDATE}"
  printf '%s\n' "Retained checkpoint: ${FINAL_REFIT}"
  printf '%s\n' "Retained source submission: ${FINAL_SUBMISSION}"
  printf '%s\n' "Retained source scope: final manifest, feature extraction, inference, auditing, and tests only."
  exit 0
fi

remove_path() {
  local path="$1"
  if [[ -e "${path}" || -L "${path}" ]]; then
    rm -rf -- "${path}"
  fi
}

for path in \
  artifacts/audio_aliases \
  artifacts/data \
  artifacts/envs \
  artifacts/logs \
  artifacts/models \
  artifacts/predictions \
  artifacts/reproduced \
  artifacts/reasoner \
  artifacts/reports \
  artifacts/teacher_labels \
  artifacts/tmp; do
  remove_path "${path}"
done

shopt -s nullglob
for path in artifacts/final_candidates/*; do
  [[ "${path}" == "${FINAL_CANDIDATE}" || "${path}" == "artifacts/final_candidates/current_phase1_submission.json" ]] || remove_path "${path}"
done
for path in artifacts/final_refit/*; do
  [[ "${path}" == "${FINAL_REFIT}" ]] || remove_path "${path}"
done
for path in artifacts/submissions/*; do
  [[ "${path}" == "artifacts/submissions/phase1_test_full_audio_delivery" ]] || remove_path "${path}"
done
for path in artifacts/submissions/phase1_test_full_audio_delivery/*; do
  [[ "${path}" == "${FINAL_SUBMISSION}" ]] || remove_path "${path}"
done

for path in artifacts/features/*; do
  case "${path}" in
    artifacts/features/feat-all-labeled-audio-delivery|\
    artifacts/features/feat-all-labeled-audio-upgrade|\
    artifacts/features/feat-e2v-plus-large|\
    artifacts/features/feat-phase1-full-provided-text|\
    artifacts/features/feat-wavlm-chunked) ;;
    *) remove_path "${path}" ;;
  esac
done
for path in artifacts/features/feat-e2v-plus-large/*; do
  [[ "${path}" == "artifacts/features/feat-e2v-plus-large/phase1_full_test" ]] || remove_path "${path}"
done
for path in artifacts/features/feat-wavlm-chunked/*; do
  [[ "${path}" == "artifacts/features/feat-wavlm-chunked/phase1_full_test" ]] || remove_path "${path}"
done
remove_path artifacts/features/feat-phase1-full-provided-text/phase1_test/alignment_timing.jsonl
remove_path artifacts/features/feat-phase1-full-provided-text/phase1_test/alignment_timing_report.json
remove_path artifacts/features/feat-phase1-full-provided-text/phase1_test/opensmile.jsonl
remove_path artifacts/features/feat-phase1-full-provided-text/phase1_test/opensmile_quality_report.json
remove_path artifacts/features/feat-all-labeled-audio-upgrade/asr.jsonl
remove_path artifacts/features/feat-all-labeled-audio-upgrade/feature_table.jsonl
remove_path "${FINAL_CANDIDATE}/comparison_vs_frozen_safe.json"
remove_path "${FINAL_CANDIDATE}/comparison_vs_frozen_safe.md"

for path in artifacts/manifests/*; do
  case "${path}" in
    artifacts/manifests/all_labeled_train.jsonl|\
    artifacts/manifests/phase1_test_full_provided_text.jsonl|\
    artifacts/manifests/phase1_test_full_provided_text_report.json) ;;
    *) remove_path "${path}" ;;
  esac
done

for path in artifacts/model_cache/*; do
  case "${path}" in
    artifacts/model_cache/audio_ssl|artifacts/model_cache/wavlm) ;;
    *) remove_path "${path}" ;;
  esac
done
remove_path artifacts/model_cache/audio_ssl/.locks
remove_path artifacts/model_cache/audio_ssl/models--emotion2vec--emotion2vec_base
remove_path artifacts/model_cache/audio_ssl/emotion2vec_manifest.json
remove_path artifacts/model_cache/wavlm/.locks
remove_path artifacts/model_cache/wavlm/models--microsoft--wavlm-large
remove_path artifacts/model_cache/wavlm/wavlm_large_multilayer_manifest.json

keep_source_file() {
  case "$1" in
    src/humomni/__init__.py|\
    src/humomni/data/__init__.py|\
    src/humomni/data/build_phase1_full_test_manifest.py|\
    src/humomni/data/build_phase2_full_test_manifest.py|\
    src/humomni/eval/__init__.py|\
    src/humomni/features/__init__.py|\
    src/humomni/features/audio_ssl_extract.py|\
    src/humomni/features/build_feature_table.py|\
    src/humomni/features/build_phase1_full_provided_text_features.py|\
    src/humomni/features/feature_store.py|\
    src/humomni/features/prosody_extract.py|\
    src/humomni/features/wavlm_extract.py|\
    src/humomni/infer/__init__.py|\
    src/humomni/infer/audit_submission.py|\
    src/humomni/infer/final_audio_delivery_features.py|\
    src/humomni/infer/make_submission.py|\
    src/humomni/infer/phase1_full_audio_delivery_system.py|\
    src/humomni/infer/phase2_full_audio_delivery_system.py|\
    src/humomni/models/__init__.py|\
    src/humomni/models/audio_delivery_pairwise.py|\
    src/humomni/utils/hashing.py|\
    src/humomni/utils/io.py|\
    src/humomni/utils/train_guard.py) return 0 ;;
    *) return 1 ;;
  esac
}

keep_script() {
  case "$1" in
    scripts/140_build_phase1_full_provided_text_manifest.sh|\
    scripts/141_build_phase1_full_provided_text_features.sh|\
    scripts/142_infer_phase1_full_audio_delivery.sh|\
    scripts/155_reproduce_final_submission.sh|\
    scripts/156_clean_to_final_version.sh|\
    scripts/160_build_phase2_full_provided_text_manifest.sh|\
    scripts/161_build_phase2_full_provided_text_features.sh|\
    scripts/162_infer_phase2_full_audio_delivery.sh) return 0 ;;
    *) return 1 ;;
  esac
}

keep_test() {
  case "$1" in
    tests/test_submission_format.py|\
    tests/test_audio_delivery_pairwise.py|\
    tests/test_phase1_full_provided_text.py|\
    tests/test_phase2_full_provided_text.py|\
    tests/test_wavlm_short_input.py|\
    tests/test_train_guard.py) return 0 ;;
    *) return 1 ;;
  esac
}

while IFS= read -r -d '' path; do
  keep_source_file "$path" || remove_path "$path"
done < <(find src -type f -print0)
while IFS= read -r -d '' path; do
  keep_script "$path" || remove_path "$path"
done < <(find scripts -type f -print0)
while IFS= read -r -d '' path; do
  keep_test "$path" || remove_path "$path"
done < <(find tests -type f -print0)
find src scripts tests -type d -empty -delete

remove_path configs
remove_path .claude
remove_path .venvs
remove_path .ruff_cache
remove_path .mypy_cache
remove_path .coverage
remove_path htmlcov
remove_path data/raw/empathyeval/.cache
remove_path data/raw/empathyeval/.gitattributes
remove_path data/raw/empathyeval/phase1-test_multi-emotion_emovdb.zip
remove_path data/raw/empathyeval/phase1-test_multi-context_meld.zip

remove_path .pytest_cache
find src tests scripts -type d -name __pycache__ -prune -exec rm -rf -- {} +
find src -maxdepth 1 -type d -name '*.egg-info' -prune -exec rm -rf -- {} +

printf '%s\n' "Retained final version: ${FINAL_ID}"
