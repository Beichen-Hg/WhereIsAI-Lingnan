# Phase 2 Submission Procedure

This document records the reproducible blind-test workflow for the EmpathyEval
Phase 2 release. It deliberately does not include prediction rows, audio data,
feature caches, or checkpoint weights.

## Model Scope

Phase 2 reuses the frozen final Phase 1 checkpoint:

```text
artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/model.pt
SHA-256: 24303cb353b767c6f8c964876beec85cbf868633aa88e3f291cf8004a5e23083
```

The runner calls the Phase 1 pairwise prediction function directly. It does not
use ASR, teacher models, fusion, fallback, test-time training, or test labels.

## Required Official Inputs

Download and extract the official archives from the EmpathyEval dataset release
under `data/raw/empathyeval/`:

- `phase2-test_multi-context_gigaspeech.zip`
- `phase2-test_multi-context_meld.zip`
- `phase2-test_multi-emotion_emovdb.zip`

The resulting blind manifest contains the following release coverage:

| Source | Questions | Groups | Candidate options |
| --- | ---: | ---: | ---: |
| GigaSpeech | 226 | 113 | 2 |
| MELD | 168 | 56 | 3 |
| EmoV-DB | 148 | 37 | 2 |
| Total | 542 | 206 | 1,252 candidate rows |

For GigaSpeech and MELD, the official `data_index` is retained as the
submission `question_id`; EmoV-DB already provides `question_id`.

## Run

```bash
PYTHON_BIN=python scripts/160_build_phase2_full_provided_text_manifest.sh
PYTHON_BIN=python scripts/161_build_phase2_full_provided_text_features.sh
PYTHON_BIN=python DEVICE=auto scripts/162_infer_phase2_full_audio_delivery.sh
```

The feature step runs locally from cached Emotion2Vec and WavLM weights. The
WavLM extractor pads clips shorter than its convolutional receptive field on
the right with zeros; this keeps extremely short release clips valid without
changing their audio content or the model.

## Submission Preflight

The runner writes these ignored local files:

```text
artifacts/submissions/phase2_test_full_audio_delivery/
  audio_delivery_all_labeled_clean_no_quality_v1/
    submission.jsonl
    submission_audit_report.json
    dependency_audit_report.json
    inference_audit_report.json
```

The local audit checks all of the following before accepting a result:

- exactly 542 rows and 542 unique question IDs;
- release-order equality with the official manifests;
- only `question_id` and `answer` in every JSONL object;
- answers limited to candidates available for that question;
- all three official subsets present;
- complete audio/feature coverage with no empty audio embeddings; and
- no label-like fields, ASR replacement, teacher calls, or training on test
  inputs.

The organizer's published Track 1 checklist requires a single newline-delimited
JSON file whose lines have this form:

```json
{"question_id":"...","answer":"A"}
```

Do not convert the file to a JSON array. When the Team-ID naming rule applies,
rename the final JSONL-content file to `<TEAM_ID>.json` immediately before
uploading it to the designated Track 1 folder. The per-question submission file
must remain local and must not be committed to this repository.
