# HumOmni 2026 Track 1 EmpathyEval

This repository contains the finalized Phase 1 EmpathyEval system and its
reproducibility entry points.

The GitHub repository intentionally excludes competition data, feature caches,
checkpoint weights, and per-question submission predictions. Download the final
checkpoint from the public Hugging Face model repository before running
inference:

https://huggingface.co/BertramM/whereisai-lingnan-track1-audio

## Final Method

The retained submission candidate is
`humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1`.

- System: `provided_text_conditioned_audio_delivery_ranking`
- Model: `AudioDeliveryPairwiseMLP`
- Feature groups: prosody, Emotion2Vec Plus Large, WavLM Base Plus, task, and text metadata
- Test-time exclusions: ASR, teacher calls, fusion, fallback, speech-quality features, and test-time training
- Training: final refit on all official labeled training data
- Test usage: blind feature extraction, inference, and auditing only

The frozen submission covers 530 questions: 200 GigaSpeech, 210 MELD, and 120
EmoV-DB. Its SHA-256 is:

```text
f84f1a12568f2556db273fc3b70e7d87b6319e83cf086c7772405e3abd185222
```

The full provenance, feature hashes, model hashes, and audit results are in
[FINAL_VERSION.md](FINAL_VERSION.md).

## Reproduce

Install the validated runtime dependencies first. Select the appropriate
PyTorch wheel for the target CPU or CUDA platform before installing the rest of
the requirements.

```bash
python -m pip install -r requirements.txt
```

Place the Hugging Face checkpoint under
`artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/`.
With the organizer-provided Phase 1 data and the required extracted feature
files in their documented `artifacts/` paths, the final checkpoint can
regenerate the frozen submission without an external API or teacher model:

```bash
PYTHON_BIN=python DEVICE=cpu scripts/155_reproduce_final_submission.sh
```

The script writes to `artifacts/reproduced/` by default, runs the dependency and
submission audits, then compares the generated JSONL byte-for-byte against the
frozen candidate.

The standard inference entry point also defaults to this final checkpoint:

```bash
PYTHON_BIN=python DEVICE=cpu scripts/142_infer_phase1_full_audio_delivery.sh
```

## Retained Assets

The repository retains these local, generated dependencies for reproducibility:

- Final checkpoint: `artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/`
- Frozen candidate: `artifacts/final_candidates/humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1/`
- Final source submission: `artifacts/submissions/phase1_test_full_audio_delivery/audio_delivery_all_labeled_clean_no_quality_v1/`
- All-labeled training manifest and final train/test feature inputs
- Pretrained snapshots: `emotion2vec/emotion2vec_plus_large` and `microsoft/wavlm-base-plus`

These binary assets, feature files, raw data, local environments, test caches,
and the per-question final submission are ignored by Git. The public code
release records their hashes in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
The submission-ready PDF is [TECHNICAL_REPORT.pdf](TECHNICAL_REPORT.pdf).

## Cleanup

To remove non-final generated artifacts while preserving the complete final
reproduction chain, run:

```bash
scripts/156_clean_to_final_version.sh --apply
```

The script is idempotent. It enforces a final-only source whitelist and removes
historical experiment code, tests, configurations, duplicate release archives,
and local dataset caches. It preserves raw audio required by the final method,
the final checkpoint, frozen submission, and the two pretrained snapshots used
by the final method.

## Validation

Run the core final-system tests with:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 pytest -q \
  tests/test_submission_format.py \
  tests/test_audio_delivery_pairwise.py \
  tests/test_phase1_full_provided_text.py \
  tests/test_train_guard.py
```

The scorer uses the official grouped-score protocol: each correct answer adds
one point, and a fully correct context-variant or tone-variant group adds one
bonus point.
