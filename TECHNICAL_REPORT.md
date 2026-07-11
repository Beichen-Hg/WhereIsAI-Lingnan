# Technical Report: Phase 1 Audio Delivery Ranking

The submission-ready PDF version is [TECHNICAL_REPORT.pdf](TECHNICAL_REPORT.pdf).

## Final System

- System: `provided_text_conditioned_audio_delivery_ranking`
- Learned model: `AudioDeliveryPairwiseMLP`
- Final checkpoint directory: `artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/`
- Checkpoint SHA-256: `24303cb353b767c6f8c964876beec85cbf868633aa88e3f291cf8004a5e23083`
- Checkpoint-config SHA-256: `d26c614e9eb27207bd6c9acdc00419e64e998ec32899410e159564af1d9d5038`
- Frozen submission SHA-256: `f84f1a12568f2556db273fc3b70e7d87b6319e83cf086c7772405e3abd185222`
- Public model repository: `https://huggingface.co/BertramM/whereisai-lingnan-track1-audio`
- Public code repository: `https://github.com/Beichen-Hg/WhereIsAI-Lingnan`

The released GitHub code intentionally excludes competition data, cached
features, model weights, and individual test predictions. The final checkpoint
is distributed through the public Hugging Face model repository associated with
this code release.

## Method

The method ranks the supplied audio-response candidates with one pairwise MLP.
The input representation contains exactly these feature groups:

- Prosody from a `librosa_basic` extractor.
- Emotion embeddings from `emotion2vec/emotion2vec_plus_large`.
- Speech embeddings from `microsoft/wavlm-base-plus`.
- Task-type indicators.
- Text-length metadata derived from official JSON-provided text.

The model has 1,775 input features, hidden dimensions `[256, 128]`, and
dropout `0.15`. The final refit used batch size `128`, learning rate `0.0003`,
weight decay `0.001`, six epochs, and random seed `2026`.

ASR, teacher models, fusion, fallback rules, speech-quality feature groups,
and test-time training are not used for candidate decisions. The semantic text
source is the official JSON-provided text.

## Data Protocol

All official labeled training data was used for the final refit. Phase 1 test
data was used only for blind manifest construction, feature extraction,
inference, and submission auditing. No Phase 1 test labels are read by the
final inference entry point.

The frozen submission contains 530 rows: 200 GigaSpeech, 210 MELD, and 120
EmoV-DB. Its answer distribution is `A=269`, `B=199`, and `C=62`.

## Runtime Environment

The final release was verified on Linux with Python `3.10.20`, PyTorch
`2.9.0+cu128`, CUDA `12.8`, NumPy `2.2.6`, PyYAML `6.0.3`, librosa `0.11.0`,
SoundFile `0.13.1`, Transformers `5.8.1`, Hugging Face Hub `1.14.0`, FunASR
`1.3.1`, and ModelScope `1.36.3`.

GPU execution accelerates feature extraction but the final scorer supports
`DEVICE=cpu`. For full audio-feature reconstruction, install a compatible
PyTorch build and the system audio tooling required by the selected backend
(including `ffmpeg` where the FunASR/torchaudio audio-loading path requires it).

## Reproduction

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Download the final Hugging Face checkpoint to:

```text
artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/
```

With the organizer-provided Phase 1 assets and the corresponding final feature
files available under `artifacts/`, run:

```bash
PYTHON_BIN=python DEVICE=cpu \
  bash scripts/155_reproduce_final_submission.sh
```

The script runs dependency and submission audits, then compares its JSONL
output byte-for-byte against the frozen submission when that local audit file
is available. The final code tests are:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_submission_format.py \
  tests/test_audio_delivery_pairwise.py \
  tests/test_phase1_full_provided_text.py \
  tests/test_train_guard.py
```

The final verification run passed all 15 tests and reproduced the frozen
submission SHA-256 exactly.

## Public Release Contents

The GitHub repository contains final source code, scripts, tests, dependency
specifications, and this report. The Hugging Face repository contains the
trained checkpoint and checkpoint metadata. Raw competition data, cached
features, pretrained model copies, local environments, and individual test
predictions are deliberately excluded from public source control.
