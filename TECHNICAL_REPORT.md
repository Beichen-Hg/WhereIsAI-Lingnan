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

## Design Rationale and Decision Flow

The task is a ranking problem: for a shared conversational situation, the
system must select the response audio whose delivery is most appropriate. The
final design therefore focuses on audio delivery rather than generating or
rewriting text. Official text is used only as a stable conditioning signal and
for length-based metadata; the decision does not depend on an ASR transcript,
an LLM judge, or a text reranker.

For every question, the final inference path is:

1. Build a leakage-safe manifest from the official JSON-provided text and the
   candidate audio paths.
2. Extract user/candidate prosody, Emotion2Vec embeddings, and WavLM
   embeddings; attach task indicators and provided-text length metadata.
3. Build every unordered candidate pair. For each pair, concatenate the left
   and right candidate features with their signed difference, absolute
   difference, and safe ratio features.
4. Apply `AudioDeliveryPairwiseMLP` to estimate the probability that the left
   candidate is preferred to the right candidate.
5. Accumulate each candidate's pairwise win probabilities, divide by its
   number of opponents, and select the candidate with the highest mean score.
6. Run dependency and submission audits before writing JSONL output.

This pairwise design supports both two-candidate and three-candidate questions
without a separate model or a fusion rule.

### Roles of the Models and Features

| Component | Role in the final system | Used for candidate decision |
|---|---|---|
| `librosa_basic` prosody extractor | Captures delivery-level acoustic statistics such as energy, rate-related and spectral cues. | Yes |
| `emotion2vec/emotion2vec_plus_large` | Produces audio emotion embeddings for the user and each response candidate. | Yes |
| `microsoft/wavlm-base-plus` | Produces general speech-representation embeddings for the user and each response candidate. | Yes |
| Official JSON-provided text | Provides context and length metadata only; it prevents semantic text replacement by ASR. | Yes, metadata only |
| Task indicators | Distinguish the official task variants. | Yes |
| `AudioDeliveryPairwiseMLP` | Learns the final pairwise preference probability from the 1,775 final features. | Yes |
| ASR, teacher, fusion, fallback, speech-quality models | Not part of the final inference path. | No |

Emotion2Vec and WavLM are complementary pretrained audio encoders: the former
contributes emotion-oriented representations, while the latter contributes
general speech representations. The pairwise MLP learns how their user-to-
candidate relationships combine with prosody and task information for the
official ranking target.

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

## How to Use the Release

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Download the public final checkpoint to the expected model directory:

```bash
hf download BertramM/whereisai-lingnan-track1-audio \
  --local-dir artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1
```

### Standard Inference

With the organizer-provided Phase 1 assets and the corresponding final feature
files available under `artifacts/`, generate predictions with:

```bash
PYTHON_BIN=python DEVICE=cpu \
  bash scripts/142_infer_phase1_full_audio_delivery.sh
```

### Full Feature Reconstruction

When only organizer-provided raw Phase 1 assets are available, rebuild the
manifest and final feature tables before inference:

```bash
PYTHON_BIN=python bash scripts/140_build_phase1_full_provided_text_manifest.sh
PYTHON_BIN=python bash scripts/141_build_phase1_full_provided_text_features.sh
PYTHON_BIN=python DEVICE=cpu \
  bash scripts/142_infer_phase1_full_audio_delivery.sh
```

### Exact Frozen-Submission Check

The following command additionally compares output against the frozen candidate
when the local audit file is available:

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
