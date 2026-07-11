---
library_name: pytorch
tags:
- audio
- empathy
- ranking
- emotion-recognition
---

# AudioDeliveryPairwiseMLP for Phase 1 EmpathyEval

This repository distributes the final trained checkpoint for the
`provided_text_conditioned_audio_delivery_ranking` system.

## Files

- `model.pt`: final `AudioDeliveryPairwiseMLP` checkpoint.
- `config.yaml`: model and training configuration.
- `feature_group_importance.json`: final feature-group analysis.

## Integrity

- `model.pt` SHA-256: `24303cb353b767c6f8c964876beec85cbf868633aa88e3f291cf8004a5e23083`
- `config.yaml` SHA-256: `d26c614e9eb27207bd6c9acdc00419e64e998ec32899410e159564af1d9d5038`

The expected submitted JSONL has SHA-256
`f84f1a12568f2556db273fc3b70e7d87b6319e83cf086c7772405e3abd185222`.

## Method

The scorer uses prosody, Emotion2Vec Plus Large, WavLM Base Plus, task
metadata, and text-length metadata from official JSON-provided text. It does
not use ASR, teacher models, fusion, fallback, speech-quality features, or
test-time training.

See the paired GitHub code repository and `TECHNICAL_REPORT.md` for setup,
feature construction, exact runtime versions, and reproduction instructions.
