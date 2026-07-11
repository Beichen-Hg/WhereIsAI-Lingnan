# Data Leakage Checklist

Use this checklist before creating features, training, validating, or submitting.

## Group Split

- [ ] Train/valid/test splits are made by `group_id`.
- [ ] No `group_id` appears in more than one split.
- [ ] Context-variant groups remain entirely inside one split.
- [ ] Tone-variant groups remain entirely inside one split.
- [ ] Any cross-validation fold assignment is generated at the group level.
- [ ] Split files are saved and versioned as run artifacts.

## `goodPara` / `badPara` Leakage

- [ ] Training code may parse `label`, `goodPara`, or `badPara` only from
  training-only gold files.
- [ ] Validation, test, and inference manifests do not include `label`,
  `goodPara`, `badPara`, answer keys, or equivalent fields.
- [ ] Candidate order is not treated as a label signal.
- [ ] Filenames and directory names are not parsed for `good`, `bad`, `correct`,
  `wrong`, labels, or answer letters.
- [ ] Metadata is treated as untrusted unless explicitly whitelisted.

## Scaler and Normalizer Leakage

- [ ] Scalers are fit only on train split features.
- [ ] Normalizers are fit only on train split features.
- [ ] Valid/test/inference data only use persisted train-fitted transforms.
- [ ] Per-dataset statistics are not recomputed using valid/test/inference data.
- [ ] Audio loudness or text length normalization does not use global statistics
  from held-out groups.

## Feature Cache Leakage

- [ ] Feature cache keys include `data_version`.
- [ ] Feature cache keys include `split`.
- [ ] Feature cache keys include `feature_version`.
- [ ] Train, valid, test, and inference caches are physically or logically
  separated.
- [ ] Cache lookup never falls back across splits.
- [ ] Cached features do not contain labels, gold answers, or teacher labels
  unless the cache is explicitly training-only.

## Teacher Label Leakage

- [ ] Teacher outputs are generated only for training-time auxiliary data unless
  a later official rule explicitly permits otherwise.
- [ ] Default inference does not require external APIs or hosted large models.
- [ ] Teacher-generated labels are not merged into validation/test manifests.
- [ ] Teacher confidence scores are not computed using held-out gold answers.
- [ ] Distillation targets are versioned and tied to the train split only.

## Leaderboard Feedback Overfitting

- [ ] Leaderboard submissions are not used as repeated validation feedback.
- [ ] Hyperparameters are selected using local group-disjoint validation, not
  public leaderboard movement.
- [ ] Failed leaderboard attempts are logged and not silently folded into model
  selection.
- [ ] Any manual rule added after leaderboard feedback is reviewed as potential
  overfitting.

## Run Artifact Audit

- [ ] Run config is saved.
- [ ] Manifest hash is saved.
- [ ] Feature versions are saved.
- [ ] Metrics are saved.
- [ ] Predictions are saved.
- [ ] Submission audit report is saved.
- [ ] The run can be reproduced without private notebook state.
