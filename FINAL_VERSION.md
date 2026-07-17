# Final Phase 1 Version

The only submission candidate retained for this repository is
`humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1`.
Its 530 answer rows exactly match the frozen result:

```text
SHA-256: f84f1a12568f2556db273fc3b70e7d87b6319e83cf086c7772405e3abd185222
```

## System

- System: `provided_text_conditioned_audio_delivery_ranking`
- Learned model: `AudioDeliveryPairwiseMLP`
- Final checkpoint: `artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/model.pt`
- Final submission: `artifacts/final_candidates/humomni_phase1_full_audio_delivery_all_labeled_clean_no_quality_v1/submission.jsonl`
- Features: `prosody`, `emotion2vec`, `wavlm`, `task`, and `textmeta`
- Test-time exclusions: ASR, teacher calls, fusion, fallback, speech-quality features, and test-time training

The final model was refit on all official labeled training data. Phase 1 test data
was used only for blind feature extraction, inference, and submission auditing.

## Retained Local Dependencies

These generated files are intentionally retained locally for exact inference and
for auditing, but are ignored by Git:

- Final checkpoint and audit files under `artifacts/final_refit/audio_delivery_pairwise_all_labeled_clean_no_quality_v1/`
- Final test manifest and audio features under `artifacts/manifests/` and `artifacts/features/`
- All-labeled training manifest and final training feature inputs
- `emotion2vec/emotion2vec_plus_large` cache snapshot
- `microsoft/wavlm-base-plus` cache snapshot

Run `scripts/155_reproduce_final_submission.sh` to regenerate a submission into
`artifacts/reproduced/` and compare it byte-for-byte with the frozen final JSONL.

Run `scripts/156_clean_to_final_version.sh --apply` to enforce the final-only
repository scope. It keeps only the final manifest builder, feature extractors,
provided-text feature table, pairwise inference, submission audit, and their
focused tests. It removes all historical model families, experiment scripts,
legacy configuration, generated outputs, duplicate release archives, and local
dataset caches while preserving required raw audio and the retained final chain.

## Phase 2 Extension

Phase 2 does not introduce a second trained model or a second scoring policy.
It reuses the final checkpoint above and the Phase 1 prediction function on the
official Phase 2 blind inputs. The Phase 2 release-specific manifest builder,
feature extraction entry point, runner, and tests are documented in
[PHASE2_SUBMISSION.md](PHASE2_SUBMISSION.md). Generated Phase 2 predictions are
not versioned in Git.
