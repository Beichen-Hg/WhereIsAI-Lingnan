# Rules Checklist

Use this checklist before each official Track 1 submission.

## Submission JSONL

- [ ] Submission file is JSONL, with exactly one JSON object per line.
- [ ] Every row has the format `{"question_id": <int>, "answer": <str>}`.
- [ ] `answer` is one of `A`, `B`, `C`, or `D`.
- [ ] Each `question_id` appears exactly once.
- [ ] There are no extra fields in the official submission file unless the
  competition explicitly allows them.
- [ ] The row count matches the expected number of test questions.
- [ ] The file can be parsed by a strict JSONL parser.

## Team ID Filename

- [ ] The submitted filename follows the official Team ID naming rule.
- [ ] The Team ID in the filename matches the registered team.
- [ ] The filename does not encode experiment hints, leaderboard feedback, or
  private labels.
- [ ] The uploaded file is the audited final prediction file, not an intermediate
  debug file.

## Reproducibility Materials for Top-10

- [ ] Final code snapshot is preserved.
- [ ] All configs used for the submitted run are saved.
- [ ] Git commit hash is recorded when the run is executed inside a git repo.
- [ ] If no git repo is available, the run records `git_commit: unavailable`.
- [ ] Dataset manifest hash is recorded.
- [ ] Feature version and cache version are recorded.
- [ ] Train/valid split files are saved and are group-disjoint.
- [ ] Metrics, predictions, and audit report are saved.
- [ ] Random seeds are recorded.
- [ ] External resources are documented, including any teacher outputs used only
  during training.
- [ ] Default inference can run without external APIs.
- [ ] Instructions are sufficient for the organizers to reproduce the submitted
  predictions from the saved artifacts.
