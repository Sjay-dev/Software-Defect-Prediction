# Release notes: final research and dashboard milestone

Release status: **locally release-ready after the verification listed below; not publicly hosted or published.** No tag, GitHub release, deployment, or push is part of this milestone.

## Included

- Consolidated research report generated from the saved full-precision experiment results.
- Tracked compact JSON/CSV/Markdown summaries for the original baseline, fixed model comparison, and source-project similarity experiment.
- Read-only Experiments dashboard views for macro and per-project results, fixed-threshold classification metrics, selected-source distances, and the exhaustive subset distribution.
- Section-level missing-report handling that leaves model scoring available.
- Correct reproduction command for `defect_prediction.source_similarity`.
- CI for backend tests, launcher tests, frontend tests, production build, and checksum-verified model reconstruction.
- Separate manually triggered workflow for the expensive complete research reproduction.

## Findings carried into the release

- Logistic regression produced the highest macro AP among the tested fixed configurations.
- Similarity-selected sources did not improve ranking overall relative to all sources or the exhaustive three-project subset mean.
- Higher F1 at threshold 0.5 is not interpreted as improved ranking.
- Similarity selection uses unlabelled target metrics; it never uses target labels for selection.

The comparisons are exploratory results on seven historical projects. They do not establish novelty, statistical significance, or production readiness.

## Compatibility and artifact policy

The served model remains `logistic-regression-jedit-holdout-v1`. Its configured version and SHA-256 pin are unchanged. Historical comparison models are not served. Existing full research outputs remain under ignored `artifacts/`; compact summaries required by the application are tracked under `reports/`.

Fresh checkouts obtain raw data through the checksum-verifying downloader and reproduce the served package with the baseline and deployment commands documented in the README. Checksum failures remain fatal.

## Verification

The working checkout and a tracked-files-only clean copy passed 46 backend tests, 33 frontend tests, 6 launcher tests, and the production frontend build. The clean copy started without `.venv`, `node_modules`, raw data, or ignored artifacts; it downloaded checksum-verified source data and reproduced the configured model SHA-256 exactly. Live health, model metadata, research-report, and prediction requests also passed. All 36 artifacts protected by the saved integrity manifest remained unchanged.

Automated frontend coverage verifies import formats, batching, global ranking, result export, footer destinations, missing-report handling, and values rendered from the tracked reports. Desktop/mobile visual inspection and screenshot capture could not be completed because the available browser-control helper stopped when it could not establish the current browser URL safely; no visual result is claimed.

## Not included

- Public hosting or a production deployment.
- A release tag or published GitHub release.
- Replacement or recalibration of the served model.
- Statistical-significance testing or a production-readiness claim.
