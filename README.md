# Software Defect Prediction

A reproducible cross-project software-defect study with a checksum-pinned FastAPI scoring service and React dashboard. The research uses 20 static class metrics from 8,416 observations across 25 releases of seven historical Java projects. Evaluation models and reports are kept separate from the currently served `logistic-regression-jedit-holdout-v1` pipeline.

The complete methods, full-precision calculations, related work, per-experiment interpretation, and limitations are in the [consolidated final report](reports/FINAL_REPORT.md). Dataset provenance, licenses, release links, features, and label construction are in [the dataset documentation](docs/DATASET.md).

## Main findings

- Logistic regression had the highest equal-project macro average precision (AP), **0.554**, among the tested fixed configurations. Random forest achieved **0.545** and histogram gradient boosting **0.514**.
- Random forest had a slightly higher macro top-20% recall than logistic regression, **0.313 versus 0.309**, but lower AP.
- Selecting the three closest projects by unlabelled metric distributions did not improve ranking overall: macro AP was **0.534**, versus **0.554** with all sources and **0.547** for the mean exhaustive three-project reference.
- Higher F1 at threshold 0.5 does not establish better ranking. AP and top-20% recall are the primary ranking measures.

These are exploratory comparisons on seven historical projects. They establish no novelty, statistical significance, causal conclusion, production readiness, or expected performance on a new organization.

## Repository map

- `src/defect_prediction/`: validated data loading, leakage-safe experiments, report generation, deployment packaging, and API.
- `config/`: immutable data manifest and exact experiment, service, and model settings.
- `docs/`: dataset, import, and predeclared experiment protocols.
- `reports/`: compact tracked research summaries and the consolidated report.
- `artifacts/`: ignored full outputs, per-record predictions, fitted research models, and deployment package.
- `frontend/`: React, TypeScript, Vite, and Tailwind dashboard.
- `scripts/dev.mjs`: cross-platform preflight and local service launcher.

## Windows setup from a clean checkout

Use Python 3.12 and Node.js 24 from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
npm ci
npm --prefix frontend ci
```

Compact research summaries are tracked, so the Experiments page does not require ignored research artifacts. The served model is deliberately ignored and must be reproduced in a clean checkout:

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.download
.\.venv\Scripts\python.exe -m defect_prediction.experiment
.\.venv\Scripts\python.exe -m defect_prediction.deployment
npm run doctor
```

The download command uses bounded retries for transient HTTP and connection
failures, and verifies every manifest checksum before accepting a file.
Deployment packaging copies the evaluated logistic pipeline byte-for-byte,
verifies its SHA-256 checksum and schema, and refuses to replace a different
artifact. It does not retrain the model.

`config/service.json` is the release configuration. Its
`4d70c015584d9bcc5a781b6d712f6db5f08784838b00de37eb0124fef3d3f40d`
checksum protects the released model bytes produced in the pinned Windows
environment; it is not rewritten when another platform reconstructs the
experiment. Joblib/pickle files are binary artifacts, and separately fitted
models are not assumed to serialize byte-for-byte identically across operating
systems even with the same Python package versions.

Ordinary CI intentionally reconstructs the baseline on Linux. After deployment
packaging has copied the generated model and recorded its checksum, CI runs
`python -m defect_prediction.ci_config` to create an untracked service config in
the runner's temporary directory. That command independently hashes the model,
requires the actual and manifest checksums to agree, preserves the release model
version and report paths, and pins only the temporary config to those generated
bytes. `SDP_SERVICE_CONFIG` makes the launcher, its preflight subprocess, and
Uvicorn use that same config. This separates artifact integrity (exact bytes must
match their own trusted config and manifest) from experiment reproducibility
(predictions and research metrics), and leaves the release checksum unchanged.

On a non-Windows machine that deliberately reconstructs rather than obtains the
exact release artifact, use the same isolated-config flow before running the
launcher:

```bash
./.venv/bin/python -m defect_prediction.ci_config --output artifacts/service.generated.json
SDP_SERVICE_CONFIG=artifacts/service.generated.json npm run doctor
SDP_SERVICE_CONFIG=artifacts/service.generated.json npm run dev
```

Start both local services:

```powershell
npm run dev
```

Open the dashboard at `http://127.0.0.1:5173` and API documentation at `http://127.0.0.1:8000/docs`. Press Ctrl+C once to stop both managed services. The launcher also supports `npm run dev:backend`, `npm run dev:frontend`, and `npm run doctor`.

This setup establishes local release readiness only. No public hosting, publishing, push, release creation, or release tag is performed by this repository workflow.

## Reproduce all reported experiments

The order below recreates the saved baseline, seven-project LOPO evaluation, source-validated threshold experiment, fixed model comparison, exhaustive source-selection experiment, compact tracked summaries, and served deployment package:

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.download
.\.venv\Scripts\python.exe -m defect_prediction.experiment
.\.venv\Scripts\python.exe -m defect_prediction.evaluate
.\.venv\Scripts\python.exe -m defect_prediction.lopo
.\.venv\Scripts\python.exe -m defect_prediction.source_validation
.\.venv\Scripts\python.exe -m defect_prediction.model_comparison
.\.venv\Scripts\python.exe -m defect_prediction.source_similarity
.\.venv\Scripts\python.exe -m defect_prediction.research_reporting
.\.venv\Scripts\python.exe -m defect_prediction.deployment
```

The similarity runner is `defect_prediction.source_similarity`. There is no `source_similarity4` module.

Relevant fixed inputs:

- [Original baseline configuration](config/experiment.json)
- [Model-comparison configuration](config/model_comparison.json) and [protocol](docs/MODEL_COMPARISON_PROTOCOL.md)
- [Source-similarity configuration](config/source_similarity.json) and [protocol](docs/SOURCE_SIMILARITY_PROTOCOL.md)
- [Pinned dataset manifest](config/data_manifest.json)
- [Read-only service configuration](config/service.json)

Large results remain ignored under `artifacts/`. Reviewable summaries are tracked in [reports/model_comparison](reports/model_comparison) and [reports/source_similarity](reports/source_similarity). The manual `Full research reproduction` GitHub Actions workflow keeps these expensive runs separate from ordinary CI.

The earlier bounded model-improvement study remains preserved and is not used to change the service. Its command is:

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.model_improvement run
```

Its frozen decision retained the served baseline because the best development gain did not meet the predeclared replacement gate.

## Dashboard and API

The Prediction screen imports CSV/TSV, JSON, XLSX, and dense ARFF data, validates explicit feature mappings, submits bounded batches, recalculates a global deterministic ranking, and exports CSV results. Scores are relative defect-risk scores, not calibrated probabilities.

The Experiments screen reads only validated compact reports from `GET /experiment-results`. It shows:

- baseline macro and per-project results;
- every fixed model on every target project;
- AP and top-20% recall as primary ranking metrics;
- threshold-0.5 classification metrics;
- selected source projects and distances; and
- the mean, sample standard deviation, and range over all 20 three-project subsets.

Missing or invalid research summaries are isolated to the affected section and do not block predictions. Historical research models never replace the served pipeline, and opening the page never trains a model.

API routes:

- `GET /health`
- `GET /model-info`
- `GET /experiment-results`
- `POST /predict`

Example live checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 10
Invoke-RestMethod http://127.0.0.1:8000/model-info | ConvertTo-Json -Depth 10
$requestBody = Get-Content .\config\example_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $requestBody | ConvertTo-Json -Depth 10
```

Import behavior and format boundaries are documented in [docs/IMPORTING.md](docs/IMPORTING.md).

## Verification

Run the complete automated release checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
npm run test:launcher
npm --prefix frontend test
npm --prefix frontend run build
npm run doctor
```

Backend tests cover project isolation, training-only preprocessing, feature exclusion, metric and tie calculations, model/result association, source-selection validity, checksums, API validation, and direct-pipeline/API agreement. Frontend tests cover all import formats, mapping and invalid-row handling, batching, retry/cancellation, global ranking, stable exports, footer links, and the values rendered from tracked experiment summaries.

CI runs the tests and production build on the explicitly pinned `ubuntu-24.04`
image and reconstructs a checksum-verified CI model without changing the release
configuration. GitHub Action runtime versions are independent of the Node.js 24
version used by this application. The separate manual workflow executes the
complete research reproduction.

## Known limitations

- The corpus contains only seven older Java projects and observational defect labels that may be incomplete or mislinked.
- Releases repeat classes, so rows are not necessarily independent.
- Exact metric-vector matches occur across projects, sometimes with different labels; a matching vector alone is not proof of copied code and no rows were silently removed.
- Projects differ in size and prevalence. The three-project reference matches source-project count, not training-row count.
- Similarity selection accesses unlabelled target metrics and is not a strictly source-only approach.
- Fixed configurations are bounded comparisons, not an exhaustive search. No statistical-significance analysis was performed.
- The served JEdit-holdout pipeline is a research demonstration, not a calibrated or production-qualified risk model.

See [release notes](reports/RELEASE_NOTES.md) for the final milestone scope and local release status.
