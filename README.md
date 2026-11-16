# Software Defect Prediction

This project trains and evaluates a simple cross-project software-defect model, reports its operating tradeoffs, exposes a FastAPI prediction service, and provides a React dashboard for scoring CSV batches and reviewing the frozen experiments. The evaluation experiments remain frozen: the service and frontend do not retrain a model, choose a new threshold, or write into an evaluation directory.

## What each major component does

- `schema.py` defines the exact 20 numeric metrics and their order.
- `data.py` validates research data, identifiers, labels, and numeric metrics.
- `modeling.py` builds leakage-safe scikit-learn pipelines. Imputation and scaling are saved inside the pipeline.
- `experiment.py`, `lopo.py`, and `source_validation.py` are the existing evaluation experiments. They remain separate and reproducible.
- `reporting.py` reads frozen source-validation results and adds the missing always-defective baseline, flagged counts, flagged percentages, and a clear separation between classification and ranking results. It does not fit a model.
- `deployment.py` copies the chosen saved pipeline byte-for-byte into a separate deployment directory and records its checksum and training provenance. It refuses to replace a different artifact.
- `prediction.py` verifies the explicitly configured checksum, checks the saved feature schema, loads the pipeline once, and contains all scoring/ranking logic.
- `api.py` contains only the FastAPI lifecycle, input/output models, and routes.
- `historical_results.py` joins the intended frozen LOPO and source-validation reports into a read-only dashboard response. It does not fit or tune anything.
- `config/service.json` explicitly pins the deployed model version, manifest path, SHA-256 checksum, and maximum batch size.
- `frontend/` is a React, TypeScript, Vite, and Tailwind CSS dashboard. Its backend URL comes from `VITE_API_BASE_URL`.
- `scripts/dev.mjs` is the cross-platform local launcher. It checks prerequisites and ports, starts the backend first, waits for `/health`, starts Vite, labels both log streams, and cleans up its process trees.

## Windows local-development quick start

These commands are for local development on your computer, not public production hosting.

Run this one-time setup from the repository root. Virtual-environment activation is optional because every command uses its Python executable directly:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
npm install
npm --prefix frontend install
```

The direct dependencies are pinned in `pyproject.toml`, `requirements.txt`, and the two npm lockfiles. `requirements-lock.txt` records the complete tested Python environment.

The configured deployment model, schema, manifest, and historical reports must already exist. Check everything without starting or stopping a service:

```powershell
npm run doctor
```

If the deployment package is missing but the preserved baseline exists, create the separate deployment copy once:

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.deployment
```

This packaging command copies the saved model; it does not fit, retrain, or overwrite an evaluation model.

Start the backend and frontend together from one terminal:

```powershell
npm run dev
```

Open the dashboard at `http://127.0.0.1:5173` and FastAPI documentation at `http://127.0.0.1:8000/docs`. Press **Ctrl+C once** in that terminal to stop both managed services and their child processes.

The launcher resolves every path from its own file, so paths containing spaces are safe. It passes an absolute `DEFECT_SERVICE_CONFIG`, uses `.venv\Scripts\python.exe`, waits for backend health before Vite, injects `VITE_API_BASE_URL=http://127.0.0.1:8000` without changing `frontend/.env`, and keeps Vite on strict port 5173.

To start only one service:

```powershell
npm run dev:backend
npm run dev:frontend
```

The frontend-only command expects a compatible backend to be available at the configured API URL.

If a port is busy, startup stops before launching anything. The error includes the listening PID and commands to inspect and stop that specific process. You can run the same preflight explicitly:

```powershell
npm run doctor
Get-NetTCPConnection -State Listen -LocalPort 8000,5173
```

Never kill all Node or Python processes; inspect the reported PID and stop only a confirmed stale development server.

## Reproduce the existing experiments

Download and checksum the pinned Zenodo data:

```powershell
python -m defect_prediction.download
```

Run the original JEdit holdout baseline, saved-pipeline reevaluation, leave-one-project-out review, and source-only threshold experiment:

```powershell
python -m defect_prediction.experiment
python -m defect_prediction.evaluate
python -m defect_prediction.lopo
python -m defect_prediction.source_validation
```

The baseline holds out every JEdit release. LOPO uses each complete project as a target in turn. Source validation chooses thresholds only from inner source-project folds; outer target labels are used only for final metrics.

## Reproduce the model-improvement experiment

The bounded candidate grid and frozen replacement gates are in `config/model_improvement.json`. The runner evaluates tuned regularised logistic regression, random forest, and histogram gradient boosting, alongside the preserved logistic model, constant-prior reference, and lines-of-code ranking reference. Its outer and inner folds both group by complete project, so releases of one project never cross a training/validation boundary. Ant is not loaded by this command.

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.model_improvement run
```

The command writes `nested_results.json`, `nested_outer_results.csv`, `development_configuration_results.csv`, and `final_selection.json` under `artifacts/model_improvement`. A validated checkpoint lets an interrupted run resume without repeating completed outer folds. The frozen v1 experiment retained `logistic-regression-jedit-holdout-v1`: the best candidate gained only 0.00608 macro project AP on development data, below the predeclared 0.01 deployment gate. Consequently no candidate artifact was packaged and `config/service.json` was not changed.

Only after the selection files exist, reproduce the explicitly non-final Ant diagnostic:

```powershell
.\.venv\Scripts\python.exe -m defect_prediction.model_improvement external-ant --ant-csv .\data\external\ant-1.7.csv
```

The complete executed comparison, per-project results, decision, commands, and limitations are recorded in `artifacts/model_improvement/REPORT.md`. The `package` subcommand refuses to create a candidate artifact when the frozen selection says to retain the baseline.

## Complete baseline reporting

Generate the additional report without fitting or overwriting any experiment:

```powershell
python -m defect_prediction.reporting
```

The outputs are `artifacts/reporting/complete_baseline_report.json` and `artifacts/reporting/complete_baseline_report.md`. Every project reports the number and percentage flagged by:

- the fixed 0.5 rule;
- the source-selected threshold; and
- always predicting defective.

The always-defective baseline has recall 1.0, flags every record, and has precision equal to the target defect prevalence. It includes precision, recall, F1, and the full confusion matrix.

Several source-selected thresholds are not useful operating policies: they flag exactly 100% of Camel, Lucene, POI, and Synapse records, and 99.886% of JEdit records. The report defines “nearly all” as at least 95%. AP and top-20% recall remain in a separate ranking-results section. No additional choice is tuned against outer-test results.

## Served model and provenance

Package the deployment copy:

```powershell
python -m defect_prediction.deployment
```

This command does not call `fit`. It makes a byte-identical copy of `artifacts/baseline/logistic_pipeline.joblib` at `artifacts/deployment/logistic-regression-jedit-holdout-v1/pipeline.joblib`, alongside a schema and manifest. Repeating the command is safe only when the existing deployment artifact is identical; otherwise it stops instead of overwriting it.

Model version `logistic-regression-jedit-holdout-v1` trained on exactly these projects:

- Camel
- Log4j
- Lucene
- POI
- Synapse
- Velocity

JEdit was evaluation-only and was not in training. The model uses median imputation with missing indicators, standard scaling, and logistic regression, all from the saved pipeline. This first deployment deliberately serves the already evaluated model; it does not silently train an all-project production model.

## API behavior

The preferred local command is `npm run dev`. To run only FastAPI, use:

```powershell
npm run dev:backend
```

Startup fails if the configured manifest, version, checksum, feature schema, or model classes do not match. The process loads the pipeline once at startup.

The routes are:

- `GET /health`: confirms the configured model loaded.
- `GET /model-info`: returns model version, verified artifact checksum, exact training/evaluation projects, features, score wording, optional frozen threshold/calibration status, and batch limit.
- `GET /experiment-results`: serves a curated, read-only view of the saved seven-project LOPO and source-validation reports.
- `POST /predict`: accepts 1–100 records and returns each identifier, model score, within-batch rank, and model version.

Every record must contain `record_id` plus all 20 training metrics. Unknown fields, missing metrics, non-numeric/infinite values, blank or duplicate identifiers, empty batches, and oversized batches are rejected. A metric may be `null` because missing-value handling is part of the saved training pipeline. Identifiers are never used as predictive features.

Scores are relative defect-risk **model scores, not calibrated probabilities**. Rank 1 is the highest score in the submitted batch; equal scores are ordered deterministically by record identifier.

## Test the running API from PowerShell

Keep the server running in the first PowerShell window. In a second window, run these exact commands from the repository root:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/model-info | ConvertTo-Json -Depth 10
$requestBody = Get-Content .\config\example_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $requestBody | ConvertTo-Json -Depth 10
```

`config/example_request.json` is a complete valid request. It contains a record identifier and the exact 20 metrics in the training schema; it contains no label or defect outcome.

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs` while the service runs.

## React frontend

`npm run dev` supplies the local backend URL to Vite without overwriting a personal `.env` file. For a separately managed frontend, create `frontend/.env` from the example and run the frontend-only command from the repository root:

```powershell
Copy-Item .\frontend\.env.example .\frontend\.env
npm run dev:frontend
```

Open `http://127.0.0.1:5173`.

The frontend URL is configured in `frontend/.env`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The dashboard has three deliberately separate screens:

- **Prediction** safely previews CSV/TSV, JSON, XLSX, and dense ARFF files locally, sends only explicitly mapped valid metric records to `POST /predict`, recalculates global ranks after all bounded batches succeed, and downloads results as CSV. It supports the dashboard `record_id` plus 20-metric schema and the original Jureczko/PROMISE two-`name` layout. Blank metric cells are sent as `null` for the saved imputer; invalid records require explicit valid-rows-only consent.
- **Experiments** reads only `GET /experiment-results`. It compares dummy and logistic-regression average precision and shows the preserved precision, recall, F1, top-20% recall, and selected-threshold warning. These are historical results, not new predictions.
- **Model information** reads `GET /model-info` and shows the version, training/evaluation projects, required features, artifact handling, and limitations.

FastAPI CORS allows the local Vite origins `http://127.0.0.1:5173` and `http://localhost:5173`. Add an intentional origin to `config/service.json` before using another frontend host.

To run frontend tests and produce the static production build:

```powershell
Set-Location .\frontend
npm test
npm run build
```

The production files are written to `frontend/dist/`.

## Run automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest
npm run test:launcher
npm --prefix frontend test
npm --prefix frontend run build
```

Tests cover research-data guardrails, project isolation, frozen-pipeline persistence, reporting calculations, API input validation, configured model loading/checksums, provenance, bounded batches, and exact agreement between direct pipeline scores and API scores.

Frontend tests cover every implemented import format, delimiters, worksheet selection, safe dense ARFF parsing, duplicate headers, explicit mappings, identifiers and labels, invalid/partial rows, format parity, batch retry/cancellation, global ranking, and stable template/result exports.

Launcher tests use controlled temporary listeners and ports to cover backend/frontend conflicts, missing configuration, missing reports, backend startup failure, process cleanup, and immediate restart. They stop only processes created by the tests.

## Dataset and interpretation

The data source, CC-BY-4.0 license, label construction, metric meanings, and limitations are documented in [docs/DATASET.md](docs/DATASET.md). Supported prediction-file layouts, validation behavior, limits, and the raw-source-code boundary are documented in [docs/IMPORTING.md](docs/IMPORTING.md). Results are cross-project research results, not proof that a cutoff will transfer to every codebase. A future deployment model may be trained separately on a declared project set, but evaluation artifacts and outer-test results must remain untouched.
