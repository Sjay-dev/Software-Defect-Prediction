# Software Defect Prediction — milestone 1

This project answers a first, narrow question: **can a simple model trained on several Java projects rank classes in a completely unseen project by defect risk?** It uses real labelled data, keeps an entire project out of training, and saves everything needed to repeat the run.

No FastAPI or React application is included in this milestone.

## What each major part does

- `config/data_manifest.json` pins the public Zenodo files and checksums. A checksum is a fingerprint that detects a changed or damaged download.
- `download.py` fetches only those pinned files and verifies every fingerprint.
- `data.py` gives columns unambiguous names and rejects unsafe inputs such as missing identifiers, invalid labels, duplicate class identifiers, or non-numeric metrics.
- `modeling.py` puts **all JEdit releases** in the test set. Its preprocessing lives inside scikit-learn pipelines, so medians and scaling values are learned only from the other projects.
- The dummy model predicts from the training defect rate and provides a “do almost nothing” reference point. Logistic regression is the first real model: it learns a weighted combination of the 20 code metrics.
- `experiment.py` trains once, evaluates once, and saves the pipelines, schema, configuration, metrics, and row-level predictions.
- `evaluate.py` loads the saved pipelines and checks them on JEdit again without retraining.
- `tests/` checks the data guardrails, the project boundary, and save/load prediction consistency.

The data source, license, label, feature meanings, and limitations are documented in [docs/DATASET.md](docs/DATASET.md).

## Exact Windows PowerShell setup

Open PowerShell in this folder, then run:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

The direct versions are pinned in `pyproject.toml` and `requirements.txt`. The tested full environment, including transitive packages, is recorded in `requirements-lock.txt`; the commands above use that complete lock before installing this local package without changing its dependencies.

## Download and verify the dataset

```powershell
python -m defect_prediction.download
```

This creates `data/raw/*.csv`. Raw files are ignored by Git because they are reproducibly retrieved from the attributed Zenodo records.

## Train and run the baseline evaluation

```powershell
python -m defect_prediction.experiment
```

The fixed experiment configuration holds out `jedit`. There is no hyperparameter search and no choice is made using JEdit’s results. Missing feature values are replaced by training-project medians, then features are standardized using training-project means and standard deviations.

Outputs are written to `artifacts/baseline/`:

- `dummy_pipeline.joblib` and `logistic_pipeline.joblib`: fitted models plus preprocessing;
- `feature_schema.json`: feature meanings and explicitly excluded metadata/outcomes;
- `experiment_config.json`: resolved split, source records, runtime versions, and settings;
- `evaluation_results.json`: average precision, precision, recall, F1, and confusion matrices;
- `test_predictions.csv`: JEdit identifiers, true labels, probabilities, and predictions.

Average precision evaluates the ranking of risky classes across every possible cutoff. Precision asks “of the classes flagged, how many were defective?” Recall asks “of all defective classes, how many were found?” F1 balances precision and recall. The confusion matrix gives the raw correct and incorrect counts. The fixed class cutoff is 0.5.

## Re-evaluate the saved models

```powershell
python -m defect_prediction.evaluate
```

This does not call `fit`; it loads the saved pipelines and writes `artifacts/baseline/reevaluation_results.json`.

## Run the tests

```powershell
python -m pytest
```

## Review with leave-one-project-out evaluation

After reproducing the original baseline, run the separate review experiment with:

```powershell
python -m defect_prediction.lopo
```

This trains the same unchanged dummy and logistic-regression pipelines seven times. Each
time, one complete project (all of its releases) is the test set and the other six projects
are training data. It writes `artifacts/lopo/lopo_results.json`,
`artifacts/lopo/lopo_comparison.csv`, `artifacts/lopo/lopo_per_project.csv`, and
`artifacts/lopo/review.md`. It also checks SHA-256
hashes before and after the run and stops if anything in `artifacts/baseline/` changes.

## Reproduce with explicit paths

The defaults above are usually simplest. Their full equivalents are:

```powershell
python -m defect_prediction.download --manifest config/data_manifest.json --output-dir data/raw
python -m defect_prediction.experiment --data-dir data/raw --manifest config/data_manifest.json --config config/experiment.json --output-dir artifacts/baseline
python -m defect_prediction.evaluate --data-dir data/raw --manifest config/data_manifest.json --artifact-dir artifacts/baseline
```

## Honest interpretation

The original held-out scores describe one frozen cross-project test only. The separate leave-one-project-out review shows how much performance changes by target project, but it is still an untuned research baseline and should not be generalized to every codebase. Future work can add confidence intervals, comparison models, and threshold selection using training data only.
