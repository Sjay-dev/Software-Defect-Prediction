"""Train both baselines once and write all reproducibility artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import load_dataset
from .modeling import (
    build_dummy_pipeline,
    build_logistic_pipeline,
    evaluate_predictions,
    positive_probabilities,
    split_by_project,
)
from .schema import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "projects": sorted(frame["project"].unique().tolist()),
        "releases": int(frame[["project", "release"]].drop_duplicates().shape[0]),
        "defective_rows": int(frame[TARGET_COLUMN].sum()),
        "defect_rate": float(frame[TARGET_COLUMN].mean()),
    }


def run_experiment(
    data_dir: Path,
    manifest_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    settings = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = load_dataset(data_dir, manifest_path)
    train, test = split_by_project(data, settings["held_out_project"])

    x_train = train.loc[:, FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    x_test = test.loc[:, FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    models = {
        "dummy_prior": build_dummy_pipeline(),
        "logistic_regression": build_logistic_pipeline(settings["random_state"]),
    }
    probabilities: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    model_results: dict[str, Any] = {}
    for name, model in models.items():
        # The imputer/scaler live inside each pipeline, so fit sees training projects only.
        model.fit(x_train, y_train)
        probabilities[name] = positive_probabilities(model, x_test)
        model_results[name], predictions[name] = evaluate_predictions(
            y_test, probabilities[name], settings["decision_threshold"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["dummy_prior"], output_dir / "dummy_pipeline.joblib")
    joblib.dump(
        models["logistic_regression"], output_dir / "logistic_pipeline.joblib"
    )

    schema = {
        "predictive_features": [
            {"name": name, "dtype": "numeric", "definition": FEATURE_DEFINITIONS[name]}
            for name in FEATURE_COLUMNS
        ],
        "metadata_not_used_as_features": list(METADATA_COLUMNS),
        "label_source_not_used_as_feature": LABEL_SOURCE_COLUMN,
        "target_not_used_as_feature": TARGET_COLUMN,
        "target_rule": "defective = 1 if bug_count > 0, else 0",
    }
    _write_json(output_dir / "feature_schema.json", schema)

    resolved_config = {
        **settings,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest": str(manifest_path),
        "data_license": data_manifest["license"],
        "zenodo_records": [record["record_url"] for record in data_manifest["records"]],
        "training_projects": sorted(train["project"].unique().tolist()),
        "test_projects": sorted(test["project"].unique().tolist()),
        "preprocessing_fit_scope": "training projects only",
        "runtime": {
            "python": platform.python_version(),
            "pandas": importlib.metadata.version("pandas"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
    }
    _write_json(output_dir / "experiment_config.json", resolved_config)

    results = {
        "experiment_name": settings["experiment_name"],
        "held_out_project": settings["held_out_project"],
        "decision_threshold": settings["decision_threshold"],
        "tuning": settings["tuning"],
        "all_data": _summary(data),
        "train": _summary(train),
        "test": _summary(test),
        "models": model_results,
    }
    _write_json(output_dir / "evaluation_results.json", results)

    prediction_table = test.loc[
        :, [*METADATA_COLUMNS, LABEL_SOURCE_COLUMN, TARGET_COLUMN]
    ].copy()
    for name in models:
        prediction_table[f"{name}_probability"] = probabilities[name]
        prediction_table[f"{name}_prediction"] = predictions[name]
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument(
        "--config", type=Path, default=Path("config/experiment.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    results = run_experiment(args.data_dir, args.manifest, args.config, args.output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

