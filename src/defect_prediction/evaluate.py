"""Re-evaluate saved pipelines on the originally held-out project without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from .data import load_dataset
from .modeling import evaluate_predictions, positive_probabilities, split_by_project
from .schema import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate_saved(
    data_dir: Path, manifest_path: Path, artifact_dir: Path
) -> dict[str, Any]:
    config = json.loads(
        (artifact_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    data = load_dataset(data_dir, manifest_path)
    _, test = split_by_project(data, config["held_out_project"])
    x_test = test.loc[:, FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]
    results: dict[str, Any] = {}
    for name, filename in {
        "dummy_prior": "dummy_pipeline.joblib",
        "logistic_regression": "logistic_pipeline.joblib",
    }.items():
        model = joblib.load(artifact_dir / filename)
        probabilities = positive_probabilities(model, x_test)
        results[name], _ = evaluate_predictions(
            y_test, probabilities, config["decision_threshold"]
        )
    payload = {
        "held_out_project": config["held_out_project"],
        "decision_threshold": config["decision_threshold"],
        "models": results,
    }
    output = artifact_dir / "reevaluation_results.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    print(json.dumps(evaluate_saved(args.data_dir, args.manifest, args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()

