"""Run the predeclared fixed-model leave-one-project-out comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .duplicate_audit import audit_cross_split_duplicates
from .modeling import positive_probabilities, split_by_project
from .schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


MODEL_NAMES = (
    "dummy_prior",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)
METRIC_NAMES = (
    "average_precision",
    "top_20_percent_recall",
    "precision",
    "recall",
    "f1",
)
EXPECTED_CLASSIFIERS = {
    "dummy_prior": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}
PROTECTED_ARTIFACT_DIRS = (
    "baseline",
    "deployment",
    "lopo",
    "model_improvement",
    "reporting",
    "source_validation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_paths(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            hashes[path.as_posix()] = _sha256(path)
    return hashes


def _dataset_checksums(data_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    names = [name for record in manifest["records"] for name in record["files"]]
    return {name: _sha256(data_dir / name) for name in sorted(names)}


def _dependency_versions() -> dict[str, str]:
    packages = ("joblib", "numpy", "pandas", "scikit-learn", "scipy")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _record_ids(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["project"].astype(str)
        + ":"
        + frame["release"].astype(str)
        + ":"
        + frame["class_name"].astype(str)
    )


def test_record_checksum(frame: pd.DataFrame) -> str:
    """Hash the ordered outer-test identities so model inputs can be compared."""
    payload = "\n".join(_record_ids(frame).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_model_pipeline(
    model_name: str, settings: dict[str, Any], seed: int
) -> Pipeline:
    """Build one fresh, fixed pipeline from the predeclared configuration."""
    config = settings["models"][model_name]
    if model_name == "dummy_prior":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy=config["strategy"])),
            ]
        )
    elif model_name == "logistic_regression":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(config["C"]),
                        solver=config["solver"],
                        max_iter=int(config["max_iter"]),
                        random_state=int(config["random_state"]),
                    ),
                ),
            ]
        )
    elif model_name == "random_forest":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(config["n_estimators"]),
                        max_depth=config["max_depth"],
                        min_samples_leaf=int(config["min_samples_leaf"]),
                        max_features=config["max_features"],
                        class_weight=config["class_weight"],
                        n_jobs=int(config["n_jobs"]),
                        random_state=seed,
                    ),
                ),
            ]
        )
    elif model_name == "hist_gradient_boosting":
        pipeline = Pipeline(
            [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=float(config["learning_rate"]),
                        max_iter=int(config["max_iter"]),
                        max_leaf_nodes=int(config["max_leaf_nodes"]),
                        l2_regularization=float(config["l2_regularization"]),
                        early_stopping=bool(config["early_stopping"]),
                        class_weight=config["class_weight"],
                        random_state=seed,
                    ),
                )
            ]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, EXPECTED_CLASSIFIERS[model_name]):
        raise AssertionError(f"Incorrect classifier associated with {model_name}.")
    return pipeline


def top_budget_positions(
    frame: pd.DataFrame, scores: np.ndarray, fraction: float
) -> np.ndarray:
    """Select an exact ceiling budget with deterministic identity-based tie breaks."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1].")
    if len(frame) != len(scores):
        raise ValueError("Scores and test records have different lengths.")
    count = max(1, math.ceil(len(frame) * fraction))
    releases = frame["release"].astype(str).to_numpy()
    names = frame["class_name"].astype(str).to_numpy()
    order = sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            releases[index],
            names[index],
            index,
        ),
    )
    return np.asarray(order[:count], dtype=int)


def evaluate_scores(
    test: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
    top_fraction: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    target = test[TARGET_COLUMN].to_numpy(dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(test) or not np.isfinite(scores).all():
        raise ValueError("Model scores must be finite and match the test-row count.")
    selected_positions = top_budget_positions(test, scores, top_fraction)
    selected = np.zeros(len(test), dtype=bool)
    selected[selected_positions] = True
    defects = int(target.sum())
    found = int(target[selected].sum())
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    )
    metrics = {
        "average_precision": float(average_precision_score(target, scores)),
        "top_20_percent_recall": float(found / defects) if defects else None,
        "top_20_percent": {
            "fraction": float(top_fraction),
            "rounding": "ceil(test_rows * fraction), minimum 1",
            "tie_break": "score descending, release ascending, class_name ascending",
            "reviewed_records": int(selected.sum()),
            "defects_found": found,
            "total_defects": defects,
        },
        "threshold": float(threshold),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    return metrics, predictions, selected


def _fit_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    pipeline = build_model_pipeline(model_name, settings, seed)
    x_train = train.loc[:, FEATURE_COLUMNS]
    x_test = test.loc[:, FEATURE_COLUMNS]
    start = time.perf_counter()
    pipeline.fit(x_train, train[TARGET_COLUMN])
    training_seconds = time.perf_counter() - start
    fitted_names = tuple(str(name) for name in pipeline.feature_names_in_)
    if fitted_names != FEATURE_COLUMNS:
        raise AssertionError("The fitted pipeline did not receive the exact feature schema.")
    start = time.perf_counter()
    scores = np.asarray(positive_probabilities(pipeline, x_test), dtype=float)
    prediction_seconds = time.perf_counter() - start
    metrics, predicted, selected = evaluate_scores(
        test,
        scores,
        threshold=float(settings["decision_threshold"]),
        top_fraction=float(settings["top_fraction"]),
    )
    classifier = pipeline.named_steps["classifier"]
    association = {
        "reported_model": model_name,
        "classifier_class": type(classifier).__name__,
        "expected_classifier_class": EXPECTED_CLASSIFIERS[model_name].__name__,
        "matches": isinstance(classifier, EXPECTED_CLASSIFIERS[model_name]),
    }
    if not association["matches"]:
        raise AssertionError(f"Result/model mismatch for {model_name}.")
    predictions = pd.DataFrame(
        {
            "held_out_project": test["project"].to_numpy(),
            "release": test["release"].to_numpy(),
            "class_name": test["class_name"].to_numpy(),
            "record_id": _record_ids(test).to_numpy(),
            "model": model_name,
            "seed": seed,
            "is_primary_seed": seed == int(settings["primary_seed"]),
            "target": test[TARGET_COLUMN].to_numpy(dtype=int),
            "score": scores,
            "predicted_at_0_5": predicted,
            "selected_top_20_percent": selected,
        }
    )
    runtime = {
        "held_out_project": str(test["project"].iloc[0]),
        "model": model_name,
        "seed": seed,
        "is_primary_seed": seed == int(settings["primary_seed"]),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
    }
    metrics["seed"] = seed
    metrics["runtime"] = {
        "training_seconds": runtime["training_seconds"],
        "prediction_seconds": runtime["prediction_seconds"],
    }
    metrics["model_association"] = association
    return metrics, predictions, runtime


def evaluate_comparison_frame(
    data: pd.DataFrame, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Evaluate every declared model against identical project-level outer folds."""
    primary_seed = int(settings["primary_seed"])
    stochastic_seeds = [int(seed) for seed in settings["stochastic_seeds"]]
    if primary_seed not in stochastic_seeds or len(stochastic_seeds) != len(set(stochastic_seeds)):
        raise ValueError("Stochastic seeds must be unique and include the primary seed.")
    folds: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    runtimes: list[dict[str, Any]] = []
    for project in sorted(data["project"].unique().tolist()):
        train, test = split_by_project(data, project)
        identity_checksum = test_record_checksum(test)
        models: dict[str, Any] = {}
        seed_sensitivity: dict[str, dict[str, Any]] = {"random_forest": {}}
        for model_name in MODEL_NAMES:
            seeds = stochastic_seeds if model_name == "random_forest" else [primary_seed]
            for seed in seeds:
                metrics, predictions, runtime = _fit_one(
                    train, test, model_name, settings, seed
                )
                if not predictions["record_id"].equals(_record_ids(test).reset_index(drop=True)):
                    raise AssertionError("Models did not receive identical ordered test rows.")
                prediction_frames.append(predictions)
                runtimes.append(runtime)
                if seed == primary_seed:
                    models[model_name] = metrics
                if model_name == "random_forest":
                    seed_sensitivity[model_name][str(seed)] = metrics
        if set(models) != set(MODEL_NAMES):
            raise AssertionError("A primary result is missing or associated with the wrong model.")
        folds.append(
            {
                "held_out_project": project,
                "training_projects": sorted(train["project"].unique().tolist()),
                "test_releases": sorted(test["release"].unique().tolist()),
                "test_record_checksum": identity_checksum,
                "sample_count": int(len(test)),
                "defective_count": int(test[TARGET_COLUMN].sum()),
                "defect_prevalence": float(test[TARGET_COLUMN].mean()),
                "models": models,
                "seed_sensitivity": seed_sensitivity,
                "duplicate_audit": audit_cross_split_duplicates(train, test),
            }
        )
    return folds, pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(runtimes)


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_standard_deviation": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def summarize_primary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        models[model_name] = {
            metric: _distribution(
                [float(fold["models"][model_name][metric]) for fold in folds]
            )
            for metric in METRIC_NAMES
        }
        models[model_name]["confusion_matrix_totals"] = {
            key: sum(
                fold["models"][model_name]["confusion_matrix"][key]
                for fold in folds
            )
            for key in ("tn", "fp", "fn", "tp")
        }
    return {
        "project_count": len(folds),
        "averaging": "unweighted arithmetic mean across held-out projects",
        "variation": "sample standard deviation across held-out projects",
        "models": models,
    }


def primary_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for model_name in MODEL_NAMES:
            metrics = fold["models"][model_name]
            matrix = metrics["confusion_matrix"]
            rows.append(
                {
                    "held_out_project": fold["held_out_project"],
                    "model": model_name,
                    "seed": metrics["seed"],
                    "test_rows": fold["sample_count"],
                    "defective_count": fold["defective_count"],
                    "defect_prevalence": fold["defect_prevalence"],
                    **{name: metrics[name] for name in METRIC_NAMES},
                    "top_20_reviewed": metrics["top_20_percent"]["reviewed_records"],
                    "top_20_defects_found": metrics["top_20_percent"]["defects_found"],
                    **{key: matrix[key] for key in ("tn", "fp", "fn", "tp")},
                    "training_seconds": metrics["runtime"]["training_seconds"],
                    "prediction_seconds": metrics["runtime"]["prediction_seconds"],
                    "test_record_checksum": fold["test_record_checksum"],
                }
            )
    return rows


def difference_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        logistic = fold["models"]["logistic_regression"]
        for model_name in MODEL_NAMES:
            if model_name == "logistic_regression":
                continue
            metrics = fold["models"][model_name]
            rows.append(
                {
                    "held_out_project": fold["held_out_project"],
                    "model": model_name,
                    **{
                        f"{metric}_difference_from_logistic": (
                            float(metrics[metric]) - float(logistic[metric])
                        )
                        for metric in METRIC_NAMES
                    },
                }
            )
    return rows


def seed_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed, metrics in fold["seed_sensitivity"]["random_forest"].items():
            rows.append(
                {
                    "held_out_project": fold["held_out_project"],
                    "model": "random_forest",
                    "seed": int(seed),
                    **{metric: metrics[metric] for metric in METRIC_NAMES},
                }
            )
    return rows


def seed_variation_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_data = seed_rows(folds)
    frame = pd.DataFrame(seed_data)
    for project, group in frame.groupby("held_out_project", sort=True):
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "scope": "within_project_across_seeds",
                    "held_out_project": project,
                    "model": "random_forest",
                    "metric": metric,
                    "seed_count": int(len(group)),
                    **_distribution(group[metric].astype(float).tolist()),
                }
            )
    macro_by_seed = frame.groupby("seed", sort=True)[list(METRIC_NAMES)].mean()
    for metric in METRIC_NAMES:
        rows.append(
            {
                "scope": "macro_project_mean_across_seeds",
                "held_out_project": "ALL_EQUAL_WEIGHT",
                "model": "random_forest",
                "metric": metric,
                "seed_count": int(len(macro_by_seed)),
                **_distribution(macro_by_seed[metric].astype(float).tolist()),
            }
        )
    return rows


def _runtime_summary(runtimes: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, seed), group in runtimes.groupby(["model", "seed"], sort=True):
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "fit_count": int(len(group)),
                "training_seconds_total": float(group["training_seconds"].sum()),
                "training_seconds_mean_per_project": float(group["training_seconds"].mean()),
                "prediction_seconds_total": float(group["prediction_seconds"].sum()),
                "prediction_seconds_mean_per_project": float(group["prediction_seconds"].mean()),
            }
        )
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    folds = payload["folds"]
    summary = payload["summary"]
    rows = primary_rows(folds)
    differences = difference_rows(folds)
    lines = [
        "# Fixed LOPO model comparison",
        "",
        "This report executes the protocol frozen in `docs/MODEL_COMPARISON_PROTOCOL.md`. "
        "No model setting or threshold was selected from outer-test results. The prior "
        "source-selected-threshold experiment remains separate.",
        "",
        f"- Projects: {', '.join(payload['dataset']['projects'])}.",
        f"- Rows/releases: {payload['dataset']['rows']:,} / {payload['dataset']['release_count']}.",
        f"- Features: all {len(payload['features'])} documented input metrics; identifiers and labels excluded.",
        f"- Primary seed: {payload['configuration']['primary_seed']}; random-forest sensitivity seeds: "
        f"{', '.join(str(seed) for seed in payload['configuration']['stochastic_seeds'])}.",
        f"- Protected frozen artifacts unchanged: **{payload['protected_artifact_integrity']['unchanged']}**.",
        "",
        "## Primary per-project results (seed 42)",
        "",
        "Top-20% uses an exact ceiling record budget. Confusion matrices are TN / FP / FN / TP at 0.5.",
        "",
        "| Project | Model | n | Prev. | AP | Top-20 recall | P | R | F1 | TN / FP / FN / TP | Train s | Predict s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['held_out_project']} | {row['model']} | {row['test_rows']} | "
            f"{row['defect_prevalence']:.3f} | {row['average_precision']:.3f} | "
            f"{row['top_20_percent_recall']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['tn']} / {row['fp']} / "
            f"{row['fn']} / {row['tp']} | {row['training_seconds']:.3f} | "
            f"{row['prediction_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Equal-weight project averages",
            "",
            "Each project counts once. The `+/-` value is sample standard deviation across the seven projects.",
            "",
            "| Model | AP | Top-20 recall | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        model = summary["models"][model_name]
        cells = [
            f"{model[metric]['mean']:.3f} +/- {model[metric]['sample_standard_deviation']:.3f}"
            for metric in METRIC_NAMES
        ]
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Per-project differences from logistic regression",
            "",
            "Positive values favor the named model; negative values favor logistic regression.",
            "",
            "| Project | Model | AP diff. | Top-20 recall diff. | Precision diff. | Recall diff. | F1 diff. |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in differences:
        lines.append(
            f"| {row['held_out_project']} | {row['model']} | "
            f"{row['average_precision_difference_from_logistic']:+.3f} | "
            f"{row['top_20_percent_recall_difference_from_logistic']:+.3f} | "
            f"{row['precision_difference_from_logistic']:+.3f} | "
            f"{row['recall_difference_from_logistic']:+.3f} | "
            f"{row['f1_difference_from_logistic']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Seed sensitivity and runtime",
            "",
            "Random-forest seeds are repeated measurements of the same seven folds, not independent projects. "
            "The table reports variation among the equal-project macro result at each seed.",
            "",
            "| Metric | Mean across seed macros | SD | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    macro_seed = [
        row
        for row in payload["seed_variation"]
        if row["scope"] == "macro_project_mean_across_seeds"
    ]
    for row in macro_seed:
        lines.append(
            f"| {row['metric']} | {row['mean']:.6f} | "
            f"{row['sample_standard_deviation']:.6f} | {row['minimum']:.6f} | "
            f"{row['maximum']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Model | Seed | Fits | Train total s | Predict total s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["runtime_summary"]:
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['fit_count']} | "
            f"{row['training_seconds_total']:.3f} | {row['prediction_seconds_total']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Duplicate-overlap audit",
            "",
            "No rows were removed. An exact metric-vector match means the 20 predictors match; it does not prove source-code duplication.",
            "",
            "| Held-out project | Exact IDs | Same class names | Metric-vector matches n (%) | Conflicting-target matches |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in folds:
        audit = fold["duplicate_audit"]
        lines.append(
            f"| {fold['held_out_project']} | {audit['exact_identifier_match_rows']} | "
            f"{audit['same_class_name_match_rows']} | {audit['exact_feature_vector_match_rows']} "
            f"({audit['exact_feature_vector_match_percent']:.1f}%) | "
            f"{audit['feature_match_with_conflicting_target_rows']} |"
        )

    logistic_ap = summary["models"]["logistic_regression"]["average_precision"]["mean"]
    lines.extend(["", "## Interpretation", ""])
    for model_name in ("random_forest", "hist_gradient_boosting"):
        model_ap = summary["models"][model_name]["average_precision"]["mean"]
        project_diffs = [
            row["average_precision_difference_from_logistic"]
            for row in differences
            if row["model"] == model_name
        ]
        wins = sum(value > 0 for value in project_diffs)
        losses = sum(value < 0 for value in project_diffs)
        lines.append(
            f"- {model_name} macro AP was {model_ap:.6f}, a {model_ap - logistic_ap:+.6f} "
            f"difference from logistic regression, with {wins} project gains and {losses} regressions."
        )
    lines.extend(
        [
            "- Threshold metrics describe the predeclared 0.5 operating point and can move differently from ranking metrics.",
            "- The seven projects are heterogeneous and the observed differences are descriptive. No statistical significance or production-readiness claim is made.",
            "- Per-record predictions, every seed run, checksums, timings, and full-precision values are in the companion CSV and JSON files.",
            "",
        ]
    )
    return "\n".join(lines)


def run_model_comparison(
    data_dir: Path,
    manifest_path: Path,
    config_path: Path,
    protocol_path: Path,
    output_dir: Path,
    artifacts_root: Path,
) -> dict[str, Any]:
    protected_paths = [artifacts_root / name for name in PROTECTED_ARTIFACT_DIRS]
    if any(output_dir.resolve() == path.resolve() for path in protected_paths):
        raise ValueError("Model-comparison output must not overwrite a frozen artifact directory.")
    before = _hash_paths(protected_paths)
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = load_dataset(data_dir, manifest_path)
    folds, predictions, runtimes = evaluate_comparison_frame(data, settings)
    summary = summarize_primary(folds)
    seed_variation = seed_variation_rows(folds)
    releases = {
        project: sorted(group["release"].unique().tolist())
        for project, group in data.groupby("project", sort=True)
    }
    after = _hash_paths(protected_paths)
    if before != after:
        raise RuntimeError("A protected baseline, service, or frozen report changed during comparison.")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": settings["experiment_name"],
        "protocol_file": protocol_path.as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "configuration_file": config_path.as_posix(),
        "configuration": settings,
        "dependencies": _dependency_versions(),
        "dataset": {
            "name": manifest["dataset_name"],
            "rows": int(len(data)),
            "release_count": int(data[["project", "release"]].drop_duplicates().shape[0]),
            "projects": sorted(data["project"].unique().tolist()),
            "releases": releases,
            "sha256": _dataset_checksums(data_dir, manifest),
        },
        "features": list(FEATURE_COLUMNS),
        "target_definition": "defective = 1 when bug_count > 0, otherwise 0",
        "excluded_columns": [*METADATA_COLUMNS, LABEL_SOURCE_COLUMN, TARGET_COLUMN],
        "outer_protocol": "leave one complete project out; train on the other six",
        "averaging": "equal-weight arithmetic mean across the seven project folds",
        "folds": folds,
        "summary": summary,
        "per_project_differences_from_logistic": difference_rows(folds),
        "seed_results": seed_rows(folds),
        "seed_variation": seed_variation,
        "runtime_summary": _runtime_summary(runtimes),
        "prediction_file": "per_record_predictions.csv",
        "protected_artifact_integrity": {
            "unchanged": before == after,
            "sha256_before": before,
            "sha256_after": after,
        },
        "claims": {
            "statistical_significance": "not assessed or claimed",
            "production_readiness": "not claimed",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(primary_rows(folds)).to_csv(
        output_dir / "per_project_results.csv", index=False
    )
    pd.DataFrame(difference_rows(folds)).to_csv(
        output_dir / "per_project_differences.csv", index=False
    )
    pd.DataFrame(seed_rows(folds)).to_csv(output_dir / "seed_results.csv", index=False)
    pd.DataFrame(seed_variation).to_csv(output_dir / "seed_variation.csv", index=False)
    runtimes.to_csv(output_dir / "runtimes.csv", index=False)
    predictions.to_csv(output_dir / "per_record_predictions.csv", index=False)
    (output_dir / "comparison_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("config/model_comparison.json"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/MODEL_COMPARISON_PROTOCOL.md")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/model_comparison")
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    result = run_model_comparison(
        args.data_dir,
        args.manifest,
        args.config,
        args.protocol,
        args.output_dir,
        args.artifacts_root,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
