"""Evaluate unlabelled-target source-project similarity selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .duplicate_audit import audit_cross_split_duplicates
from .model_comparison import evaluate_scores, test_record_checksum
from .modeling import build_logistic_pipeline, positive_probabilities, split_by_project
from .schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


METRIC_NAMES = (
    "average_precision",
    "top_20_percent_recall",
    "precision",
    "recall",
    "f1",
)
CONFUSION_NAMES = ("tn", "fp", "fn", "tp")
PROTECTED_ARTIFACT_DIRS = (
    "baseline",
    "deployment",
    "lopo",
    "model_comparison",
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


def _hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[path.as_posix()] = _sha256(path)
    return result


def _dependency_versions() -> dict[str, str]:
    packages = ("joblib", "numpy", "pandas", "scikit-learn", "scipy")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _dataset_checksums(data_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    names = sorted(name for record in manifest["records"] for name in record["files"])
    return {name: _sha256(data_dir / name) for name in names}


def _record_ids(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["project"].astype(str)
        + ":"
        + frame["release"].astype(str)
        + ":"
        + frame["class_name"].astype(str)
    )


def fit_similarity_preprocessor(source_features: pd.DataFrame) -> Pipeline:
    """Fit similarity-only preprocessing; no target table is accepted here."""
    if tuple(source_features.columns) != FEATURE_COLUMNS:
        raise ValueError("Similarity preprocessing requires the exact 20-feature schema.")
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(source_features)
    if tuple(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise AssertionError("Identifiers or labels reached similarity preprocessing.")
    return pipeline


def project_summary(transformed: np.ndarray) -> np.ndarray:
    """Return feature medians followed by feature IQRs in schema order."""
    values = np.asarray(transformed, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("Project summary requires a transformed 20-feature matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Similarity preprocessing produced a non-finite value.")
    medians = np.median(values, axis=0)
    iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    summary = np.concatenate([medians, iqr])
    if summary.shape != (40,) or not np.isfinite(summary).all():
        raise AssertionError("Every project summary must contain 40 finite values.")
    return summary


def select_similar_sources(
    source_features: pd.DataFrame,
    source_projects: pd.Series,
    target_features: pd.DataFrame,
    target_project: str,
    *,
    count: int,
) -> dict[str, Any]:
    """Select sources using metrics and project IDs; label columns are inaccessible."""
    if tuple(source_features.columns) != FEATURE_COLUMNS:
        raise ValueError("Source similarity input must contain exactly the 20 features.")
    if tuple(target_features.columns) != FEATURE_COLUMNS:
        raise ValueError("Target similarity input must contain exactly the 20 features.")
    unique_sources = sorted(source_projects.astype(str).unique().tolist())
    if not 0 < count <= len(unique_sources):
        raise ValueError("Selected-source count is incompatible with available projects.")
    preprocessor = fit_similarity_preprocessor(source_features)
    transformed_source = np.asarray(preprocessor.transform(source_features), dtype=float)
    transformed_target = np.asarray(preprocessor.transform(target_features), dtype=float)
    target_vector = project_summary(transformed_target)
    project_values = source_projects.astype(str).to_numpy()
    distances: list[dict[str, Any]] = []
    for project in unique_sources:
        vector = project_summary(transformed_source[project_values == project])
        distance = float(np.linalg.norm(vector - target_vector))
        if not math.isfinite(distance):
            raise ValueError(f"Non-finite similarity distance for {project}.")
        distances.append({"source_project": project, "distance": distance})
    distances.sort(key=lambda row: (row["distance"], row["source_project"]))
    for rank, row in enumerate(distances, start=1):
        row["rank"] = rank
        row["selected"] = rank <= count

    imputer = preprocessor.named_steps["imputer"]
    scaler = preprocessor.named_steps["scaler"]
    all_missing_mask = source_features.isna().all(axis=0).to_numpy()
    constant_mask = np.asarray(scaler.var_) == 0.0
    return {
        "target_project": target_project,
        "selected_source_projects": [
            row["source_project"] for row in distances if row["selected"]
        ],
        "distances": distances,
        "preprocessing_audit": {
            "fit_scope": "all outer-source rows only",
            "feature_count": len(FEATURE_COLUMNS),
            "summary_value_count": 40,
            "imputer_statistics": [float(value) for value in imputer.statistics_],
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "source_all_missing_features": [
                FEATURE_COLUMNS[index]
                for index, value in enumerate(all_missing_mask)
                if value
            ],
            "source_constant_after_imputation_features": [
                FEATURE_COLUMNS[index]
                for index, value in enumerate(constant_mask)
                if value
            ],
            "target_labels_accessed": False,
        },
    }


def select_sources_for_outer(
    data: pd.DataFrame, target_project: str, *, count: int
) -> dict[str, Any]:
    source, target = split_by_project(data, target_project)
    return select_similar_sources(
        source.loc[:, FEATURE_COLUMNS],
        source["project"],
        target.loc[:, FEATURE_COLUMNS],
        target_project,
        count=count,
    )


def source_combinations(projects: Iterable[str], count: int) -> list[tuple[str, ...]]:
    ordered = sorted(set(str(project) for project in projects))
    combinations = list(itertools.combinations(ordered, count))
    if len(combinations) != math.comb(len(ordered), count):
        raise AssertionError("Source-subset enumeration is incomplete.")
    return combinations


def fit_prediction_pipeline(
    training: pd.DataFrame, *, random_state: int
) -> tuple[Pipeline, float]:
    """Fit a new baseline pipeline using only the supplied selected rows."""
    pipeline = build_logistic_pipeline(random_state)
    start = time.perf_counter()
    pipeline.fit(training.loc[:, FEATURE_COLUMNS], training[TARGET_COLUMN])
    elapsed = time.perf_counter() - start
    if tuple(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise AssertionError("Identifiers or labels reached prediction preprocessing.")
    return pipeline, float(elapsed)


def evaluate_source_set(
    training: pd.DataFrame,
    target: pd.DataFrame,
    settings: dict[str, Any],
    *,
    strategy: str,
    subset_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training_projects = sorted(training["project"].unique().tolist())
    target_projects = sorted(target["project"].unique().tolist())
    if set(training_projects) & set(target_projects):
        raise AssertionError("Target rows entered prediction training.")
    pipeline, training_seconds = fit_prediction_pipeline(
        training, random_state=int(settings["random_state"])
    )
    start = time.perf_counter()
    scores = np.asarray(
        positive_probabilities(pipeline, target.loc[:, FEATURE_COLUMNS]), dtype=float
    )
    prediction_seconds = time.perf_counter() - start
    metrics, predicted, selected = evaluate_scores(
        target,
        scores,
        threshold=float(settings["decision_threshold"]),
        top_fraction=float(settings["top_fraction"]),
    )
    result = {
        "strategy": strategy,
        "subset_id": subset_id,
        "source_projects": training_projects,
        "training_rows": int(len(training)),
        "training_defective_rows": int(training[TARGET_COLUMN].sum()),
        "training_defect_prevalence": float(training[TARGET_COLUMN].mean()),
        "target_project": target_projects[0],
        "target_rows": int(len(target)),
        "target_defect_prevalence": float(target[TARGET_COLUMN].mean()),
        "target_record_checksum": test_record_checksum(target),
        "metrics": metrics,
        "runtime": {
            "training_seconds": float(training_seconds),
            "prediction_seconds": float(prediction_seconds),
        },
    }
    predictions = pd.DataFrame(
        {
            "target_project": target["project"].to_numpy(),
            "release": target["release"].to_numpy(),
            "class_name": target["class_name"].to_numpy(),
            "record_id": _record_ids(target).to_numpy(),
            "strategy": strategy,
            "subset_id": subset_id,
            "source_projects": "|".join(training_projects),
            "target": target[TARGET_COLUMN].to_numpy(dtype=int),
            "score": scores,
            "predicted_at_0_5": predicted,
            "selected_top_20_percent": selected,
        }
    )
    return result, predictions


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_standard_deviation": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _result_value(result: dict[str, Any], name: str) -> float:
    if name in METRIC_NAMES:
        return float(result["metrics"][name])
    if name in CONFUSION_NAMES:
        return float(result["metrics"]["confusion_matrix"][name])
    if name in ("training_rows", "training_defect_prevalence"):
        return float(result[name])
    if name in ("training_seconds", "prediction_seconds"):
        return float(result["runtime"][name])
    raise KeyError(name)


def summarize_random_subsets(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != 20 or len({result["subset_id"] for result in results}) != 20:
        raise AssertionError("Random reference must contain 20 unique combinations.")
    names = (
        *METRIC_NAMES,
        *CONFUSION_NAMES,
        "training_rows",
        "training_defect_prevalence",
        "training_seconds",
        "prediction_seconds",
    )
    return {
        "combination_count": len(results),
        "aggregation": "arithmetic mean across all 20 subsets; no best-subset selection",
        "distributions": {
            name: _distribution([_result_value(result, name) for result in results])
            for name in names
        },
    }


def _differences(
    all_sources: dict[str, Any],
    similar_sources: dict[str, Any],
    random_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "similar_minus_all": {
            name: _result_value(similar_sources, name) - _result_value(all_sources, name)
            for name in METRIC_NAMES
        },
        "random_mean_minus_all": {
            name: random_summary["distributions"][name]["mean"]
            - _result_value(all_sources, name)
            for name in METRIC_NAMES
        },
        "similar_minus_random_mean": {
            name: _result_value(similar_sources, name)
            - random_summary["distributions"][name]["mean"]
            for name in METRIC_NAMES
        },
    }


def evaluate_outer_target(
    data: pd.DataFrame, target_project: str, settings: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    source, target = split_by_project(data, target_project)
    count = int(settings["selected_source_count"])
    selection = select_similar_sources(
        source.loc[:, FEATURE_COLUMNS],
        source["project"],
        target.loc[:, FEATURE_COLUMNS],
        target_project,
        count=count,
    )
    all_projects = sorted(source["project"].unique().tolist())
    combinations = source_combinations(all_projects, count)
    if len(combinations) != int(
        settings["random_subset_reference"]["expected_combinations_per_target"]
    ):
        raise AssertionError("Unexpected source-combination count.")
    prediction_frames: list[pd.DataFrame] = []

    all_result, predictions = evaluate_source_set(
        source,
        target,
        settings,
        strategy="all_sources",
        subset_id="all_six_sources",
    )
    prediction_frames.append(predictions)

    selected = selection["selected_source_projects"]
    similar_training = source[source["project"].isin(selected)].copy()
    similar_result, predictions = evaluate_source_set(
        similar_training,
        target,
        settings,
        strategy="similar_sources",
        subset_id="similar_three_sources",
    )
    prediction_frames.append(predictions)

    random_results: list[dict[str, Any]] = []
    for index, combination in enumerate(combinations, start=1):
        training = source[source["project"].isin(combination)].copy()
        result, predictions = evaluate_source_set(
            training,
            target,
            settings,
            strategy="random_subset_reference",
            subset_id=f"combination_{index:02d}",
        )
        if tuple(result["source_projects"]) != combination:
            raise AssertionError("Result was associated with the wrong source subset.")
        random_results.append(result)
        prediction_frames.append(predictions)

    random_summary = summarize_random_subsets(random_results)
    fold = {
        "target_project": target_project,
        "target_releases": sorted(target["release"].unique().tolist()),
        "target_rows": int(len(target)),
        "target_defective_rows": int(target[TARGET_COLUMN].sum()),
        "target_defect_prevalence": float(target[TARGET_COLUMN].mean()),
        "target_record_checksum": test_record_checksum(target),
        "similarity_selection": selection,
        "all_sources": all_result,
        "similar_sources": similar_result,
        "random_subsets": random_results,
        "random_subset_summary": random_summary,
        "differences_from_all_sources": _differences(
            all_result, similar_result, random_summary
        ),
        "duplicate_audit": audit_cross_split_duplicates(source, target),
    }
    return fold, pd.concat(prediction_frames, ignore_index=True)


def _strategy_value(fold: dict[str, Any], strategy: str, metric: str) -> float:
    if strategy == "random_subset_mean":
        return float(
            fold["random_subset_summary"]["distributions"][metric]["mean"]
        )
    return _result_value(fold[strategy], metric)


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = ("all_sources", "similar_sources", "random_subset_mean")
    return {
        "project_count": len(folds),
        "averaging": (
            "Random subsets averaged within target first; arithmetic mean then gives "
            "each of seven targets equal weight."
        ),
        "strategies": {
            strategy: {
                metric: _distribution(
                    [_strategy_value(fold, strategy, metric) for fold in folds]
                )
                for metric in METRIC_NAMES
            }
            for strategy in strategies
        },
        "differences_from_all_sources": {
            comparison: {
                metric: _distribution(
                    [
                        float(
                            fold["differences_from_all_sources"][comparison][metric]
                        )
                        for fold in folds
                    ]
                )
                for metric in METRIC_NAMES
            }
            for comparison in (
                "similar_minus_all",
                "random_mean_minus_all",
                "similar_minus_random_mean",
            )
        },
    }


def _baseline_reproduction(
    folds: list[dict[str, Any]], comparison_results_path: Path
) -> dict[str, Any]:
    if not comparison_results_path.exists():
        return {
            "checked": False,
            "matches": None,
            "reason": f"Reference file not found: {comparison_results_path.as_posix()}",
        }
    reference = json.loads(comparison_results_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for fold in folds:
        reference_fold = next(
            item
            for item in reference["folds"]
            if item["held_out_project"] == fold["target_project"]
        )
        expected = reference_fold["models"]["logistic_regression"]
        actual = fold["all_sources"]["metrics"]
        metric_match = all(actual[name] == expected[name] for name in METRIC_NAMES)
        confusion_match = all(
            actual["confusion_matrix"][name]
            == expected["confusion_matrix"][name]
            for name in CONFUSION_NAMES
        )
        checks.append(
            {
                "target_project": fold["target_project"],
                "metrics_match": metric_match,
                "confusion_matrix_matches": confusion_match,
            }
        )
    matches = all(
        row["metrics_match"] and row["confusion_matrix_matches"] for row in checks
    )
    if not matches:
        raise RuntimeError("All-source results did not reproduce fixed logistic results.")
    return {"checked": True, "matches": True, "folds": checks}


def selected_source_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for distance in fold["similarity_selection"]["distances"]:
            rows.append(
                {
                    "target_project": fold["target_project"],
                    **distance,
                }
            )
    return rows


def random_subset_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for result in fold["random_subsets"]:
            matrix = result["metrics"]["confusion_matrix"]
            rows.append(
                {
                    "target_project": fold["target_project"],
                    "subset_id": result["subset_id"],
                    "source_projects": "|".join(result["source_projects"]),
                    "training_rows": result["training_rows"],
                    "training_defect_prevalence": result[
                        "training_defect_prevalence"
                    ],
                    **{name: result["metrics"][name] for name in METRIC_NAMES},
                    **{name: matrix[name] for name in CONFUSION_NAMES},
                    **result["runtime"],
                }
            )
    return rows


def per_project_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        all_result = fold["all_sources"]
        similar = fold["similar_sources"]
        random = fold["random_subset_summary"]["distributions"]
        row: dict[str, Any] = {
            "target_project": fold["target_project"],
            "target_rows": fold["target_rows"],
            "target_defect_prevalence": fold["target_defect_prevalence"],
            "selected_source_projects": "|".join(similar["source_projects"]),
            "all_training_rows": all_result["training_rows"],
            "all_training_defect_prevalence": all_result[
                "training_defect_prevalence"
            ],
            "similar_training_rows": similar["training_rows"],
            "similar_training_defect_prevalence": similar[
                "training_defect_prevalence"
            ],
            "random_training_rows_mean": random["training_rows"]["mean"],
            "random_training_rows_minimum": random["training_rows"]["minimum"],
            "random_training_rows_maximum": random["training_rows"]["maximum"],
        }
        for metric in METRIC_NAMES:
            row[f"all_{metric}"] = all_result["metrics"][metric]
            row[f"similar_{metric}"] = similar["metrics"][metric]
            row[f"similar_minus_all_{metric}"] = fold[
                "differences_from_all_sources"
            ]["similar_minus_all"][metric]
            for statistic, value in random[metric].items():
                row[f"random_{metric}_{statistic}"] = value
            row[f"random_mean_minus_all_{metric}"] = fold[
                "differences_from_all_sources"
            ]["random_mean_minus_all"][metric]
            row[f"similar_minus_random_mean_{metric}"] = fold[
                "differences_from_all_sources"
            ]["similar_minus_random_mean"][metric]
        for name in CONFUSION_NAMES:
            row[f"all_{name}"] = all_result["metrics"]["confusion_matrix"][name]
            row[f"similar_{name}"] = similar["metrics"]["confusion_matrix"][name]
            for statistic, value in random[name].items():
                row[f"random_{name}_{statistic}"] = value
        row.update(
            {
                "all_training_seconds": all_result["runtime"]["training_seconds"],
                "all_prediction_seconds": all_result["runtime"][
                    "prediction_seconds"
                ],
                "similar_training_seconds": similar["runtime"]["training_seconds"],
                "similar_prediction_seconds": similar["runtime"][
                    "prediction_seconds"
                ],
            }
        )
        for name in (
            "training_rows",
            "training_defect_prevalence",
            "training_seconds",
            "prediction_seconds",
        ):
            for statistic, value in random[name].items():
                row[f"random_{name}_{statistic}"] = value
        rows.append(row)
    return rows


def equal_weight_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, metrics in summary["strategies"].items():
        for metric, distribution in metrics.items():
            rows.append({"strategy": strategy, "metric": metric, **distribution})
    for comparison, metrics in summary["differences_from_all_sources"].items():
        for metric, distribution in metrics.items():
            rows.append(
                {
                    "strategy": comparison,
                    "metric": metric,
                    **distribution,
                }
            )
    return rows


def fit_result_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        results = [
            fold["all_sources"],
            fold["similar_sources"],
            *fold["random_subsets"],
        ]
        for result in results:
            matrix = result["metrics"]["confusion_matrix"]
            rows.append(
                {
                    "target_project": fold["target_project"],
                    "strategy": result["strategy"],
                    "subset_id": result["subset_id"],
                    "source_projects": "|".join(result["source_projects"]),
                    "training_rows": result["training_rows"],
                    "training_defect_prevalence": result[
                        "training_defect_prevalence"
                    ],
                    **{name: result["metrics"][name] for name in METRIC_NAMES},
                    **{name: matrix[name] for name in CONFUSION_NAMES},
                    **result["runtime"],
                }
            )
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    folds = payload["folds"]
    summary = payload["summary"]
    lines = [
        "# Source-project similarity selection",
        "",
        "## Outcome",
        "",
        "This fixed experiment compares all six sources, the three closest sources by "
        "unlabelled metric distributions, and the mean of all 20 three-source subsets. "
        "No outer-target label selected a source, model, subset, or threshold.",
        "",
        f"- Rows/releases/projects: {payload['dataset']['rows']:,} / "
        f"{payload['dataset']['release_count']} / {len(payload['dataset']['projects'])}.",
        "- Prediction model: the unchanged fixed logistic-regression pipeline at threshold 0.5.",
        "- Similarity: source-fitted median imputation and standard scaling; 40-value median/IQR summaries; Euclidean distance.",
        "- The method accesses unlabelled target metrics. Target labels never enter selection.",
        "- The subset reference matches three source projects, not equal training-row counts.",
        f"- Existing all-source logistic results reproduced exactly: **{payload['baseline_reproduction']['matches']}**.",
        f"- Protected prior artifacts unchanged: **{payload['protected_artifact_integrity']['unchanged']}**.",
        "- Configuration: `config/source_similarity.json`; protocol: `docs/SOURCE_SIMILARITY_PROTOCOL.md`.",
        "",
        "## Selected sources and distances",
        "",
        "| Target | Selected sources (distance) | Other sources (distance) |",
        "|---|---|---|",
    ]
    for fold in folds:
        selected = [
            f"{row['source_project']} ({row['distance']:.3f})"
            for row in fold["similarity_selection"]["distances"]
            if row["selected"]
        ]
        other = [
            f"{row['source_project']} ({row['distance']:.3f})"
            for row in fold["similarity_selection"]["distances"]
            if not row["selected"]
        ]
        lines.append(
            f"| {fold['target_project']} | {', '.join(selected)} | {', '.join(other)} |"
        )

    lines.extend(
        [
            "",
            "## Per-target ranking results",
            "",
            "Random values are mean +/- sample SD with the full min-max range across 20 subsets.",
            "",
            "| Target | All AP | Similar AP (diff.) | Random AP | All top-20 R | Similar top-20 R (diff.) | Random top-20 R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in folds:
        all_metrics = fold["all_sources"]["metrics"]
        similar = fold["similar_sources"]["metrics"]
        diff = fold["differences_from_all_sources"]["similar_minus_all"]
        random = fold["random_subset_summary"]["distributions"]
        ap = random["average_precision"]
        top = random["top_20_percent_recall"]
        lines.append(
            f"| {fold['target_project']} | {all_metrics['average_precision']:.3f} | "
            f"{similar['average_precision']:.3f} ({diff['average_precision']:+.3f}) | "
            f"{ap['mean']:.3f} +/- {ap['sample_standard_deviation']:.3f} "
            f"[{ap['minimum']:.3f}, {ap['maximum']:.3f}] | "
            f"{all_metrics['top_20_percent_recall']:.3f} | "
            f"{similar['top_20_percent_recall']:.3f} "
            f"({diff['top_20_percent_recall']:+.3f}) | "
            f"{top['mean']:.3f} +/- {top['sample_standard_deviation']:.3f} "
            f"[{top['minimum']:.3f}, {top['maximum']:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Secondary metrics at 0.5",
            "",
            "Random confusion entries are means across the 20 subsets and need not be integers.",
            "",
            "| Target | Strategy | Precision | Recall | F1 | TN / FP / FN / TP |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for fold in folds:
        for label, result in (
            ("all", fold["all_sources"]),
            ("similar", fold["similar_sources"]),
        ):
            metrics = result["metrics"]
            matrix = metrics["confusion_matrix"]
            lines.append(
                f"| {fold['target_project']} | {label} | {metrics['precision']:.3f} | "
                f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
                f"{matrix['tn']:.0f} / {matrix['fp']:.0f} / {matrix['fn']:.0f} / {matrix['tp']:.0f} |"
            )
        random = fold["random_subset_summary"]["distributions"]
        lines.append(
            f"| {fold['target_project']} | random mean | {random['precision']['mean']:.3f} | "
            f"{random['recall']['mean']:.3f} | {random['f1']['mean']:.3f} | "
            f"{random['tn']['mean']:.1f} / {random['fp']['mean']:.1f} / "
            f"{random['fn']['mean']:.1f} / {random['tp']['mean']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Training data and runtime",
            "",
            "| Target | Strategy | Training rows | Training prevalence | Train s | Predict s |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for fold in folds:
        for label, result in (
            ("all", fold["all_sources"]),
            ("similar", fold["similar_sources"]),
        ):
            lines.append(
                f"| {fold['target_project']} | {label} | {result['training_rows']} | "
                f"{result['training_defect_prevalence']:.3f} | "
                f"{result['runtime']['training_seconds']:.3f} | "
                f"{result['runtime']['prediction_seconds']:.3f} |"
            )
        random = fold["random_subset_summary"]["distributions"]
        lines.append(
            f"| {fold['target_project']} | random mean [row range] | "
            f"{random['training_rows']['mean']:.1f} "
            f"[{random['training_rows']['minimum']:.0f}, {random['training_rows']['maximum']:.0f}] | "
            f"{random['training_defect_prevalence']['mean']:.3f} | "
            f"{random['training_seconds']['mean']:.3f} | "
            f"{random['prediction_seconds']['mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Equal-weight project averages",
            "",
            "The random reference is averaged within each target before targets receive equal weight.",
            "",
            "| Strategy | AP | Top-20 recall | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ("all_sources", "similar_sources", "random_subset_mean"):
        values = summary["strategies"][strategy]
        cells = [
            f"{values[name]['mean']:.3f} +/- {values[name]['sample_standard_deviation']:.3f}"
            for name in METRIC_NAMES
        ]
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")

    similarity_ap = summary["differences_from_all_sources"]["similar_minus_all"][
        "average_precision"
    ]
    similarity_top = summary["differences_from_all_sources"]["similar_minus_all"][
        "top_20_percent_recall"
    ]
    random_ap = summary["differences_from_all_sources"]["random_mean_minus_all"][
        "average_precision"
    ]
    similar_random_ap = summary["differences_from_all_sources"][
        "similar_minus_random_mean"
    ]["average_precision"]
    similar_random_top = summary["differences_from_all_sources"][
        "similar_minus_random_mean"
    ]["top_20_percent_recall"]
    ap_diffs = [
        fold["differences_from_all_sources"]["similar_minus_all"][
            "average_precision"
        ]
        for fold in folds
    ]
    top_diffs = [
        fold["differences_from_all_sources"]["similar_minus_all"][
            "top_20_percent_recall"
        ]
        for fold in folds
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Similar selection changed macro AP by {similarity_ap['mean']:+.6f} "
            f"versus all sources, improving {sum(value > 0 for value in ap_diffs)} targets "
            f"and regressing {sum(value < 0 for value in ap_diffs)}.",
            f"- Similar selection changed macro top-20% recall by {similarity_top['mean']:+.6f}, "
            f"improving {sum(value > 0 for value in top_diffs)} targets and regressing "
            f"{sum(value < 0 for value in top_diffs)}.",
            f"- The exhaustive three-source reference changed macro AP by "
            f"{random_ap['mean']:+.6f} versus all sources. Similar selection should be judged "
            "against this size-matched average, not against the best subset.",
            f"- Directly versus the three-source reference, similarity selection changed macro "
            f"AP by {similar_random_ap['mean']:+.6f} and top-20% recall by "
            f"{similar_random_top['mean']:+.6f}.",
            "- Results are mixed and heterogeneous across projects. They are descriptive; no novelty, statistical significance, or production-readiness claim is made.",
            "- Metric-vector matches were retained. A matching vector alone is not evidence of copied source code.",
            "",
            "Full-precision results and all 20 combinations are in the companion CSV files. "
            "Large per-record predictions and the complete JSON payload remain under the ignored `artifacts/source_similarity` directory.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.source_similarity",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_source_similarity_experiment(
    data_dir: Path,
    manifest_path: Path,
    config_path: Path,
    protocol_path: Path,
    output_dir: Path,
    report_dir: Path,
    artifacts_root: Path,
    comparison_results_path: Path,
) -> dict[str, Any]:
    protected_paths = [artifacts_root / name for name in PROTECTED_ARTIFACT_DIRS]
    if any(output_dir.resolve() == path.resolve() for path in protected_paths):
        raise ValueError("Similarity output must not overwrite an existing experiment.")
    before = _hash_paths(protected_paths)
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = load_dataset(data_dir, manifest_path)
    folds: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for target_project in sorted(data["project"].unique().tolist()):
        fold, predictions = evaluate_outer_target(data, target_project, settings)
        folds.append(fold)
        prediction_frames.append(predictions)
    summary = summarize_folds(folds)
    baseline_reproduction = _baseline_reproduction(folds, comparison_results_path)
    after = _hash_paths(protected_paths)
    if before != after:
        raise RuntimeError("A deployed model or previous experiment changed during this run.")
    releases = {
        project: sorted(group["release"].unique().tolist())
        for project, group in data.groupby("project", sort=True)
    }
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
            "release_count": int(
                data[["project", "release"]].drop_duplicates().shape[0]
            ),
            "projects": sorted(data["project"].unique().tolist()),
            "releases": releases,
            "sha256": _dataset_checksums(data_dir, manifest),
        },
        "features": list(FEATURE_COLUMNS),
        "excluded_columns": [*METADATA_COLUMNS, LABEL_SOURCE_COLUMN, TARGET_COLUMN],
        "target_definition": "defective = 1 when bug_count > 0, otherwise 0",
        "unlabelled_target_access": (
            "Target metrics are used for similarity; target labels are inaccessible to "
            "selection and used only for final evaluation."
        ),
        "folds": folds,
        "summary": summary,
        "baseline_reproduction": baseline_reproduction,
        "protected_artifact_integrity": {
            "unchanged": before == after,
            "sha256_before": before,
            "sha256_after": after,
        },
        "claims": {
            "novelty": "not claimed",
            "statistical_significance": "not assessed or claimed",
            "production_readiness": "not claimed",
        },
        "large_output_files": {
            "per_record_predictions": "per_record_predictions.csv",
            "fit_results": "fit_results.csv",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "per_record_predictions.csv", index=False)
    pd.DataFrame(fit_result_rows(folds)).to_csv(
        output_dir / "fit_results.csv", index=False
    )
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pd.DataFrame(per_project_rows(folds)).to_csv(
        report_dir / "per_project_summary.csv", index=False
    )
    pd.DataFrame(selected_source_rows(folds)).to_csv(
        report_dir / "selected_sources.csv", index=False
    )
    pd.DataFrame(random_subset_rows(folds)).to_csv(
        report_dir / "random_subset_results.csv", index=False
    )
    pd.DataFrame(equal_weight_rows(summary)).to_csv(
        report_dir / "equal_weight_summary.csv", index=False
    )
    (report_dir / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("config/source_similarity.json"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/SOURCE_SIMILARITY_PROTOCOL.md")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/source_similarity")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/source_similarity")
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--comparison-results",
        type=Path,
        default=Path("artifacts/model_comparison/comparison_results.json"),
    )
    args = parser.parse_args()
    payload = run_source_similarity_experiment(
        args.data_dir,
        args.manifest,
        args.config,
        args.protocol,
        args.output_dir,
        args.report_dir,
        args.artifacts_root,
        args.comparison_results,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
