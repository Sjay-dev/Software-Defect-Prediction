"""Nested, project-grouped model comparison and honest deployment selection."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data import load_dataset, validate_dataset
from .modeling import build_logistic_pipeline, positive_probabilities
from .schema import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


DEFAULT_CONFIG = Path("config/model_improvement.json")
DEFAULT_DATA_MANIFEST = Path("config/data_manifest.json")
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("artifacts/model_improvement")
PROTECTED_DIRS = (
    Path("artifacts/baseline"),
    Path("artifacts/deployment/logistic-regression-jedit-holdout-v1"),
)


class AdaptiveSkewLog1p(BaseEstimator, TransformerMixin):
    """Learn log1p columns from training data only; preserve array shape."""

    def __init__(self, skew_threshold: float = 1.0) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, values: Any, target: Any = None) -> "AdaptiveSkewLog1p":
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError("AdaptiveSkewLog1p expects a two-dimensional feature matrix.")
        frame = pd.DataFrame(array)
        minimum = frame.min(axis=0).to_numpy(dtype=float)
        skew = frame.skew(axis=0).fillna(0.0).to_numpy(dtype=float)
        self.log_indices_ = np.flatnonzero(
            (minimum >= 0.0) & (skew >= float(self.skew_threshold))
        )
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float).copy()
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("AdaptiveSkewLog1p received a different feature shape.")
        selected = array[:, self.log_indices_]
        # Training metrics are nonnegative. Signed log1p is a defensive extension for
        # an invalid negative inference value; it never clips or fabricates a value.
        array[:, self.log_indices_] = np.sign(selected) * np.log1p(np.abs(selected))
        return array

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{index}" for index in range(self.n_features_in_)])
        return np.asarray(input_features, dtype=object)


@dataclass(frozen=True)
class ModelConfiguration:
    identifier: str
    family: str
    preprocessing: str
    parameters: dict[str, Any]
    complexity_rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "family": self.family,
            "preprocessing": self.preprocessing,
            "parameters": self.parameters,
            "complexity_rank": self.complexity_rank,
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[str(path)] = _sha256(path)
    return result


def _config_id(prefix: str, values: dict[str, Any]) -> str:
    parts = [prefix]
    for key, value in values.items():
        normalized = "none" if value is None else str(value).replace(".", "p")
        parts.append(f"{key}-{normalized}")
    return "__".join(parts)


def build_configurations(settings: dict[str, Any]) -> list[ModelConfiguration]:
    configurations = [
        ModelConfiguration(
            identifier="baseline_logistic__standard__C-1__class_weight-none",
            family="baseline_logistic",
            preprocessing="standard",
            parameters={"C": 1.0, "class_weight": None},
            complexity_rank=0,
        )
    ]
    logistic = settings["candidates"]["tuned_logistic_regression"]
    for preprocessing, c_value, class_weight in itertools.product(
        logistic["preprocessing"], logistic["C"], logistic["class_weight"]
    ):
        values = {
            "preprocessing": preprocessing,
            "C": c_value,
            "class_weight": class_weight,
        }
        configurations.append(
            ModelConfiguration(
                identifier=_config_id("tuned_logistic", values),
                family="tuned_logistic_regression",
                preprocessing=preprocessing,
                parameters={"C": float(c_value), "class_weight": class_weight},
                complexity_rank=1 if preprocessing == "standard" else 2,
            )
        )
    forest = settings["candidates"]["random_forest"]
    for n_estimators, max_depth, min_leaf, max_features, class_weight in itertools.product(
        forest["n_estimators"],
        forest["max_depth"],
        forest["min_samples_leaf"],
        forest["max_features"],
        forest["class_weight"],
    ):
        values = {
            "trees": n_estimators,
            "depth": max_depth,
            "leaf": min_leaf,
            "features": max_features,
            "class_weight": class_weight,
        }
        configurations.append(
            ModelConfiguration(
                identifier=_config_id("random_forest", values),
                family="random_forest",
                preprocessing="median_imputation",
                parameters={
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth,
                    "min_samples_leaf": int(min_leaf),
                    "max_features": max_features,
                    "class_weight": class_weight,
                },
                complexity_rank=3,
            )
        )
    boosting = settings["candidates"]["hist_gradient_boosting"]
    for learning_rate, max_leaf_nodes, max_iter, l2, class_weight in itertools.product(
        boosting["learning_rate"],
        boosting["max_leaf_nodes"],
        boosting["max_iter"],
        boosting["l2_regularization"],
        boosting["class_weight"],
    ):
        values = {
            "rate": learning_rate,
            "leaves": max_leaf_nodes,
            "iterations": max_iter,
            "l2": l2,
            "class_weight": class_weight,
        }
        configurations.append(
            ModelConfiguration(
                identifier=_config_id("hist_gradient_boosting", values),
                family="hist_gradient_boosting",
                preprocessing="median_imputation",
                parameters={
                    "learning_rate": float(learning_rate),
                    "max_leaf_nodes": int(max_leaf_nodes),
                    "max_iter": int(max_iter),
                    "l2_regularization": float(l2),
                    "class_weight": class_weight,
                },
                complexity_rank=4,
            )
        )
    identifiers = [configuration.identifier for configuration in configurations]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Candidate configuration identifiers are not unique.")
    return configurations


def build_pipeline(configuration: ModelConfiguration, random_state: int) -> Pipeline:
    if configuration.family == "baseline_logistic":
        return build_logistic_pipeline(random_state)
    if configuration.family == "tuned_logistic_regression":
        if configuration.preprocessing == "standard":
            steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        elif configuration.preprocessing == "adaptive_log1p_robust":
            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("adaptive_log1p", AdaptiveSkewLog1p(skew_threshold=1.0)),
                ("scaler", RobustScaler()),
            ]
        else:
            raise ValueError(f"Unknown preprocessing: {configuration.preprocessing}")
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    C=configuration.parameters["C"],
                    class_weight=configuration.parameters["class_weight"],
                    max_iter=3000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            )
        )
        return Pipeline(steps)
    if configuration.family == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        **configuration.parameters,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if configuration.family == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        **configuration.parameters,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model family: {configuration.family}")


def _rank_budget(scores: np.ndarray, identifiers: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1].")
    count = max(1, math.ceil(len(scores) * fraction))
    order = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), str(identifiers[index]), index),
    )
    return np.asarray(order[:count], dtype=int)


def _reliability(target: np.ndarray, scores: np.ndarray, bins: int) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_score": None,
                    "observed_rate": None,
                }
            )
            continue
        mean_score = float(scores[mask].mean())
        observed_rate = float(target[mask].mean())
        ece += (count / len(scores)) * abs(mean_score - observed_rate)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_score": mean_score,
                "observed_rate": observed_rate,
            }
        )
    return {"expected_calibration_error": float(ece), "bins": rows}


def evaluate_scores(
    target: np.ndarray,
    scores: np.ndarray,
    identifiers: np.ndarray,
    *,
    threshold: float,
    top_fraction: float,
    reliability_bins: int,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite scores.")
    selected = _rank_budget(scores, identifiers, top_fraction)
    found = int(target[selected].sum())
    total_positive = int(target.sum())
    top_precision = found / len(selected)
    top_recall = found / total_positive if total_positive else None
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    )
    reliability = _reliability(target, scores, reliability_bins)
    return {
        "records": int(len(target)),
        "positives": total_positive,
        "prevalence": float(target.mean()),
        "average_precision": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores))
        if len(np.unique(target)) == 2
        else None,
        "brier_score": float(brier_score_loss(target, scores)),
        "expected_calibration_error": reliability["expected_calibration_error"],
        "reliability": reliability["bins"],
        "top_20_percent": {
            "fraction": top_fraction,
            "rounding": "ceil(n * fraction), minimum 1",
            "tie_break": "score descending, then stable record identifier",
            "reviewed_records": int(len(selected)),
            "defects_found": found,
            "recall": top_recall,
            "precision": float(top_precision),
        },
        "threshold": float(threshold),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _record_ids(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["project"].astype(str)
        + ":"
        + frame["release"].astype(str)
        + ":"
        + frame["class_name"].astype(str)
    ).to_numpy(dtype=str)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    configuration: ModelConfiguration,
    random_state: int,
) -> np.ndarray:
    pipeline = build_pipeline(configuration, random_state)
    pipeline.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])
    return np.asarray(
        positive_probabilities(pipeline, test.loc[:, FEATURE_COLUMNS]), dtype=float
    )


def _loc_scores(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    reference = np.sort(train["loc"].to_numpy(dtype=float))
    return np.searchsorted(reference, test["loc"].to_numpy(dtype=float), side="right") / len(reference)


def cross_project_predictions(
    frame: pd.DataFrame,
    *,
    configuration: ModelConfiguration | None,
    reference: str | None,
    random_state: int,
) -> pd.DataFrame:
    projects = sorted(frame["project"].unique().tolist())
    rows: list[pd.DataFrame] = []
    for held_out in projects:
        test = frame.loc[frame["project"].eq(held_out)].copy()
        train = frame.loc[~frame["project"].eq(held_out)].copy()
        if set(train["project"]) & set(test["project"]):
            raise AssertionError("Project leakage in grouped prediction generation.")
        if configuration is not None:
            scores = _fit_predict(train, test, configuration, random_state)
        elif reference == "constant_prior":
            scores = np.full(len(test), float(train[TARGET_COLUMN].mean()))
        elif reference == "loc_ranking":
            scores = _loc_scores(train, test)
        else:
            raise ValueError("A configuration or known reference is required.")
        rows.append(
            pd.DataFrame(
                {
                    "project": test["project"].to_numpy(),
                    "record_id": _record_ids(test),
                    "target": test[TARGET_COLUMN].to_numpy(dtype=int),
                    "score": scores,
                },
                index=test.index,
            )
        )
    result = pd.concat(rows).sort_index()
    if set(result.index) != set(frame.index):
        raise AssertionError("Grouped predictions do not cover the source frame exactly once.")
    return result


def select_threshold(
    predictions: pd.DataFrame, thresholds: list[float]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        per_project = []
        for _, group in predictions.groupby("project", sort=True):
            predicted = (group["score"].to_numpy() >= threshold).astype(int)
            target = group["target"].to_numpy(dtype=int)
            per_project.append(
                {
                    "f1": f1_score(target, predicted, zero_division=0),
                    "precision": precision_score(target, predicted, zero_division=0),
                    "recall": recall_score(target, predicted, zero_division=0),
                }
            )
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(np.mean([item["f1"] for item in per_project])),
                "macro_precision": float(
                    np.mean([item["precision"] for item in per_project])
                ),
                "macro_recall": float(
                    np.mean([item["recall"] for item in per_project])
                ),
            }
        )
    selected = sorted(
        rows,
        key=lambda item: (
            -item["macro_f1"],
            -item["macro_precision"],
            abs(item["threshold"] - 0.5),
            -item["threshold"],
        ),
    )[0]
    return {"selected": selected, "grid_results": rows}


def summarize_grouped_predictions(
    predictions: pd.DataFrame,
    settings: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    projects: dict[str, Any] = {}
    for project, group in predictions.groupby("project", sort=True):
        projects[str(project)] = evaluate_scores(
            group["target"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
            group["record_id"].to_numpy(dtype=str),
            threshold=threshold,
            top_fraction=float(settings["top_fraction"]),
            reliability_bins=int(settings["reliability_bins"]),
        )
    metric_paths = {
        "average_precision": lambda item: item["average_precision"],
        "top_20_percent_recall": lambda item: item["top_20_percent"]["recall"],
        "top_20_percent_precision": lambda item: item["top_20_percent"]["precision"],
        "roc_auc": lambda item: item["roc_auc"],
        "brier_score": lambda item: item["brier_score"],
        "expected_calibration_error": lambda item: item[
            "expected_calibration_error"
        ],
        "precision": lambda item: item["precision"],
        "recall": lambda item: item["recall"],
        "f1": lambda item: item["f1"],
    }
    macro = {}
    for name, getter in metric_paths.items():
        values = [getter(item) for item in projects.values()]
        finite = [float(value) for value in values if value is not None]
        macro[name] = float(np.mean(finite)) if finite else None
    confusion = {
        key: int(sum(item["confusion_matrix"][key] for item in projects.values()))
        for key in ("tn", "fp", "fn", "tp")
    }
    return {
        "threshold": float(threshold),
        "macro": macro,
        "confusion_matrix_total": confusion,
        "projects": projects,
    }


def evaluate_configuration_cv(
    frame: pd.DataFrame,
    configuration: ModelConfiguration,
    settings: dict[str, Any],
) -> dict[str, Any]:
    predictions = cross_project_predictions(
        frame,
        configuration=configuration,
        reference=None,
        random_state=int(settings["random_state"]),
    )
    summary = summarize_grouped_predictions(predictions, settings)
    threshold = select_threshold(predictions, [float(item) for item in settings["threshold_grid"]])
    return {
        "configuration": configuration.as_dict(),
        "summary_at_0_5": summary,
        "selected_threshold": threshold,
        "predictions": predictions,
    }


def choose_configuration(
    results: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    if not results:
        raise ValueError("No candidate results were supplied.")
    best_ap = max(
        result["summary_at_0_5"]["macro"]["average_precision"] for result in results
    )
    ap_eligible = [
        result
        for result in results
        if best_ap - result["summary_at_0_5"]["macro"]["average_precision"]
        <= float(selection["average_precision_tolerance"])
    ]
    best_recall = max(
        result["summary_at_0_5"]["macro"]["top_20_percent_recall"]
        for result in ap_eligible
    )
    recall_eligible = [
        result
        for result in ap_eligible
        if best_recall
        - result["summary_at_0_5"]["macro"]["top_20_percent_recall"]
        <= float(selection["top_20_recall_tolerance"])
    ]
    chosen = sorted(
        recall_eligible,
        key=lambda result: (
            result["configuration"]["complexity_rank"],
            -result["summary_at_0_5"]["macro"]["average_precision"],
            -result["summary_at_0_5"]["macro"]["top_20_percent_recall"],
            result["configuration"]["identifier"],
        ),
    )[0]
    return {
        "chosen": chosen,
        "best_observed_macro_average_precision": best_ap,
        "best_eligible_macro_top_20_recall": best_recall,
        "ap_eligible_identifiers": [
            item["configuration"]["identifier"] for item in ap_eligible
        ],
        "recall_eligible_identifiers": [
            item["configuration"]["identifier"] for item in recall_eligible
        ],
        "rule": (
            "Keep configurations within the configured AP tolerance of the best macro "
            "project AP; then within the configured top-20% recall tolerance; then prefer "
            "the lower complexity rank."
        ),
    }


def _serializable_cv_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "predictions"}


def _evaluate_outer_scores(
    test: pd.DataFrame,
    scores: np.ndarray,
    selected_threshold: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "target": test[TARGET_COLUMN].to_numpy(dtype=int),
        "scores": scores,
        "identifiers": _record_ids(test),
        "top_fraction": float(settings["top_fraction"]),
        "reliability_bins": int(settings["reliability_bins"]),
    }
    return {
        "at_0_5": evaluate_scores(threshold=0.5, **common),
        "at_training_selected_threshold": evaluate_scores(
            threshold=selected_threshold, **common
        ),
    }


def _summary_from_outer(
    outer_folds: list[dict[str, Any]], result_key: str
) -> dict[str, Any]:
    projects: dict[str, Any] = {}
    for fold in outer_folds:
        result = fold[result_key]
        projects[fold["held_out_project"]] = result
    variants = {}
    for variant in ("at_0_5", "at_training_selected_threshold"):
        items = [value[variant] for value in projects.values()]
        macro = {}
        for metric in (
            "average_precision",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
            "precision",
            "recall",
            "f1",
        ):
            values = [item[metric] for item in items if item[metric] is not None]
            macro[metric] = float(np.mean(values)) if values else None
        for metric in ("recall", "precision"):
            values = [item["top_20_percent"][metric] for item in items]
            macro[f"top_20_percent_{metric}"] = float(np.mean(values))
        variants[variant] = {
            "macro": macro,
            "confusion_matrix_total": {
                key: int(sum(item["confusion_matrix"][key] for item in items))
                for key in ("tn", "fp", "fn", "tp")
            },
        }
    return {"projects": projects, "variants": variants}


def _evaluate_reference_outer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reference: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    inner = cross_project_predictions(
        train,
        configuration=None,
        reference=reference,
        random_state=int(settings["random_state"]),
    )
    threshold = select_threshold(inner, [float(item) for item in settings["threshold_grid"]])
    if reference == "constant_prior":
        scores = np.full(len(test), float(train[TARGET_COLUMN].mean()))
    elif reference == "loc_ranking":
        scores = _loc_scores(train, test)
    else:
        raise ValueError(reference)
    result = _evaluate_outer_scores(
        test, scores, threshold["selected"]["threshold"], settings
    )
    return {"selected_threshold": threshold["selected"], **result}


def run_nested_experiment(
    data: pd.DataFrame,
    settings: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    protected_before = _hash_tree(PROTECTED_DIRS)
    configurations = build_configurations(settings)
    families = [
        "baseline_logistic",
        "tuned_logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
    outer_folds: list[dict[str, Any]] = []
    projects = sorted(data["project"].unique().tolist())
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "nested_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        outer_folds = checkpoint.get("outer_folds", [])
        completed = int(checkpoint.get("completed_outer_folds", -1))
        completed_projects = [fold.get("held_out_project") for fold in outer_folds]
        if completed != len(outer_folds) or completed_projects != projects[:completed]:
            raise RuntimeError(
                "Nested checkpoint does not match the deterministic project order; "
                "refusing an unsafe resume."
            )
        known_identifiers = {item.identifier for item in configurations}
        for fold in outer_folds:
            if set(fold.get("family_results", {})) != set(families):
                raise RuntimeError("Nested checkpoint contains an incomplete outer fold.")
            recorded = {
                result["configuration"]["identifier"]
                for result in fold["family_results"].values()
            }
            recorded.add(fold["selection_procedure"]["configuration"]["identifier"])
            if not recorded <= known_identifiers:
                raise RuntimeError(
                    "Nested checkpoint was produced by a different candidate grid."
                )
        print(
            f"[resume] loaded {completed}/{len(projects)} completed outer folds",
            flush=True,
        )
    for outer_number, held_out in enumerate(projects, start=1):
        if outer_number <= len(outer_folds):
            continue
        print(f"[outer {outer_number}/{len(projects)}] held out: {held_out}", flush=True)
        train = data.loc[~data["project"].eq(held_out)].copy()
        test = data.loc[data["project"].eq(held_out)].copy()
        inner_projects = sorted(train["project"].unique().tolist())
        if held_out in inner_projects:
            raise AssertionError("Outer project entered inner training data.")
        inner_results: list[dict[str, Any]] = []
        for config_number, configuration in enumerate(configurations, start=1):
            print(
                f"  [config {config_number}/{len(configurations)}] {configuration.identifier}",
                flush=True,
            )
            inner_results.append(
                evaluate_configuration_cv(train, configuration, settings)
            )
        family_selections = {}
        family_outer_results = {}
        for family in families:
            selection = choose_configuration(
                [
                    result
                    for result in inner_results
                    if result["configuration"]["family"] == family
                ],
                settings["selection"],
            )
            selected = selection["chosen"]
            configuration = next(
                item
                for item in configurations
                if item.identifier == selected["configuration"]["identifier"]
            )
            scores = _fit_predict(
                train, test, configuration, int(settings["random_state"])
            )
            threshold = selected["selected_threshold"]["selected"]["threshold"]
            family_outer_results[family] = {
                "configuration": configuration.as_dict(),
                "selected_threshold": selected["selected_threshold"]["selected"],
                **_evaluate_outer_scores(test, scores, threshold, settings),
            }
            family_selections[family] = {
                **{key: value for key, value in selection.items() if key != "chosen"},
                "chosen": _serializable_cv_result(selected),
            }
        overall_selection = choose_configuration(inner_results, settings["selection"])
        overall_selected = overall_selection["chosen"]
        overall_configuration = next(
            item
            for item in configurations
            if item.identifier
            == overall_selected["configuration"]["identifier"]
        )
        overall_scores = _fit_predict(
            train, test, overall_configuration, int(settings["random_state"])
        )
        overall_threshold = overall_selected["selected_threshold"]["selected"][
            "threshold"
        ]
        fold = {
            "held_out_project": held_out,
            "outer_training_projects": inner_projects,
            "outer_test_rows": int(len(test)),
            "family_results": family_outer_results,
            "family_inner_selections": family_selections,
            "selection_procedure": {
                "configuration": overall_configuration.as_dict(),
                "selected_threshold": overall_selected["selected_threshold"][
                    "selected"
                ],
                "inner_selection": {
                    **{
                        key: value
                        for key, value in overall_selection.items()
                        if key != "chosen"
                    },
                    "chosen": _serializable_cv_result(overall_selected),
                },
                **_evaluate_outer_scores(
                    test, overall_scores, overall_threshold, settings
                ),
            },
            "constant_prior": _evaluate_reference_outer(
                train, test, "constant_prior", settings
            ),
            "loc_ranking": _evaluate_reference_outer(
                train, test, "loc_ranking", settings
            ),
        }
        outer_folds.append(fold)
        _write_json(
            checkpoint_path,
            {
                "experiment": settings["experiment_name"],
                "configuration_identifiers": [item.identifier for item in configurations],
                "completed_outer_folds": len(outer_folds),
                "outer_folds": outer_folds,
            },
        )

    nested_summary = {
        family: _summary_from_outer(
            [
                {
                    "held_out_project": fold["held_out_project"],
                    "result": fold["family_results"][family],
                }
                for fold in outer_folds
            ],
            "result",
        )
        for family in families
    }
    nested_summary["selection_procedure"] = _summary_from_outer(
        outer_folds, "selection_procedure"
    )
    nested_summary["constant_prior"] = _summary_from_outer(
        outer_folds, "constant_prior"
    )
    nested_summary["loc_ranking"] = _summary_from_outer(
        outer_folds, "loc_ranking"
    )

    print("[final development selection] evaluating all configurations", flush=True)
    development_results = []
    for config_number, configuration in enumerate(configurations, start=1):
        print(
            f"  [config {config_number}/{len(configurations)}] {configuration.identifier}",
            flush=True,
        )
        development_results.append(
            evaluate_configuration_cv(data, configuration, settings)
        )
    development_selection = choose_configuration(
        development_results, settings["selection"]
    )
    selected = development_selection["chosen"]
    baseline = next(
        item
        for item in development_results
        if item["configuration"]["family"] == "baseline_logistic"
    )
    selected_ap = selected["summary_at_0_5"]["macro"]["average_precision"]
    baseline_ap = baseline["summary_at_0_5"]["macro"]["average_precision"]
    selected_top = selected["summary_at_0_5"]["macro"]["top_20_percent_recall"]
    baseline_top = baseline["summary_at_0_5"]["macro"]["top_20_percent_recall"]
    selected_projects = selected["summary_at_0_5"]["projects"]
    baseline_projects = baseline["summary_at_0_5"]["projects"]
    project_wins = sum(
        selected_projects[project]["average_precision"]
        > baseline_projects[project]["average_precision"]
        for project in projects
    )
    nested_selected_ap = nested_summary["selection_procedure"]["variants"][
        "at_0_5"
    ]["macro"]["average_precision"]
    nested_baseline_ap = nested_summary["baseline_logistic"]["variants"][
        "at_0_5"
    ]["macro"]["average_precision"]
    rules = settings["selection"]
    checks = {
        "candidate_is_not_baseline": selected["configuration"]["family"]
        != "baseline_logistic",
        "minimum_development_ap_gain": selected_ap - baseline_ap
        >= float(rules["minimum_deployment_ap_gain"]),
        "minimum_project_ap_wins": project_wins
        >= int(rules["minimum_project_ap_wins"]),
        "top_20_recall_not_materially_worse": selected_top - baseline_top
        >= -float(rules["maximum_top_20_recall_loss"]),
        "nested_selection_not_materially_worse": nested_selected_ap
        - nested_baseline_ap
        >= -float(rules["maximum_nested_selection_ap_loss"]),
    }
    deploy = all(checks.values())
    final_selection = {
        **{
            key: value
            for key, value in development_selection.items()
            if key != "chosen"
        },
        "chosen": _serializable_cv_result(selected),
        "baseline": _serializable_cv_result(baseline),
        "development_macro_ap_gain": selected_ap - baseline_ap,
        "development_macro_top_20_recall_gain": selected_top - baseline_top,
        "project_ap_wins": int(project_wins),
        "nested_selection_macro_ap_gain": nested_selected_ap - nested_baseline_ap,
        "replacement_checks": checks,
        "deploy_new_model": deploy,
        "decision": "deploy_candidate" if deploy else "retain_baseline",
    }
    config_rows = []
    for result in development_results:
        config_rows.append(
            {
                **result["configuration"],
                **result["summary_at_0_5"]["macro"],
                "selected_threshold": result["selected_threshold"]["selected"][
                    "threshold"
                ],
                "selected_threshold_macro_f1": result["selected_threshold"][
                    "selected"
                ]["macro_f1"],
            }
        )
    pd.DataFrame(config_rows).to_csv(
        output_dir / "development_configuration_results.csv", index=False
    )
    outer_rows = []
    for fold in outer_folds:
        for name, result in {
            **fold["family_results"],
            "selection_procedure": fold["selection_procedure"],
            "constant_prior": fold["constant_prior"],
            "loc_ranking": fold["loc_ranking"],
        }.items():
            fixed = result["at_0_5"]
            selected_result = result["at_training_selected_threshold"]
            outer_rows.append(
                {
                    "held_out_project": fold["held_out_project"],
                    "result": name,
                    "configuration": result.get("configuration", {}).get(
                        "identifier", name
                    ),
                    "selected_threshold": result["selected_threshold"][
                        "threshold"
                    ],
                    "average_precision": fixed["average_precision"],
                    "top_20_recall": fixed["top_20_percent"]["recall"],
                    "top_20_precision": fixed["top_20_percent"]["precision"],
                    "roc_auc": fixed["roc_auc"],
                    "brier_score": fixed["brier_score"],
                    "ece": fixed["expected_calibration_error"],
                    "precision_at_0_5": fixed["precision"],
                    "recall_at_0_5": fixed["recall"],
                    "f1_at_0_5": fixed["f1"],
                    "tn_at_0_5": fixed["confusion_matrix"]["tn"],
                    "fp_at_0_5": fixed["confusion_matrix"]["fp"],
                    "fn_at_0_5": fixed["confusion_matrix"]["fn"],
                    "tp_at_0_5": fixed["confusion_matrix"]["tp"],
                    "precision_selected": selected_result["precision"],
                    "recall_selected": selected_result["recall"],
                    "f1_selected": selected_result["f1"],
                    "tn_selected": selected_result["confusion_matrix"]["tn"],
                    "fp_selected": selected_result["confusion_matrix"]["fp"],
                    "fn_selected": selected_result["confusion_matrix"]["fn"],
                    "tp_selected": selected_result["confusion_matrix"]["tp"],
                }
            )
    pd.DataFrame(outer_rows).to_csv(output_dir / "nested_outer_results.csv", index=False)

    protected_after = _hash_tree(PROTECTED_DIRS)
    if protected_before != protected_after:
        raise RuntimeError("A protected baseline or rollback deployment artifact changed.")
    payload = {
        "experiment": settings["experiment_name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": settings["random_state"],
        "protocol": {
            "outer": "leave one entire project out",
            "inner": "leave one entire source project out for every configuration",
            "selection_primary": "macro project average precision",
            "selection_supporting": "macro project top-20% recall",
            "threshold_selection": "maximize macro project F1 using inner predictions only",
            "ant_usage": "excluded from all development, tuning, selection, and fitting",
            "calibration": "not calibrated; Brier score, ECE, and reliability bins are diagnostic only",
            "top_20_rule": "ceil(0.20 * records), score descending, stable identifier tie break",
        },
        "data": {
            "rows": int(len(data)),
            "projects": projects,
            "releases": int(data[["project", "release"]].drop_duplicates().shape[0]),
        },
        "configurations": [item.as_dict() for item in configurations],
        "outer_folds": outer_folds,
        "nested_summary": nested_summary,
        "final_development_selection": final_selection,
        "protected_artifacts": {
            "unchanged": True,
            "sha256_before": protected_before,
            "sha256_after": protected_after,
        },
        "runtime": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _write_json(output_dir / "nested_results.json", payload)
    _write_json(output_dir / "final_selection.json", final_selection)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return payload


def _schema_payload() -> dict[str, Any]:
    return {
        "predictive_features": [
            {"name": name, "dtype": "numeric", "definition": FEATURE_DEFINITIONS[name]}
            for name in FEATURE_COLUMNS
        ],
        "metadata_not_used_as_features": list(METADATA_COLUMNS),
        "label_source_not_used_as_feature": LABEL_SOURCE_COLUMN,
        "target_not_used_as_feature": TARGET_COLUMN,
        "target_rule": "defective = 1 if bug_count > 0, else 0",
    }


def model_version_for(configuration: ModelConfiguration) -> str:
    family = {
        "tuned_logistic_regression": "regularized-logistic",
        "random_forest": "random-forest",
        "hist_gradient_boosting": "hist-gradient-boosting",
    }.get(configuration.family, configuration.family.replace("_", "-"))
    return f"{family}-jureczko-all-v1"


def package_selected_model(
    data: pd.DataFrame,
    settings: dict[str, Any],
    results_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selection = results["final_development_selection"]
    if not selection["deploy_new_model"]:
        raise RuntimeError("The frozen replacement rule retained the baseline; no candidate may be packaged.")
    selected_raw = selection["chosen"]["configuration"]
    configuration = ModelConfiguration(
        identifier=selected_raw["identifier"],
        family=selected_raw["family"],
        preprocessing=selected_raw["preprocessing"],
        parameters=selected_raw["parameters"],
        complexity_rank=selected_raw["complexity_rank"],
    )
    version = model_version_for(configuration)
    artifact_dir = output_root / version
    if artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite deployment artifact: {artifact_dir}")
    artifact_dir.mkdir(parents=True)
    pipeline = build_pipeline(configuration, int(settings["random_state"]))
    pipeline.fit(data.loc[:, FEATURE_COLUMNS], data[TARGET_COLUMN])
    model_path = artifact_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    schema_path = artifact_dir / "feature_schema.json"
    _write_json(schema_path, _schema_payload())
    threshold = selection["chosen"]["selected_threshold"]["selected"]["threshold"]
    manifest = {
        "model_version": version,
        "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "feature_schema_file": schema_path.name,
        "model_type": f"scikit-learn {configuration.family} pipeline with saved preprocessing",
        "score_description": "Relative defect-risk model score; not a calibrated probability.",
        "training_projects": sorted(data["project"].unique().tolist()),
        "evaluation_only_projects": [],
        "external_diagnostic_projects": ["ant"],
        "training_experiment": settings["experiment_name"],
        "training_configuration": configuration.as_dict(),
        "preprocessing_fit_scope": "final fit on all seven development projects after nested grouped evaluation",
        "decision_threshold": threshold,
        "threshold_selection": "all-development leave-one-project-out predictions only; Ant excluded",
        "calibration_status": "not calibrated; scores remain ranking scores",
        "selection_results_file": str(results_path),
        "selection_results_sha256": _sha256(results_path),
        "data_manifest": str(DEFAULT_DATA_MANIFEST),
        "data_manifest_sha256": _sha256(DEFAULT_DATA_MANIFEST),
        "packaging_method": "Fresh fit of the frozen selected configuration; pipeline includes preprocessing.",
        "rollback_model_version": "logistic-regression-jedit-holdout-v1",
    }
    _write_json(artifact_dir / "model_manifest.json", manifest)
    loaded = joblib.load(model_path)
    if tuple(str(item) for item in loaded.feature_names_in_) != FEATURE_COLUMNS:
        raise RuntimeError("Packaged pipeline feature order is invalid.")
    return manifest


def load_ant_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, skiprows=1)
    expected = 24
    if raw.shape[1] != expected:
        raise ValueError(f"Ant CSV must have {expected} columns; found {raw.shape[1]}.")
    raw.columns = [*METADATA_COLUMNS, *FEATURE_COLUMNS, LABEL_SOURCE_COLUMN]
    raw["project"] = raw["project"].astype(str).str.strip()
    frame = validate_dataset(raw, require_multiple_projects=False)
    return frame


def evaluate_external_ant(
    ant_path: Path,
    baseline_model: Path,
    selected_manifest_path: Path | None,
    output_path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    ant = load_ant_csv(ant_path)
    models: dict[str, tuple[Any, float]] = {
        "original_baseline": (joblib.load(baseline_model), 0.5)
    }
    if selected_manifest_path is not None:
        manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
        models["selected_candidate"] = (
            joblib.load(selected_manifest_path.parent / manifest["model_file"]),
            float(manifest["decision_threshold"]),
        )
    results = {}
    predictions = ant.loc[:, [*METADATA_COLUMNS, LABEL_SOURCE_COLUMN, TARGET_COLUMN]].copy()
    for name, (model, threshold) in models.items():
        scores = np.asarray(
            positive_probabilities(model, ant.loc[:, FEATURE_COLUMNS]), dtype=float
        )
        results[name] = {
            "at_0_5": evaluate_scores(
                ant[TARGET_COLUMN].to_numpy(dtype=int),
                scores,
                _record_ids(ant),
                threshold=0.5,
                top_fraction=float(settings["top_fraction"]),
                reliability_bins=int(settings["reliability_bins"]),
            ),
            "at_frozen_selected_threshold": evaluate_scores(
                ant[TARGET_COLUMN].to_numpy(dtype=int),
                scores,
                _record_ids(ant),
                threshold=threshold,
                top_fraction=float(settings["top_fraction"]),
                reliability_bins=int(settings["reliability_bins"]),
            ),
        }
        predictions[f"{name}_score"] = scores
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "external diagnostic after model/configuration freeze; Ant is not an untouched final test",
        "ant_file": str(ant_path),
        "ant_sha256": _sha256(ant_path),
        "rows": int(len(ant)),
        "defective_rows": int(ant[TARGET_COLUMN].sum()),
        "results": results,
    }
    _write_json(output_path, payload)
    predictions.to_csv(output_path.with_suffix(".csv"), index=False)
    return payload


def _load_settings(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "package", "external-ant"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ant-csv", type=Path)
    parser.add_argument("--selected-manifest", type=Path)
    args = parser.parse_args()
    settings = _load_settings(args.config)
    if args.command == "run":
        data = load_dataset(args.data_dir, args.manifest)
        result = run_nested_experiment(data, settings, args.output_dir)
        print(json.dumps(result["final_development_selection"], indent=2, default=_json_default))
    elif args.command == "package":
        data = load_dataset(args.data_dir, args.manifest)
        manifest = package_selected_model(
            data,
            settings,
            args.output_dir / "nested_results.json",
            Path("artifacts/deployment"),
        )
        print(json.dumps(manifest, indent=2, default=_json_default))
    else:
        if args.ant_csv is None:
            parser.error("external-ant requires --ant-csv")
        result = evaluate_external_ant(
            args.ant_csv,
            Path("artifacts/baseline/logistic_pipeline.joblib"),
            args.selected_manifest,
            args.output_dir / "ant_external_diagnostic.json",
            settings,
        )
        print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
