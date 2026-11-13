"""Leakage-safe project splitting, model construction, and metric calculation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
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

from .data import DataValidationError
from .schema import FEATURE_COLUMNS, TARGET_COLUMN


def split_by_project(
    frame: pd.DataFrame, held_out_project: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put all rows/releases from one project in test, with none in training."""
    test_mask = frame["project"].eq(held_out_project)
    if not test_mask.any():
        available = sorted(frame["project"].unique().tolist())
        raise DataValidationError(
            f"Held-out project '{held_out_project}' was not found. Available: {available}"
        )
    train = frame.loc[~test_mask].copy()
    test = frame.loc[test_mask].copy()
    if train.empty:
        raise DataValidationError("No training projects remain after the holdout.")
    overlap = set(train["project"]) & set(test["project"])
    if overlap:
        raise AssertionError(f"Project leakage detected: {sorted(overlap)}")
    if train[list(FEATURE_COLUMNS)].isna().all(axis=0).any():
        raise DataValidationError("A feature is entirely missing in the training projects.")
    if train[TARGET_COLUMN].nunique() != 2:
        raise DataValidationError("Training data must contain defective and clean classes.")
    return train, test


def build_dummy_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )


def build_logistic_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def positive_probabilities(model: ClassifierMixin, features: pd.DataFrame) -> Any:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted classifier has no positive class (1).")
    return probabilities[:, classes.index(1)]


def evaluate_predictions(
    y_true: pd.Series, probabilities: Any, threshold: float
) -> tuple[dict[str, Any], Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    return metrics, predictions

