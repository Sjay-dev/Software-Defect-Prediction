from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from defect_prediction.data import validate_dataset
from defect_prediction.model_comparison import (
    EXPECTED_CLASSIFIERS,
    MODEL_NAMES,
    build_model_pipeline,
    evaluate_comparison_frame,
    evaluate_scores,
    top_budget_positions,
)
from defect_prediction.modeling import split_by_project
from defect_prediction.schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


def _settings() -> dict:
    settings = json.loads(Path("config/model_comparison.json").read_text(encoding="utf-8"))
    settings["models"]["random_forest"]["n_estimators"] = 5
    settings["models"]["hist_gradient_boosting"]["max_iter"] = 5
    return settings


def test_every_model_uses_the_same_isolated_outer_project_records(sample_data):
    data = validate_dataset(sample_data)
    folds, predictions, _ = evaluate_comparison_frame(data, _settings())

    assert {fold["held_out_project"] for fold in folds} == {"alpha", "beta", "heldout"}
    for fold in folds:
        project = fold["held_out_project"]
        assert project not in fold["training_projects"]
        assert fold["test_releases"] == ["1.0", "2.0"]
        assert fold["sample_count"] == 8
        primary = predictions[
            predictions["held_out_project"].eq(project)
            & predictions["is_primary_seed"]
        ]
        record_sequences = primary.groupby("model")["record_id"].apply(tuple)
        assert len(set(record_sequences)) == 1
        assert set(primary["model"]) == set(MODEL_NAMES)


def test_preprocessing_is_fitted_only_on_training_rows_and_is_model_appropriate(sample_data):
    data = validate_dataset(sample_data)
    train, test = split_by_project(data, "heldout")
    settings = _settings()
    expected_medians = train.loc[:, FEATURE_COLUMNS].median().to_numpy()

    for model_name in ("dummy_prior", "logistic_regression", "random_forest"):
        pipeline = build_model_pipeline(model_name, settings, seed=42)
        pipeline.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])
        np.testing.assert_allclose(
            pipeline.named_steps["imputer"].statistics_, expected_medians
        )
        assert not np.isclose(
            pipeline.named_steps["imputer"].statistics_[0], test["wmc"].median()
        )

    logistic = build_model_pipeline("logistic_regression", settings, seed=42)
    logistic.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])
    imputed_train = logistic.named_steps["imputer"].transform(train.loc[:, FEATURE_COLUMNS])
    np.testing.assert_allclose(
        logistic.named_steps["scaler"].mean_, imputed_train.mean(axis=0)
    )
    boosting = build_model_pipeline("hist_gradient_boosting", settings, seed=42)
    assert "imputer" not in boosting.named_steps
    assert "scaler" not in boosting.named_steps


def test_only_the_twenty_features_reach_each_model(sample_data):
    data = validate_dataset(sample_data)
    train, _ = split_by_project(data, "heldout")
    excluded = {*METADATA_COLUMNS, LABEL_SOURCE_COLUMN, TARGET_COLUMN}

    for model_name in MODEL_NAMES:
        pipeline = build_model_pipeline(model_name, _settings(), seed=42)
        pipeline.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])
        assert tuple(pipeline.feature_names_in_) == FEATURE_COLUMNS
        assert set(pipeline.feature_names_in_).isdisjoint(excluded)
        assert len(pipeline.feature_names_in_) == 20


def test_metrics_and_top_budget_use_ceiling_with_deterministic_ties(sample_data):
    test = validate_dataset(sample_data)
    test = test[test["project"].eq("heldout")].iloc[:5].copy()
    test.loc[:, "release"] = ["2.0", "1.0", "1.0", "2.0", "1.0"]
    test.loc[:, "class_name"] = ["Z", "B", "A", "A", "C"]
    test[TARGET_COLUMN] = np.asarray([0, 0, 1, 1, 0], dtype="int8")
    scores = np.full(5, 0.75)

    selected = top_budget_positions(test, scores, fraction=0.2)
    metrics, predicted, selected_mask = evaluate_scores(
        test, scores, threshold=0.5, top_fraction=0.2
    )

    assert selected.tolist() == [2]
    assert selected_mask.tolist() == [False, False, True, False, False]
    assert metrics["top_20_percent_recall"] == 0.5
    assert metrics["top_20_percent"]["reviewed_records"] == 1
    assert metrics["confusion_matrix"] == {
        "labels": [0, 1],
        "matrix": [[0, 3], [0, 2]],
        "tn": 0,
        "fp": 3,
        "fn": 0,
        "tp": 2,
    }
    np.testing.assert_array_equal(predicted, np.ones(5, dtype=int))


def test_reported_model_names_are_bound_to_the_correct_estimators():
    expected = {
        "dummy_prior": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
    }
    assert EXPECTED_CLASSIFIERS == expected
    for model_name, classifier_type in expected.items():
        pipeline = build_model_pipeline(model_name, _settings(), seed=42)
        assert isinstance(pipeline.named_steps["classifier"], classifier_type)


def test_seed_sensitivity_is_nested_within_projects_not_added_as_folds(sample_data):
    data = validate_dataset(sample_data)
    folds, _, _ = evaluate_comparison_frame(data, _settings())

    assert len(folds) == data["project"].nunique() == 3
    for fold in folds:
        assert set(fold["seed_sensitivity"]["random_forest"]) == {"17", "42", "91"}
