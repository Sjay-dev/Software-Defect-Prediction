from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defect_prediction.data import validate_dataset
from defect_prediction.lopo import evaluate_lopo_frame
from defect_prediction.modeling import split_by_project
from defect_prediction.schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)
from defect_prediction.source_similarity import (
    evaluate_outer_target,
    evaluate_source_set,
    fit_prediction_pipeline,
    fit_similarity_preprocessor,
    select_similar_sources,
    select_sources_for_outer,
    source_combinations,
)


def _settings() -> dict:
    return json.loads(Path("config/source_similarity.json").read_text(encoding="utf-8"))


@pytest.fixture
def seven_project_data(sample_data) -> pd.DataFrame:
    template = sample_data[sample_data["project"].eq("alpha")].copy()
    frames = []
    for project_index, project in enumerate(("a", "b", "c", "d", "e", "f", "target")):
        frame = template.copy()
        frame["project"] = project
        frame["class_name"] = [
            f"{project}.Class{index}" for index in range(len(frame))
        ]
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            if feature == "wmc":
                frame[feature] = frame[feature] + project_index
            else:
                frame[feature] = frame[feature] + project_index * (feature_index + 1) / 10
        frames.append(frame)
    return validate_dataset(pd.concat(frames, ignore_index=True))


def test_target_labels_cannot_affect_similarity_selection(seven_project_data):
    original = select_sources_for_outer(seven_project_data, "target", count=3)
    changed = seven_project_data.copy()
    target_mask = changed["project"].eq("target")
    changed.loc[target_mask, TARGET_COLUMN] = 1 - changed.loc[target_mask, TARGET_COLUMN]
    changed.loc[target_mask, LABEL_SOURCE_COLUMN] = changed.loc[target_mask, TARGET_COLUMN]

    repeated = select_sources_for_outer(changed, "target", count=3)

    assert original["selected_source_projects"] == repeated["selected_source_projects"]
    assert original["distances"] == repeated["distances"]
    assert not original["preprocessing_audit"]["target_labels_accessed"]


def test_similarity_preprocessing_is_fitted_on_sources_only(seven_project_data):
    source, target = split_by_project(seven_project_data, "target")
    source_features = source.loc[:, FEATURE_COLUMNS]
    preprocessor = fit_similarity_preprocessor(source_features)
    expected_medians = source_features.median().fillna(0.0).to_numpy()

    np.testing.assert_allclose(
        preprocessor.named_steps["imputer"].statistics_, expected_medians
    )
    imputed_source = preprocessor.named_steps["imputer"].transform(source_features)
    np.testing.assert_allclose(
        preprocessor.named_steps["scaler"].mean_, imputed_source.mean(axis=0)
    )

    extreme_target = target.loc[:, FEATURE_COLUMNS].copy()
    extreme_target.loc[:, "loc"] = 1e12
    first = select_similar_sources(
        source_features, source["project"], target.loc[:, FEATURE_COLUMNS], "target", count=3
    )
    second = select_similar_sources(
        source_features, source["project"], extreme_target, "target", count=3
    )
    assert first["preprocessing_audit"] == second["preprocessing_audit"]


def test_all_missing_and_constant_similarity_features_are_retained():
    rows = 12
    source = pd.DataFrame(
        {
            feature: np.arange(rows, dtype=float) + index
            for index, feature in enumerate(FEATURE_COLUMNS)
        }
    )
    source["wmc"] = np.nan
    source["dit"] = 2.0
    projects = pd.Series(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    target = source.iloc[:4].copy()
    target["wmc"] = [1.0, np.nan, 3.0, 4.0]

    result = select_similar_sources(
        source, projects, target, "target", count=2
    )

    audit = result["preprocessing_audit"]
    assert audit["summary_value_count"] == 40
    assert "wmc" in audit["source_all_missing_features"]
    assert {"wmc", "dit"}.issubset(
        audit["source_constant_after_imputation_features"]
    )
    assert all(np.isfinite(row["distance"]) for row in result["distances"])


def test_prediction_preprocessing_is_fitted_on_selected_sources_only(seven_project_data):
    source, target = split_by_project(seven_project_data, "target")
    selected = source[source["project"].isin(["a", "b", "c"])].copy()
    pipeline, _ = fit_prediction_pipeline(selected, random_state=42)

    expected = selected.loc[:, FEATURE_COLUMNS].median().to_numpy()
    np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, expected)
    assert not np.allclose(
        pipeline.named_steps["imputer"].statistics_,
        source.loc[:, FEATURE_COLUMNS].median().to_numpy(),
    )
    imputed_selected = pipeline.named_steps["imputer"].transform(
        selected.loc[:, FEATURE_COLUMNS]
    )
    np.testing.assert_allclose(
        pipeline.named_steps["scaler"].mean_, imputed_selected.mean(axis=0)
    )
    assert set(selected["project"]).isdisjoint(target["project"])


def test_identifiers_and_labels_are_excluded_from_both_preprocessing_paths(
    seven_project_data,
):
    source, _ = split_by_project(seven_project_data, "target")
    excluded = {*METADATA_COLUMNS, LABEL_SOURCE_COLUMN, TARGET_COLUMN}
    similarity = fit_similarity_preprocessor(source.loc[:, FEATURE_COLUMNS])
    prediction, _ = fit_prediction_pipeline(source, random_state=42)

    assert tuple(similarity.feature_names_in_) == FEATURE_COLUMNS
    assert tuple(prediction.feature_names_in_) == FEATURE_COLUMNS
    assert set(similarity.feature_names_in_).isdisjoint(excluded)
    assert set(prediction.feature_names_in_).isdisjoint(excluded)


def test_selection_and_distance_ties_are_deterministic():
    project_names = ["delta", "alpha", "charlie", "bravo"]
    source = pd.DataFrame(
        np.tile(np.arange(len(FEATURE_COLUMNS), dtype=float), (8, 1)),
        columns=FEATURE_COLUMNS,
    )
    projects = pd.Series(
        [name for name in project_names for _ in range(2)], dtype="string"
    )
    target = source.iloc[:2].copy()

    first = select_similar_sources(source, projects, target, "target", count=3)
    second = select_similar_sources(source, projects, target, "target", count=3)

    assert first == second
    assert first["selected_source_projects"] == ["alpha", "bravo", "charlie"]
    assert [row["source_project"] for row in first["distances"]] == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
    ]


def test_every_unique_three_project_combination_is_evaluated(seven_project_data):
    combinations = source_combinations(["a", "b", "c", "d", "e", "f"], 3)
    assert len(combinations) == len(set(combinations)) == 20
    assert combinations[0] == ("a", "b", "c")
    assert combinations[-1] == ("d", "e", "f")

    fold, predictions = evaluate_outer_target(
        seven_project_data, "target", _settings()
    )
    assert len(fold["random_subsets"]) == 20
    assert len({row["subset_id"] for row in fold["random_subsets"]}) == 20
    assert fold["random_subset_summary"]["combination_count"] == 20
    assert set(fold["similar_sources"]["source_projects"]).isdisjoint({"target"})
    assert set(predictions["strategy"]) == {
        "all_sources",
        "similar_sources",
        "random_subset_reference",
    }


def test_target_overlap_is_rejected_from_prediction_training(seven_project_data):
    _, target = split_by_project(seven_project_data, "target")
    with pytest.raises(AssertionError, match="Target rows entered"):
        evaluate_source_set(
            target,
            target,
            _settings(),
            strategy="invalid",
            subset_id="invalid",
        )


def test_all_source_results_reproduce_existing_logistic_lopo(seven_project_data):
    settings = _settings()
    existing = evaluate_lopo_frame(seven_project_data, random_state=42, threshold=0.5)
    for existing_fold in existing:
        target_project = existing_fold["held_out_project"]
        source, target = split_by_project(seven_project_data, target_project)
        result, _ = evaluate_source_set(
            source,
            target,
            settings,
            strategy="all_sources",
            subset_id="all_six_sources",
        )
        expected = existing_fold["models"]["logistic_regression"]
        for metric in ("average_precision", "precision", "recall", "f1"):
            assert result["metrics"][metric] == expected[metric]
        assert result["metrics"]["confusion_matrix"] == expected["confusion_matrix"]
