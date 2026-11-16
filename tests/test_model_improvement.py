from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from defect_prediction.data import validate_dataset
from defect_prediction.model_improvement import (
    AdaptiveSkewLog1p,
    ModelConfiguration,
    build_configurations,
    choose_configuration,
    cross_project_predictions,
    evaluate_scores,
    package_selected_model,
)
from defect_prediction.modeling import positive_probabilities
from defect_prediction.schema import FEATURE_COLUMNS, TARGET_COLUMN


def _settings() -> dict:
    return json.loads(Path("config/model_improvement.json").read_text(encoding="utf-8"))


def _selection_result(identifier: str, ap: float, recall: float, complexity: int) -> dict:
    return {
        "configuration": {
            "identifier": identifier,
            "family": identifier,
            "preprocessing": "test",
            "parameters": {},
            "complexity_rank": complexity,
        },
        "summary_at_0_5": {
            "macro": {
                "average_precision": ap,
                "top_20_percent_recall": recall,
            }
        },
    }


def test_bounded_candidate_grid_contains_all_requested_families():
    configurations = build_configurations(_settings())

    assert len(configurations) == 29
    assert {item.family for item in configurations} == {
        "baseline_logistic",
        "tuned_logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert len({item.identifier for item in configurations}) == len(configurations)


def test_adaptive_log_transform_is_learned_only_from_training_values():
    train = np.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [100.0, 4.0]])
    transformer = AdaptiveSkewLog1p(skew_threshold=1.0).fit(train)
    original = np.asarray([[8.0, 5.0]])

    transformed = transformer.transform(original)

    np.testing.assert_array_equal(original, [[8.0, 5.0]])
    np.testing.assert_allclose(transformed[0, 0], np.log1p(8.0))
    np.testing.assert_allclose(transformed[0, 1], 5.0)


def test_grouped_predictions_cover_every_row_without_project_leakage(sample_data):
    data = validate_dataset(sample_data)
    configuration = ModelConfiguration(
        identifier="test-logistic",
        family="tuned_logistic_regression",
        preprocessing="standard",
        parameters={"C": 0.5, "class_weight": None},
        complexity_rank=1,
    )

    predictions = cross_project_predictions(
        data, configuration=configuration, reference=None, random_state=42
    )

    assert predictions.index.tolist() == data.index.tolist()
    assert predictions.groupby("project").size().to_dict() == {
        "alpha": 8,
        "beta": 8,
        "heldout": 8,
    }
    assert predictions["score"].between(0.0, 1.0).all()


def test_selection_uses_ap_then_top_20_recall_then_complexity():
    selection = {
        "average_precision_tolerance": 0.002,
        "top_20_recall_tolerance": 0.01,
    }
    results = [
        _selection_result("complex", 0.700, 0.60, 4),
        _selection_result("simple", 0.699, 0.60, 1),
        _selection_result("better-recall", 0.6985, 0.63, 3),
        _selection_result("outside-ap", 0.697, 0.90, 0),
    ]

    chosen = choose_configuration(results, selection)

    assert chosen["chosen"]["configuration"]["identifier"] == "better-recall"
    assert "outside-ap" not in chosen["ap_eligible_identifiers"]


def test_top_20_metric_uses_ceiling_and_stable_identifier_tie_break():
    result = evaluate_scores(
        np.asarray([0, 1, 0, 1, 1, 0]),
        np.asarray([0.5] * 6),
        np.asarray(["z", "a", "b", "c", "d", "e"]),
        threshold=0.5,
        top_fraction=0.2,
        reliability_bins=2,
    )

    assert result["top_20_percent"]["reviewed_records"] == 2
    assert result["top_20_percent"]["defects_found"] == 1


def test_selected_pipeline_package_has_real_hash_and_prediction_parity(
    tmp_path, sample_data
):
    data = validate_dataset(sample_data)
    configuration = ModelConfiguration(
        identifier="tuned-logistic-test",
        family="tuned_logistic_regression",
        preprocessing="adaptive_log1p_robust",
        parameters={"C": 0.5, "class_weight": None},
        complexity_rank=2,
    )
    frozen = {
        "final_development_selection": {
            "deploy_new_model": True,
            "chosen": {
                "configuration": configuration.as_dict(),
                "selected_threshold": {"selected": {"threshold": 0.35}},
            },
        }
    }
    results_path = tmp_path / "nested_results.json"
    results_path.write_text(json.dumps(frozen), encoding="utf-8")

    manifest = package_selected_model(
        data, _settings(), results_path, tmp_path / "deployment"
    )
    artifact_dir = tmp_path / "deployment" / manifest["model_version"]
    model_path = artifact_dir / manifest["model_file"]
    restored = joblib.load(model_path)
    direct = positive_probabilities(restored, data.loc[:, FEATURE_COLUMNS])

    assert manifest["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert manifest["decision_threshold"] == 0.35
    assert len(np.unique(np.round(direct, 12))) > 1
    assert tuple(restored.feature_names_in_) == FEATURE_COLUMNS
    assert set(restored.classes_) == set(data[TARGET_COLUMN].unique())


def test_retained_baseline_cannot_be_repackaged_as_a_candidate(tmp_path, sample_data):
    data = validate_dataset(sample_data)
    results_path = tmp_path / "nested_results.json"
    results_path.write_text(
        json.dumps(
            {
                "final_development_selection": {
                    "deploy_new_model": False,
                    "decision": "retain_baseline",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="retained the baseline"):
        package_selected_model(data, _settings(), results_path, tmp_path / "deployment")
