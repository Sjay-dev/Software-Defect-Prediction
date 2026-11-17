from __future__ import annotations

import json
from pathlib import Path

from defect_prediction.historical_results import (
    load_historical_results,
    load_research_results,
)


def _metrics(ap, precision, recall, f1, *, tn, fp, fn, tp):
    return {
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def test_historical_results_join_frozen_metrics_without_recalculation(tmp_path):
    projects = ["a", "b", "c", "d", "e", "f", "g"]
    dummy = _metrics(0.2, 0.0, 0.0, 0.0, tn=8, fp=0, fn=2, tp=0)
    logistic = _metrics(0.6, 0.5, 0.5, 0.5, tn=7, fp=1, fn=1, tp=1)
    selected = _metrics(0.6, 0.2, 1.0, 1 / 3, tn=0, fp=8, fn=0, tp=2)
    lopo = {
        "decision_threshold": 0.5,
        "tuning": "none",
        "folds": [
            {
                "held_out_project": project,
                "sample_count": 10,
                "defect_prevalence": 0.2,
                "models": {"dummy_prior": dummy, "logistic_regression": logistic},
            }
            for project in projects
        ],
        "summary": {
            "project_count": 7,
            "macro_average_note": "Every project has equal weight.",
            "models": {"dummy_prior": {"average_precision": {"mean": 0.2}}, "logistic_regression": {"average_precision": {"mean": 0.6}}},
        },
    }
    source = {
        "target_label_usage": "final outer evaluation only",
        "folds": [
            {
                "held_out_project": project,
                "at_threshold_0_5": logistic,
                "at_source_selected_threshold": selected,
                "selected_threshold": 0.1,
                "top_20_percent_budget": {"recall": 0.5, "reviewed_records": 2},
            }
            for project in projects
        ],
        "summary": {
            "top_20_percent_recall": {"mean": 0.5},
            "variants": {
                "source_selected_threshold": {
                    "precision": {"mean": 0.2},
                    "recall": {"mean": 1.0},
                    "f1": {"mean": 1 / 3},
                }
            },
        },
    }
    lopo_path = tmp_path / "lopo.json"
    source_path = tmp_path / "source.json"
    lopo_path.write_text(json.dumps(lopo), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")

    result = load_historical_results(lopo_path, source_path)

    assert result["project_count"] == 7
    assert result["projects"][0]["dummy_prior"]["average_precision"] == 0.2
    assert result["projects"][0]["logistic_regression"]["average_precision"] == 0.6
    assert result["projects"][0]["top_20_percent_recall"] == 0.5
    assert result["projects"][0]["source_selected_flagged_percentage"] == 100.0
    assert result["source_selected_thresholds_flagging_all_or_nearly_all"] == projects


def test_compact_research_reports_are_validated_and_cross_checked():
    root = Path(__file__).resolve().parents[1]
    result = load_research_results(
        root / "reports" / "baseline" / "summary.json",
        root / "reports" / "model_comparison" / "summary.json",
        root / "reports" / "source_similarity" / "summary.json",
    )

    assert result["available_sections"] == [
        "baseline",
        "model_comparison",
        "source_similarity",
    ]
    assert result["unavailable_sections"] == []
    assert result["model_comparison"]["summary"]["models"][
        "logistic_regression"
    ]["average_precision"]["mean"] == 0.5541701254966209
    assert result["source_similarity"]["summary"]["strategies"][
        "similar_sources"
    ]["average_precision"]["mean"] == 0.5335447093133792


def test_compact_research_report_failure_is_section_local(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = load_research_results(
        tmp_path / "missing.json",
        root / "reports" / "model_comparison" / "summary.json",
        None,
    )

    assert result["available_sections"] == ["model_comparison"]
    assert result["baseline"] is None
    assert result["model_comparison"] is not None
    assert result["source_similarity"] is None
    assert {item["section"] for item in result["unavailable_sections"]} == {
        "baseline",
        "source_similarity",
    }
