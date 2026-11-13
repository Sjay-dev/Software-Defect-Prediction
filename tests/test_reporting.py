from __future__ import annotations

from defect_prediction.reporting import build_report_payload


def test_complete_report_adds_always_defective_and_flagged_counts():
    matrix = {
        "labels": [0, 1],
        "matrix": [[8, 2], [1, 4]],
        "tn": 8,
        "fp": 2,
        "fn": 1,
        "tp": 4,
    }
    selected_matrix = {
        "labels": [0, 1],
        "matrix": [[0, 10], [0, 5]],
        "tn": 0,
        "fp": 10,
        "fn": 0,
        "tp": 5,
    }
    source = {
        "folds": [
            {
                "held_out_project": "target",
                "test_rows": 15,
                "selected_threshold": 0.1,
                "average_precision": 0.6,
                "at_threshold_0_5": {
                    "precision": 2 / 3,
                    "recall": 0.8,
                    "f1": 8 / 11,
                    "confusion_matrix": matrix,
                },
                "at_source_selected_threshold": {
                    "precision": 1 / 3,
                    "recall": 1.0,
                    "f1": 0.5,
                    "confusion_matrix": selected_matrix,
                },
                "top_20_percent_budget": {
                    "recall": 0.4,
                    "defects_found": 2,
                    "total_defects": 5,
                    "reviewed_records": 3,
                    "target_records": 15,
                },
            }
        ]
    }

    report = build_report_payload(source)
    project = report["projects"][0]
    always = project["classification_rules"]["always_predict_defective"]

    assert always["flagged_records"] == 15
    assert always["flagged_percentage"] == 100.0
    assert always["precision"] == 1 / 3
    assert always["recall"] == 1.0
    assert always["f1"] == 0.5
    assert always["confusion_matrix"]["matrix"] == [[0, 10], [0, 5]]
    assert project["classification_rules"]["fixed_threshold_0_5"]["flagged_records"] == 6
    assert project["ranking_results"]["average_precision"] == 0.6
    assert report["source_selected_thresholds_flagging_all_or_nearly_all"] == [
        "target"
    ]
