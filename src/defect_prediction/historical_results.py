"""Curate frozen evaluation artifacts for the read-only dashboard endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HistoricalResultsError(RuntimeError):
    """Raised when intended historical artifacts are absent or inconsistent."""


def _flagged(metrics: dict[str, Any]) -> int:
    matrix = metrics["confusion_matrix"]
    return int(matrix["fp"] + matrix["tp"])


def load_historical_results(
    lopo_results_path: Path, source_validation_results_path: Path
) -> dict[str, Any]:
    """Join two frozen reports without fitting, tuning, or changing either file."""
    try:
        lopo = json.loads(lopo_results_path.read_text(encoding="utf-8"))
        source_validation = json.loads(
            source_validation_results_path.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise HistoricalResultsError(f"Unable to read historical results: {error}") from error

    lopo_by_project = {fold["held_out_project"]: fold for fold in lopo["folds"]}
    source_by_project = {
        fold["held_out_project"]: fold for fold in source_validation["folds"]
    }
    if set(lopo_by_project) != set(source_by_project) or len(lopo_by_project) != 7:
        raise HistoricalResultsError(
            "Historical reports must contain the same seven target projects."
        )

    projects: list[dict[str, Any]] = []
    nearly_all: list[str] = []
    for project in sorted(lopo_by_project):
        fold = lopo_by_project[project]
        source_fold = source_by_project[project]
        logistic = fold["models"]["logistic_regression"]
        if logistic != source_fold["at_threshold_0_5"]:
            raise HistoricalResultsError(
                f"Fixed-threshold metrics disagree for project '{project}'."
            )
        selected = source_fold["at_source_selected_threshold"]
        flagged_records = _flagged(selected)
        flagged_percentage = 100.0 * flagged_records / fold["sample_count"]
        if flagged_percentage >= 95.0:
            nearly_all.append(project)
        projects.append(
            {
                "project": project,
                "records": fold["sample_count"],
                "defect_prevalence": fold["defect_prevalence"],
                "dummy_prior": fold["models"]["dummy_prior"],
                "logistic_regression": logistic,
                "top_20_percent_recall": source_fold["top_20_percent_budget"][
                    "recall"
                ],
                "top_20_percent_reviewed_records": source_fold[
                    "top_20_percent_budget"
                ]["reviewed_records"],
                "source_selected_threshold": source_fold["selected_threshold"],
                "source_selected_flagged_records": flagged_records,
                "source_selected_flagged_percentage": flagged_percentage,
            }
        )

    lopo_summary = lopo["summary"]
    source_summary = source_validation["summary"]
    return {
        "report_type": "historical_cross_project_evaluation",
        "historical": True,
        "project_count": lopo_summary["project_count"],
        "decision_threshold": lopo["decision_threshold"],
        "tuning": lopo["tuning"],
        "macro_average_note": lopo_summary["macro_average_note"],
        "target_label_usage": source_validation["target_label_usage"],
        "ranking_note": (
            "Average precision and top-20% recall are ranking results. They are "
            "separate from fixed-threshold classification metrics."
        ),
        "nearly_all_definition": "At least 95% of target records flagged.",
        "source_selected_thresholds_flagging_all_or_nearly_all": nearly_all,
        "summary": {
            "dummy_prior": lopo_summary["models"]["dummy_prior"],
            "logistic_regression": lopo_summary["models"]["logistic_regression"],
            "top_20_percent_recall": source_summary["top_20_percent_recall"],
        },
        "projects": projects,
    }
