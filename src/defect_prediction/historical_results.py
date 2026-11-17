"""Curate frozen evaluation artifacts for the read-only dashboard endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HistoricalResultsError(RuntimeError):
    """Raised when intended historical artifacts are absent or inconsistent."""


SECTION_TYPES = {
    "baseline": "historical_cross_project_evaluation",
    "model_comparison": "fixed_model_comparison",
    "source_similarity": "source_project_similarity_selection",
}
MODEL_NAMES = {
    "dummy_prior",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
}
METRIC_NAMES = (
    "average_precision",
    "top_20_percent_recall",
    "precision",
    "recall",
    "f1",
)
CONFUSION_NAMES = ("tn", "fp", "fn", "tp")


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
                "source_selected_metrics": selected,
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
            "source_selected_threshold": source_summary["variants"][
                "source_selected_threshold"
            ],
        },
        "projects": projects,
    }


def _project_names(report: dict[str, Any]) -> set[str]:
    return {str(project["project"]) for project in report["projects"]}


def _validate_common(section: str, report: dict[str, Any]) -> None:
    if report.get("report_type") != SECTION_TYPES[section]:
        raise HistoricalResultsError(
            f"{section} has unexpected report_type {report.get('report_type')!r}."
        )
    if report.get("schema_version") != 1:
        raise HistoricalResultsError(f"{section} must use schema version 1.")
    projects = report.get("projects")
    if not isinstance(projects, list) or len(projects) != 7:
        raise HistoricalResultsError(f"{section} must contain seven projects.")
    names = _project_names(report)
    if len(names) != 7 or report.get("project_count") != 7:
        raise HistoricalResultsError(f"{section} has invalid project identities.")


def _validate_metrics(section: str, project: str, metrics: dict[str, Any]) -> None:
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise HistoricalResultsError(
                f"{section} has invalid {name} for {project}."
            )
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, dict) or any(
        not isinstance(matrix.get(name), int) or matrix[name] < 0
        for name in CONFUSION_NAMES
    ):
        raise HistoricalResultsError(
            f"{section} has an invalid confusion matrix for {project}."
        )


def _validate_baseline(report: dict[str, Any]) -> None:
    _validate_common("baseline", report)
    if report.get("decision_threshold") != 0.5:
        raise HistoricalResultsError("Baseline threshold must be 0.5.")
    for project in report["projects"]:
        name = str(project["project"])
        logistic = dict(project.get("logistic_regression", {}))
        logistic["top_20_percent_recall"] = project.get(
            "top_20_percent_recall"
        )
        _validate_metrics("baseline logistic regression", name, logistic)
        if not isinstance(project.get("source_selected_metrics"), dict):
            raise HistoricalResultsError(
                f"Baseline source-selected metrics are missing for {name}."
            )


def _validate_model_comparison(report: dict[str, Any]) -> None:
    _validate_common("model_comparison", report)
    if set(report.get("models", [])) != MODEL_NAMES:
        raise HistoricalResultsError("Model comparison has an unexpected model set.")
    if report.get("decision_threshold") != 0.5:
        raise HistoricalResultsError("Model comparison threshold must be 0.5.")
    for project in report["projects"]:
        if set(project.get("models", {})) != MODEL_NAMES:
            raise HistoricalResultsError(
                f"Model comparison is incomplete for {project.get('project')}."
            )
        for model, metrics in project["models"].items():
            _validate_metrics(
                f"model comparison {model}", str(project["project"]), metrics
            )


def _validate_similarity(report: dict[str, Any]) -> None:
    _validate_common("source_similarity", report)
    if report.get("decision_threshold") != 0.5:
        raise HistoricalResultsError("Similarity threshold must be 0.5.")
    if report.get("selected_source_count") != 3 or report.get(
        "random_subset_count"
    ) != 20:
        raise HistoricalResultsError("Similarity source/subset counts are invalid.")
    for project in report["projects"]:
        distances = project.get("selected_sources", [])
        selected = [row for row in distances if row.get("selected")]
        random_reference = project.get("random_subset_reference", {})
        if len(distances) != 6 or len(selected) != 3:
            raise HistoricalResultsError(
                f"Similarity distances are incomplete for {project.get('project')}."
            )
        if random_reference.get("combination_count") != 20:
            raise HistoricalResultsError(
                f"Similarity subsets are incomplete for {project.get('project')}."
            )
        for strategy in ("all_sources", "similar_sources"):
            _validate_metrics(
                f"similarity {strategy}",
                str(project["project"]),
                project.get(strategy, {}).get("metrics", {}),
            )


def _metric_equal(left: dict[str, Any], right: dict[str, Any], name: str) -> bool:
    if name == "top_20_percent_recall":
        return left.get(name) == right.get(name)
    return left.get(name) == right.get(name)


def _validate_cross_report_consistency(sections: dict[str, dict[str, Any]]) -> None:
    project_sets = [_project_names(report) for report in sections.values()]
    if project_sets and any(names != project_sets[0] for names in project_sets[1:]):
        raise HistoricalResultsError("Research summaries describe different projects.")
    baseline = sections.get("baseline")
    model = sections.get("model_comparison")
    similarity = sections.get("source_similarity")
    if baseline is not None and model is not None:
        baseline_by_project = {row["project"]: row for row in baseline["projects"]}
        for project in model["projects"]:
            expected = baseline_by_project[project["project"]]
            actual = project["models"]["logistic_regression"]
            for name in ("average_precision", "precision", "recall", "f1"):
                if not _metric_equal(actual, expected["logistic_regression"], name):
                    raise HistoricalResultsError(
                        f"Logistic metrics disagree for {project['project']}."
                    )
            if actual["top_20_percent_recall"] != expected["top_20_percent_recall"]:
                raise HistoricalResultsError(
                    f"Logistic ranking metrics disagree for {project['project']}."
                )
    if model is not None and similarity is not None:
        model_by_project = {row["project"]: row for row in model["projects"]}
        for project in similarity["projects"]:
            expected = model_by_project[project["project"]]["models"][
                "logistic_regression"
            ]
            actual = project["all_sources"]["metrics"]
            for name in METRIC_NAMES:
                if actual[name] != expected[name]:
                    raise HistoricalResultsError(
                        f"Similarity all-source metrics disagree for {project['project']}."
                    )
            if any(
                actual["confusion_matrix"][name]
                != expected["confusion_matrix"][name]
                for name in CONFUSION_NAMES
            ):
                raise HistoricalResultsError(
                    f"Similarity all-source confusion matrix disagrees for {project['project']}."
                )


def load_research_results(
    baseline_summary_path: Path | None,
    model_comparison_summary_path: Path | None,
    source_similarity_summary_path: Path | None,
) -> dict[str, Any]:
    """Load validated compact reports independently so one missing section is non-fatal."""
    paths = {
        "baseline": baseline_summary_path,
        "model_comparison": model_comparison_summary_path,
        "source_similarity": source_similarity_summary_path,
    }
    sections: dict[str, dict[str, Any]] = {}
    unavailable: list[dict[str, str]] = []
    validators = {
        "baseline": _validate_baseline,
        "model_comparison": _validate_model_comparison,
        "source_similarity": _validate_similarity,
    }
    for section, path in paths.items():
        if path is None:
            unavailable.append({"section": section, "reason": "not configured"})
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            validators[section](report)
            sections[section] = report
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, HistoricalResultsError) as error:
            unavailable.append({"section": section, "reason": str(error)})
    try:
        _validate_cross_report_consistency(sections)
    except HistoricalResultsError as error:
        # Inconsistent research data must never be served as if it were valid.
        unavailable.extend(
            {"section": section, "reason": str(error)} for section in sections
        )
        sections = {}
    return {
        "report_type": "consolidated_historical_research",
        "historical": True,
        "available_sections": sorted(sections),
        "unavailable_sections": unavailable,
        "baseline": sections.get("baseline"),
        "model_comparison": sections.get("model_comparison"),
        "source_similarity": sections.get("source_similarity"),
    }
