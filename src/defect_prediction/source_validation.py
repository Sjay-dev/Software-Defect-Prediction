"""Nested source-project validation for threshold selection without target leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import load_dataset
from .modeling import (
    build_logistic_pipeline,
    evaluate_predictions,
    positive_probabilities,
    split_by_project,
)
from .schema import FEATURE_COLUMNS, TARGET_COLUMN


THRESHOLD_GRID = tuple(index / 20 for index in range(1, 20))
ORIGINAL_THRESHOLD = 0.5
REVIEW_FRACTION = 0.20


def _sha256_files(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def choose_threshold(mean_f1_by_threshold: dict[float, float]) -> float:
    """Apply the predeclared deterministic tie-break to threshold scores."""
    if not mean_f1_by_threshold:
        raise ValueError("At least one threshold score is required.")
    best_score = max(mean_f1_by_threshold.values())
    tied = [
        threshold
        for threshold, score in mean_f1_by_threshold.items()
        if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
    ]
    return min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))


def select_threshold_from_project_predictions(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_grid: Iterable[float] = THRESHOLD_GRID,
) -> dict[str, Any]:
    """Select on macro project F1, giving every source project equal weight."""
    if not predictions:
        raise ValueError("Inner validation predictions are required.")
    score_table: list[dict[str, Any]] = []
    mean_scores: dict[float, float] = {}
    for threshold in threshold_grid:
        project_scores = {
            project: float(
                f1_score(
                    labels,
                    (probabilities >= threshold).astype(int),
                    zero_division=0,
                )
            )
            for project, (labels, probabilities) in predictions.items()
        }
        mean_score = float(statistics.mean(project_scores.values()))
        mean_scores[threshold] = mean_score
        score_table.append(
            {
                "threshold": threshold,
                "mean_project_f1": mean_score,
                "project_f1": project_scores,
            }
        )
    selected = choose_threshold(mean_scores)
    return {
        "selected_threshold": selected,
        "selected_mean_project_f1": mean_scores[selected],
        "selection_rule": (
            "Maximize unweighted mean per-project F1 over thresholds 0.05 to 0.95 "
            "in steps of 0.05; ties choose the threshold closest to 0.5, then the "
            "lower threshold."
        ),
        "threshold_scores": score_table,
    }


def select_threshold_for_outer_project(
    data: pd.DataFrame, outer_project: str, *, random_state: int
) -> dict[str, Any]:
    """Use only the six source projects to select an outer-fold threshold."""
    source, _ = split_by_project(data, outer_project)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    inner_folds: list[dict[str, Any]] = []
    for inner_project in sorted(source["project"].unique()):
        inner_train, inner_validation = split_by_project(source, inner_project)
        model = build_logistic_pipeline(random_state)
        model.fit(
            inner_train.loc[:, FEATURE_COLUMNS], inner_train[TARGET_COLUMN]
        )
        probabilities = positive_probabilities(
            model, inner_validation.loc[:, FEATURE_COLUMNS]
        )
        predictions[inner_project] = (
            inner_validation[TARGET_COLUMN].to_numpy(),
            probabilities,
        )
        inner_folds.append(
            {
                "validation_project": inner_project,
                "validation_releases": sorted(
                    inner_validation["release"].unique().tolist()
                ),
                "training_projects": sorted(
                    inner_train["project"].unique().tolist()
                ),
                "training_rows": int(len(inner_train)),
                "validation_rows": int(len(inner_validation)),
            }
        )
    selection = select_threshold_from_project_predictions(predictions)
    selection["outer_project_excluded"] = outer_project
    selection["inner_folds"] = inner_folds
    return selection


def top_budget_indices(
    target: pd.DataFrame, probabilities: np.ndarray, fraction: float = REVIEW_FRACTION
) -> list[Any]:
    """Rank deterministically and return exactly ceil(fraction * rows) indices."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if len(target) != len(probabilities):
        raise ValueError("Target rows and probabilities must have the same length.")
    ranking = target.loc[:, ["release", "class_name"]].copy()
    ranking["probability"] = probabilities
    ranking["original_order"] = np.arange(len(ranking))
    ordered = ranking.sort_values(
        ["probability", "release", "class_name", "original_order"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    budget = math.ceil(fraction * len(target))
    return ordered.head(budget).index.tolist()


def top_budget_recall(
    target: pd.DataFrame, probabilities: np.ndarray, fraction: float = REVIEW_FRACTION
) -> dict[str, Any]:
    selected_indices = top_budget_indices(target, probabilities, fraction)
    total_defects = int(target[TARGET_COLUMN].sum())
    found = int(target.loc[selected_indices, TARGET_COLUMN].sum())
    return {
        "review_fraction": fraction,
        "reviewed_records": len(selected_indices),
        "target_records": int(len(target)),
        "total_defects": total_defects,
        "defects_found": found,
        "recall": float(found / total_defects) if total_defects else 0.0,
        "ranking_rule": (
            "Descending logistic probability; ties use release, class_name, then "
            "original row order ascending. The record budget is ceil(0.20 * rows)."
        ),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_standard_deviation": float(statistics.stdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _comparison_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fixed = fold["at_threshold_0_5"]
        selected = fold["at_source_selected_threshold"]
        fixed_matrix = fixed["confusion_matrix"]
        selected_matrix = selected["confusion_matrix"]
        rows.append(
            {
                "held_out_project": fold["held_out_project"],
                "test_rows": fold["test_rows"],
                "defect_prevalence": fold["defect_prevalence"],
                "selected_threshold": fold["selected_threshold"],
                "average_precision": fold["average_precision"],
                "precision_at_0_5": fixed["precision"],
                "recall_at_0_5": fixed["recall"],
                "f1_at_0_5": fixed["f1"],
                **{f"{key}_at_0_5": fixed_matrix[key] for key in ("tn", "fp", "fn", "tp")},
                "precision_selected": selected["precision"],
                "recall_selected": selected["recall"],
                "f1_selected": selected["f1"],
                **{f"{key}_selected": selected_matrix[key] for key in ("tn", "fp", "fn", "tp")},
                "top_20_percent_recall": fold["top_20_percent_budget"]["recall"],
                "top_20_percent_records": fold["top_20_percent_budget"]["reviewed_records"],
                "top_20_percent_defects_found": fold["top_20_percent_budget"]["defects_found"],
            }
        )
    return rows


def _summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    variants = {
        "threshold_0_5": "at_threshold_0_5",
        "source_selected_threshold": "at_source_selected_threshold",
    }
    metrics = ("precision", "recall", "f1")
    summary: dict[str, Any] = {
        "project_count": len(folds),
        "average_precision": _distribution(
            [fold["average_precision"] for fold in folds]
        ),
        "selected_threshold": _distribution(
            [fold["selected_threshold"] for fold in folds]
        ),
        "top_20_percent_recall": _distribution(
            [fold["top_20_percent_budget"]["recall"] for fold in folds]
        ),
        "variants": {},
        "change_counts": {},
    }
    for label, key in variants.items():
        summary["variants"][label] = {
            metric: _distribution([fold[key][metric] for fold in folds])
            for metric in metrics
        }
    for metric in metrics:
        deltas = [
            fold["at_source_selected_threshold"][metric]
            - fold["at_threshold_0_5"][metric]
            for fold in folds
        ]
        summary["change_counts"][metric] = {
            "improved_projects": sum(delta > 1e-12 for delta in deltas),
            "regressed_projects": sum(delta < -1e-12 for delta in deltas),
            "unchanged_projects": sum(abs(delta) <= 1e-12 for delta in deltas),
            "mean_change": float(statistics.mean(deltas)),
        }
    return summary


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Source-validation threshold experiment",
        "",
        "Existing `artifacts/baseline` and `artifacts/lopo` files were SHA-256 hashed "
        f"before and after this experiment and remained unchanged: "
        f"**{payload['artifact_integrity']['unchanged']}**.",
        "",
        "For each outer project, every release stayed untouched while six fresh inner "
        "leave-one-project-out pipelines produced source-validation probabilities. The fixed "
        "grid was 0.05 through 0.95 in steps of 0.05. The chosen threshold maximizes mean "
        "per-project F1; ties choose the value closest to 0.5, then the lower value.",
        "",
        "## Outer-project results",
        "",
        "AP is listed once because both decisions use the same probabilities; only the "
        "classification threshold changes. Confusion matrices are TN / FP / FN / TP.",
        "",
        "| Project | Rows | Prev. | Selected t | AP | P/R/F1 at 0.5 | CM at 0.5 | P/R/F1 selected | CM selected | Top-20% recall (found/total; reviewed) |",
        "|---|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    for fold in payload["folds"]:
        fixed = fold["at_threshold_0_5"]
        selected = fold["at_source_selected_threshold"]
        fm = fixed["confusion_matrix"]
        sm = selected["confusion_matrix"]
        lines.append(
            f"| {fold['held_out_project']} | {fold['test_rows']} | "
            f"{fold['defect_prevalence']:.3f} | {fold['selected_threshold']:.2f} | "
            f"{fold['average_precision']:.3f} | "
            f"{fixed['precision']:.3f}/{fixed['recall']:.3f}/{fixed['f1']:.3f} | "
            f"{fm['tn']}/{fm['fp']}/{fm['fn']}/{fm['tp']} | "
            f"{selected['precision']:.3f}/{selected['recall']:.3f}/{selected['f1']:.3f} | "
            f"{sm['tn']}/{sm['fp']}/{sm['fn']}/{sm['tp']} | "
            f"{fold['top_20_percent_budget']['recall']:.3f} "
            f"({fold['top_20_percent_budget']['defects_found']}/"
            f"{fold['top_20_percent_budget']['total_defects']}; "
            f"n={fold['top_20_percent_budget']['reviewed_records']}) |"
        )
    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Across-project mean and variation",
            "",
            "Values are unweighted project means +/- sample standard deviations.",
            "",
            "| Decision rule | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in ("threshold_0_5", "source_selected_threshold"):
        values = summary["variants"][label]
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{values[metric]['mean']:.3f} +/- {values[metric]['sample_standard_deviation']:.3f}"
                for metric in ("precision", "recall", "f1")
            )
            + " |"
        )
    changes = summary["change_counts"]
    f1_improved = [
        fold["held_out_project"]
        for fold in payload["folds"]
        if fold["at_source_selected_threshold"]["f1"]
        > fold["at_threshold_0_5"]["f1"] + 1e-12
    ]
    f1_regressed = [
        fold["held_out_project"]
        for fold in payload["folds"]
        if fold["at_source_selected_threshold"]["f1"]
        < fold["at_threshold_0_5"]["f1"] - 1e-12
    ]
    all_positive = [
        fold["held_out_project"]
        for fold in payload["folds"]
        if fold["at_source_selected_threshold"]["confusion_matrix"]["tn"] == 0
        and fold["at_source_selected_threshold"]["confusion_matrix"]["fn"] == 0
    ]
    lines.extend(
        [
            "",
            f"Mean AP is {summary['average_precision']['mean']:.3f} +/- "
            f"{summary['average_precision']['sample_standard_deviation']:.3f} and is "
            "identical for both threshold rules in every project.",
            f"Recall improves on {changes['recall']['improved_projects']} projects, regresses "
            f"on {changes['recall']['regressed_projects']}, and is unchanged on "
            f"{changes['recall']['unchanged_projects']}.",
            f"F1 improves on {changes['f1']['improved_projects']} projects, regresses on "
            f"{changes['f1']['regressed_projects']}, and is unchanged on "
            f"{changes['f1']['unchanged_projects']}.",
            f"The F1 improvements are: {', '.join(f1_improved)}. The F1 regressions are: "
            f"{', '.join(f1_regressed) or 'none'}.",
            f"At the selected cutoff, {', '.join(all_positive)} classify every target row "
            "as defective (TN=0 and FN=0). That achieves perfect recall by accepting every "
            "false alarm, so it is not a useful operating policy on its own.",
            f"Reviewing the top ceil(20%) scored records captures mean recall "
            f"{summary['top_20_percent_recall']['mean']:.3f} +/- "
            f"{summary['top_20_percent_recall']['sample_standard_deviation']:.3f}. This is "
            "a record-count budget, not a developer-time estimate.",
            "",
            "Lower selected thresholds usually flag more defects but also create more false "
            "alarms. These results describe a source-validated research experiment, not a "
            "production-ready operating policy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_source_validation_experiment(
    data_dir: Path,
    manifest_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    baseline_dir: Path,
    lopo_dir: Path,
) -> dict[str, Any]:
    protected_before = {
        "baseline": _sha256_files(baseline_dir),
        "lopo": _sha256_files(lopo_dir),
    }
    settings = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    if settings["decision_threshold"] != ORIGINAL_THRESHOLD:
        raise ValueError("This experiment requires the preserved original threshold 0.5.")
    data = load_dataset(data_dir, manifest_path)
    preserved_lopo = json.loads(
        (lopo_dir / "lopo_results.json").read_text(encoding="utf-8")
    )
    preserved_by_project = {
        fold["held_out_project"]: fold for fold in preserved_lopo["folds"]
    }

    folds: list[dict[str, Any]] = []
    for outer_project in sorted(data["project"].unique()):
        selection = select_threshold_for_outer_project(
            data, outer_project, random_state=settings["random_state"]
        )
        source, target = split_by_project(data, outer_project)
        model = build_logistic_pipeline(settings["random_state"])
        model.fit(source.loc[:, FEATURE_COLUMNS], source[TARGET_COLUMN])
        probabilities = positive_probabilities(
            model, target.loc[:, FEATURE_COLUMNS]
        )
        fixed_metrics, _ = evaluate_predictions(
            target[TARGET_COLUMN], probabilities, ORIGINAL_THRESHOLD
        )
        selected_metrics, _ = evaluate_predictions(
            target[TARGET_COLUMN],
            probabilities,
            selection["selected_threshold"],
        )
        if not math.isclose(
            fixed_metrics["average_precision"],
            selected_metrics["average_precision"],
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AssertionError("Average precision changed when only threshold changed.")
        preserved_metrics = preserved_by_project[outer_project]["models"][
            "logistic_regression"
        ]
        folds.append(
            {
                "held_out_project": outer_project,
                "outer_releases": sorted(target["release"].unique().tolist()),
                "source_projects": sorted(source["project"].unique().tolist()),
                "test_rows": int(len(target)),
                "defect_prevalence": float(target[TARGET_COLUMN].mean()),
                "selected_threshold": selection["selected_threshold"],
                "source_validation_mean_f1": selection[
                    "selected_mean_project_f1"
                ],
                "average_precision": fixed_metrics["average_precision"],
                "average_precision_unchanged": (
                    fixed_metrics["average_precision"]
                    == selected_metrics["average_precision"]
                ),
                "at_threshold_0_5": fixed_metrics,
                "at_source_selected_threshold": selected_metrics,
                "top_20_percent_budget": top_budget_recall(target, probabilities),
                "inner_validation": selection,
                "fixed_threshold_matches_preserved_lopo": (
                    fixed_metrics == preserved_metrics
                ),
            }
        )

    protected_after = {
        "baseline": _sha256_files(baseline_dir),
        "lopo": _sha256_files(lopo_dir),
    }
    if protected_before != protected_after:
        raise RuntimeError("A protected baseline or LOPO artifact changed.")
    summary = _summarize(folds)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "nested source-project validation for threshold selection",
        "random_state": settings["random_state"],
        "original_threshold": ORIGINAL_THRESHOLD,
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_selection_rule": (
            "Maximize unweighted mean per-project F1 across inner source-project "
            "validation folds; ties choose closest to 0.5, then lower."
        ),
        "target_label_usage": "final outer evaluation only",
        "features": list(FEATURE_COLUMNS),
        "folds": folds,
        "summary": summary,
        "artifact_integrity": {
            "unchanged": protected_before == protected_after,
            "sha256_before": protected_before,
            "sha256_after": protected_after,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_validation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(_comparison_rows(folds)).to_csv(
        output_dir / "threshold_comparison.csv", index=False
    )
    (output_dir / "report.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/source_validation")
    )
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--lopo-dir", type=Path, default=Path("artifacts/lopo"))
    args = parser.parse_args()
    payload = run_source_validation_experiment(
        args.data_dir,
        args.manifest,
        args.config,
        args.output_dir,
        args.baseline_dir,
        args.lopo_dir,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
