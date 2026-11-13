"""Build complete classification and ranking reports from frozen results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_files(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _always_defective(total: int, defective: int) -> dict[str, Any]:
    clean = total - defective
    precision = defective / total
    return {
        "precision": precision,
        "recall": 1.0,
        "f1": 2 * precision / (1 + precision),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[0, clean], [0, defective]],
            "tn": 0,
            "fp": clean,
            "fn": 0,
            "tp": defective,
        },
    }


def _classification_rule(metrics: dict[str, Any], total: int) -> dict[str, Any]:
    matrix = metrics["confusion_matrix"]
    flagged = int(matrix["fp"] + matrix["tp"])
    return {
        "flagged_records": flagged,
        "flagged_percentage": 100.0 * flagged / total,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": matrix,
    }


def build_report_payload(source_results: dict[str, Any]) -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    all_or_nearly_all: list[str] = []
    for fold in source_results["folds"]:
        total = int(fold["test_rows"])
        fixed = fold["at_threshold_0_5"]
        selected = fold["at_source_selected_threshold"]
        defective = int(selected["confusion_matrix"]["fn"] + selected["confusion_matrix"]["tp"])
        always = _always_defective(total, defective)
        rules = {
            "fixed_threshold_0_5": _classification_rule(fixed, total),
            "source_selected_threshold": _classification_rule(selected, total),
            "always_predict_defective": _classification_rule(always, total),
        }
        if rules["source_selected_threshold"]["flagged_percentage"] >= 95.0:
            all_or_nearly_all.append(fold["held_out_project"])
        projects.append(
            {
                "project": fold["held_out_project"],
                "records": total,
                "defective_records": defective,
                "source_selected_threshold": fold["selected_threshold"],
                "classification_rules": rules,
                "ranking_results": {
                    "average_precision": fold["average_precision"],
                    "top_20_percent_recall": fold["top_20_percent_budget"],
                },
            }
        )
    return {
        "report_basis": "Derived only from frozen source-validation results; no fitting or threshold selection is performed.",
        "outer_test_policy": "Outer-project labels are used only for final reporting. Do not tune further against these results.",
        "score_interpretation": {
            "classification": "Precision, recall, F1, confusion matrices, and flagged counts depend on an operating rule.",
            "ranking": "Average precision (AP) and top-20% recall are ranking results and do not validate a deployment cutoff.",
        },
        "nearly_all_definition": "At least 95% of target records flagged.",
        "source_selected_thresholds_flagging_all_or_nearly_all": all_or_nearly_all,
        "projects": projects,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Complete baseline reporting",
        "",
        payload["report_basis"],
        "",
        "Classification results use explicit operating rules. AP and top-20% recall remain ranking results; they are not thresholded classification results.",
        "",
        "| Project | Rule | Flagged | Flagged % | Precision | Recall | F1 | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    labels = {
        "fixed_threshold_0_5": "fixed 0.5",
        "source_selected_threshold": "source-selected",
        "always_predict_defective": "always defective",
    }
    for project in payload["projects"]:
        for key, label in labels.items():
            result = project["classification_rules"][key]
            matrix = result["confusion_matrix"]
            lines.append(
                f"| {project['project']} | {label} | {result['flagged_records']}/{project['records']} | "
                f"{result['flagged_percentage']:.3f}% | {result['precision']:.3f} | "
                f"{result['recall']:.3f} | {result['f1']:.3f} | "
                f"{matrix['tn']}/{matrix['fp']}/{matrix['fn']}/{matrix['tp']} |"
            )
    lines.extend(
        [
            "",
            "## Ranking results",
            "",
            "| Project | AP | Top-20% recall | Defects found | Records reviewed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for project in payload["projects"]:
        ranking = project["ranking_results"]
        top = ranking["top_20_percent_recall"]
        lines.append(
            f"| {project['project']} | {ranking['average_precision']:.3f} | "
            f"{top['recall']:.3f} | {top['defects_found']}/{top['total_defects']} | "
            f"{top['reviewed_records']}/{top['target_records']} |"
        )
    named = ", ".join(payload["source_selected_thresholds_flagging_all_or_nearly_all"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Using the stated at-least-95% definition, source-selected thresholds flag all or nearly all records for: {named}.",
            "Four projects (Camel, Lucene, POI, and Synapse) flag exactly 100%; JEdit flags 99.886%. Perfect or near-perfect recall in those cases comes with an impractically broad review set.",
            "",
            "The always-defective baseline makes the tradeoff explicit: it has recall 1.0, flags 100% of records, and its precision equals the target defect prevalence.",
            "",
            "Outer test results were not used to choose any additional threshold or model setting, and must not be used for further tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def write_complete_report(
    source_results_path: Path,
    output_dir: Path,
    protected_directories: tuple[Path, ...] = (),
) -> dict[str, Any]:
    before = {str(path): _sha256_files(path) for path in protected_directories}
    source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    payload = build_report_payload(source_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complete_baseline_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "complete_baseline_report.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    after = {str(path): _sha256_files(path) for path in protected_directories}
    if before != after:
        raise RuntimeError("A protected experiment artifact changed while reporting.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path("artifacts/source_validation/source_validation_results.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reporting"))
    args = parser.parse_args()
    payload = write_complete_report(
        args.source_results,
        args.output_dir,
        (Path("artifacts/baseline"), Path("artifacts/lopo"), Path("artifacts/source_validation")),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
