"""Build compact tracked research summaries and the consolidated final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .historical_results import load_historical_results
from .schema import FEATURE_COLUMNS, FEATURE_DEFINITIONS


MODEL_NAMES = (
    "dummy_prior",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)
METRIC_NAMES = (
    "average_precision",
    "top_20_percent_recall",
    "precision",
    "recall",
    "f1",
)
CONFUSION_NAMES = ("tn", "fp", "fn", "tp")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {name: metrics[name] for name in METRIC_NAMES if name in metrics}
    result["confusion_matrix"] = {
        name: metrics["confusion_matrix"][name] for name in CONFUSION_NAMES
    }
    return result


def build_baseline_summary(
    lopo_path: Path, source_validation_path: Path
) -> dict[str, Any]:
    payload = load_historical_results(lopo_path, source_validation_path)
    payload["schema_version"] = 1
    payload["source_artifacts"] = {
        "lopo_sha256": _sha256(lopo_path),
        "source_validation_sha256": _sha256(source_validation_path),
    }
    return payload


def build_model_comparison_summary(
    full: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    for fold in full["folds"]:
        projects.append(
            {
                "project": fold["held_out_project"],
                "records": fold["sample_count"],
                "defective_records": fold["defective_count"],
                "defect_prevalence": fold["defect_prevalence"],
                "models": {
                    model: _metrics(fold["models"][model]) for model in MODEL_NAMES
                },
            }
        )
    return {
        "schema_version": 1,
        "report_type": "fixed_model_comparison",
        "historical": True,
        "experiment_name": full["experiment_name"],
        "project_count": len(projects),
        "decision_threshold": full["configuration"]["decision_threshold"],
        "top_fraction": full["configuration"]["top_fraction"],
        "primary_seed": full["configuration"]["primary_seed"],
        "models": list(MODEL_NAMES),
        "summary": full["summary"],
        "seed_variation": [
            row
            for row in full["seed_variation"]
            if row["scope"] == "macro_project_mean_across_seeds"
        ],
        "projects": projects,
        "ranking_note": (
            "Average precision and top-20% recall are ranking metrics. "
            "Classification metrics use the fixed threshold 0.5."
        ),
        "finding": (
            "Logistic regression has the highest equal-project macro average "
            "precision among these fixed configurations."
        ),
        "source_artifact_sha256": _sha256(source_path),
        "configuration_file": "config/model_comparison.json",
        "protocol_file": "docs/MODEL_COMPARISON_PROTOCOL.md",
    }


def build_similarity_summary(
    full: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    for fold in full["folds"]:
        projects.append(
            {
                "project": fold["target_project"],
                "records": fold["target_rows"],
                "defective_records": fold["target_defective_rows"],
                "defect_prevalence": fold["target_defect_prevalence"],
                "selected_sources": fold["similarity_selection"]["distances"],
                "all_sources": {
                    "source_projects": fold["all_sources"]["source_projects"],
                    "training_rows": fold["all_sources"]["training_rows"],
                    "training_defect_prevalence": fold["all_sources"][
                        "training_defect_prevalence"
                    ],
                    "metrics": _metrics(fold["all_sources"]["metrics"]),
                },
                "similar_sources": {
                    "source_projects": fold["similar_sources"]["source_projects"],
                    "training_rows": fold["similar_sources"]["training_rows"],
                    "training_defect_prevalence": fold["similar_sources"][
                        "training_defect_prevalence"
                    ],
                    "metrics": _metrics(fold["similar_sources"]["metrics"]),
                },
                "random_subset_reference": fold["random_subset_summary"],
                "differences": fold["differences_from_all_sources"],
            }
        )
    return {
        "schema_version": 1,
        "report_type": "source_project_similarity_selection",
        "historical": True,
        "experiment_name": full["experiment_name"],
        "project_count": len(projects),
        "decision_threshold": full["configuration"]["decision_threshold"],
        "top_fraction": full["configuration"]["top_fraction"],
        "selected_source_count": full["configuration"]["selected_source_count"],
        "random_subset_count": full["configuration"]["random_subset_reference"][
            "expected_combinations_per_target"
        ],
        "summary": full["summary"],
        "projects": projects,
        "unlabelled_target_access": full["unlabelled_target_access"],
        "ranking_note": (
            "Average precision and top-20% recall are ranking metrics. "
            "Higher F1 at threshold 0.5 does not establish better ranking."
        ),
        "finding": "Similarity selection did not improve ranking overall.",
        "source_artifact_sha256": _sha256(source_path),
        "configuration_file": "config/source_similarity.json",
        "protocol_file": "docs/SOURCE_SIMILARITY_PROTOCOL.md",
    }


def _model_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for project in summary["projects"]:
        for model, metrics in project["models"].items():
            rows.append(
                {
                    "project": project["project"],
                    "records": project["records"],
                    "defect_prevalence": project["defect_prevalence"],
                    "model": model,
                    **{name: metrics[name] for name in METRIC_NAMES},
                    **metrics["confusion_matrix"],
                }
            )
    return rows


def _expected_dummy_top_recall(model_full: dict[str, Any]) -> float:
    expected = [
        math.ceil(
            model_full["configuration"]["top_fraction"] * fold["sample_count"]
        )
        / fold["sample_count"]
        for fold in model_full["folds"]
    ]
    return sum(expected) / len(expected)


def _feature_table() -> list[str]:
    rows = ["| Feature | Definition |", "|---|---|"]
    rows.extend(
        f"| `{name}` | {FEATURE_DEFINITIONS[name]} |" for name in FEATURE_COLUMNS
    )
    return rows


def _final_report(
    baseline: dict[str, Any],
    model: dict[str, Any],
    similarity: dict[str, Any],
    model_full: dict[str, Any],
) -> str:
    baseline_logistic = baseline["summary"]["logistic_regression"]
    source_selected = baseline["summary"]["source_selected_threshold"]
    model_summary = model["summary"]["models"]
    similarity_summary = similarity["summary"]
    sim_diff = similarity_summary["differences_from_all_sources"][
        "similar_minus_all"
    ]
    sim_random_diff = similarity_summary["differences_from_all_sources"][
        "similar_minus_random_mean"
    ]
    dummy_observed = model_summary["dummy_prior"]["top_20_percent_recall"]["mean"]
    dummy_expected = _expected_dummy_top_recall(model_full)

    lines = [
        "# Software Defect Prediction: consolidated research report",
        "",
        "Generated from validated full-precision experiment files. Displayed values are rounded only after calculations.",
        "",
        "## Executive findings",
        "",
        "- Logistic regression had the highest macro average precision (AP) among the tested fixed model configurations.",
        "- Similarity selection did not improve ranking overall, either against all six sources or against the mean of all 20 size-matched three-project subsets.",
        "- Higher F1 at threshold 0.5 does not establish improved ranking; AP and top-20% recall are the primary ranking measures.",
        "- Similarity selection accessed unlabelled target metrics. Target labels were excluded from selection and used only for final evaluation.",
        "- These are exploratory comparisons on seven historical Java projects. No novelty, statistical-significance, or production-readiness claim is established.",
        "",
        "## Research questions",
        "",
        "1. How well does the fixed logistic baseline rank defective classes under seven-project leave-one-project-out evaluation?",
        "2. Do fixed random-forest or histogram-gradient-boosting configurations improve ranking over logistic regression and a dummy-prior reference?",
        "3. Does training on the three projects with the closest unlabelled metric distributions improve ranking over all six sources or the exhaustive three-source reference?",
        "4. How do fixed-threshold classification results differ from ranking results, including the earlier source-validated threshold experiment?",
        "",
        "## Dataset, features, and label",
        "",
        "The experiments use 8,416 Java classes from 25 releases of Camel, JEdit, Log4j, Lucene, POI, Synapse, and Velocity. Files are pinned to immutable Zenodo records with checksums in [`config/data_manifest.json`](../config/data_manifest.json); provenance, CC BY 4.0 terms, and release links are in [`docs/DATASET.md`](../docs/DATASET.md). One row represents one class in one release.",
        "",
        "The source `bug` count is normalized as `bug_count`; the binary target is `defective = 1` when `bug_count > 0`, otherwise 0. Project, release, class name, bug count, and the derived target are never predictive features.",
        "",
        *_feature_table(),
        "",
        "## Evaluation design",
        "",
        "Every outer fold holds out all releases of one project and trains on the other projects. Pipelines are fitted fresh in each fold; imputation and scaling learn only from prediction-training rows. AP and top-20% recall are primary. Top-20% recall uses exactly `ceil(0.20 × target rows)` records, ranking by score descending and breaking ties by release, class name, then original order. Precision, recall, F1, and confusion matrices use the explicitly fixed threshold 0.5 unless a source-selected threshold is named.",
        "",
        "Project-level results receive equal weight. Random three-project subsets are averaged within each target before averaging the seven targets; the 20 subsets are not treated as independent projects.",
        "",
        "## Original baseline and source-validation threshold experiment",
        "",
        f"The fixed logistic model achieved macro AP **{baseline_logistic['average_precision']['mean']:.3f}**, precision **{baseline_logistic['precision']['mean']:.3f}**, recall **{baseline_logistic['recall']['mean']:.3f}**, and F1 **{baseline_logistic['f1']['mean']:.3f}** at threshold 0.5. Mean top-20% recall was **{baseline['summary']['top_20_percent_recall']['mean']:.3f}**.",
        "",
        f"Thresholds selected only from source-project validation increased macro recall to **{source_selected['recall']['mean']:.3f}** and F1 to **{source_selected['f1']['mean']:.3f}**, while precision fell to **{source_selected['precision']['mean']:.3f}**. Camel, JEdit, Lucene, POI, and Synapse flagged all or at least 95% of target records. Because the ranking scores did not change, AP was unchanged; the higher F1 describes a different operating cutoff, not better ranking.",
        "",
        "## Fixed model comparison",
        "",
        "All rows below are equal-weight means across the same seven outer projects. Configurations were declared before the run and were not selected from outer results.",
        "",
        "| Model | AP | Top-20% recall | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        values = model_summary[model_name]
        lines.append(
            f"| {model_name} | {values['average_precision']['mean']:.3f} | "
            f"{values['top_20_percent_recall']['mean']:.3f} | "
            f"{values['precision']['mean']:.3f} | {values['recall']['mean']:.3f} | "
            f"{values['f1']['mean']:.3f} |"
        )
    logistic_ap = model_summary["logistic_regression"]["average_precision"]["mean"]
    forest_ap = model_summary["random_forest"]["average_precision"]["mean"]
    boosting_ap = model_summary["hist_gradient_boosting"]["average_precision"]["mean"]
    forest_top = model_summary["random_forest"]["top_20_percent_recall"]["mean"]
    logistic_top = model_summary["logistic_regression"]["top_20_percent_recall"]["mean"]
    lines.extend(
        [
            "",
            f"Logistic regression had the highest macro AP. Random forest differed by **{forest_ap - logistic_ap:+.3f}** AP and **{forest_top - logistic_top:+.3f}** top-20% recall; histogram gradient boosting differed by **{boosting_ap - logistic_ap:+.3f}** AP. Random forest and boosting had higher F1 at 0.5, but that reflects different score distributions at a fixed cutoff and does not overturn the AP ranking result.",
            "",
            "### Dummy ranking ties",
            "",
            f"Dummy prior assigns the same score to every target row. With the required deterministic identity tie-break, its observed macro top-20% recall was **{dummy_observed:.3f}**. Under genuinely random selection of the same exact per-project budgets, the expected macro recall is the mean reviewed fraction, **{dummy_expected:.3f}**. The observed value can be above or below that expectation because the deterministic release/class-name order happens to include a particular set of labels; it is not evidence that the dummy model ranks defects.",
            "",
            "## Similar-project selection and exhaustive reference",
            "",
            "Similarity preprocessing uses source-only median imputation and standard scaling. Each source and the unlabelled target is summarized by 20 feature medians plus 20 interquartile ranges; Euclidean distance selects the three closest sources. A new baseline logistic pipeline is then fitted only on the selected raw training rows. The similarity scaler is not reused for prediction.",
            "",
            "| Training strategy | AP | Top-20% recall | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ("all_sources", "similar_sources", "random_subset_mean"):
        values = similarity_summary["strategies"][strategy]
        lines.append(
            f"| {strategy} | {values['average_precision']['mean']:.3f} | "
            f"{values['top_20_percent_recall']['mean']:.3f} | "
            f"{values['precision']['mean']:.3f} | {values['recall']['mean']:.3f} | "
            f"{values['f1']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Similarity selection changed AP by **{sim_diff['average_precision']['mean']:+.3f}** and top-20% recall by **{sim_diff['top_20_percent_recall']['mean']:+.3f}** versus all sources. Versus the mean exhaustive three-source reference, the changes were **{sim_random_diff['average_precision']['mean']:+.3f}** AP and **{sim_random_diff['top_20_percent_recall']['mean']:+.3f}** top-20% recall. It improved AP over all sources only for Log4j and regressed for the other six targets. Similarity therefore did not improve ranking overall.",
            "",
            "Similarity increased macro F1 at 0.5, but behavior was heterogeneous: target-specific recall ranged from 0.027 to 0.871. Higher F1 at this cutoff does not establish improved ranking. The exhaustive reference matches the number of source projects, not training-row counts; unequal source-project sizes remain a confounder.",
            "",
            "## Related work",
            "",
            "- Jureczko and Madeyski introduced and analyzed the multi-project corpus and investigated project clustering for defect prediction: [*Towards Identifying Software Project Clusters with Regard to Defect Prediction*](https://doi.org/10.1145/1868328.1868342) (PROMISE 2010).",
            "- Turhan, Menzies, Bener, and Di Stefano studied filtered cross-company data and within-company predictors: [*On the Relative Value of Cross-Company and Within-Company Data for Defect Prediction*](https://doi.org/10.1007/s10664-008-9103-7) (Empirical Software Engineering, 2009).",
            "- Herbold evaluated training-data selection for cross-project prediction: [*Training Data Selection for Cross-Project Defect Prediction*](https://doi.org/10.1145/2499393.2499395) (PROMISE 2013).",
            "- Nam, Pan, and Kim proposed transfer defect learning: [*Transfer Defect Learning*](https://doi.org/10.1109/ICSE.2013.6606584) (ICSE 2013).",
            "",
            "These primary studies motivate cross-project transfer and source selection. Their datasets, selection rules, validation designs, and metrics differ, so their reported outcomes are not treated as directly comparable effect estimates for this experiment.",
            "",
            "## Limitations",
            "",
            "- Only seven historical Java projects are evaluated; results may not transfer to current languages, development practices, or organizations.",
            "- Releases repeat classes over time, so rows are not necessarily temporally or structurally independent.",
            "- Exact 20-metric vector matches occur across projects, sometimes with conflicting labels. A metric match alone does not prove copied code; no rows were removed automatically.",
            "- Defect labels are historical observational data and may be incomplete or incorrectly linked.",
            "- Projects have unequal sizes and defect prevalence. Three-project references do not guarantee equal training-row counts.",
            "- Target metric distributions are available to similarity selection, so it is an unlabelled-target-access approach rather than a strictly source-only method.",
            "- The fixed configurations are bounded comparisons, not an exhaustive model search; scores are not calibrated probabilities.",
            "- No novelty, statistical-significance, causal, or production-readiness claim is established.",
            "",
            "## Reproduction",
            "",
            "From the repository root on PowerShell, after the one-time environment setup in the README:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.download",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.experiment",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.evaluate",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.lopo",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.source_validation",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.model_comparison",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.source_similarity",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.research_reporting",
            ".\\.venv\\Scripts\\python.exe -m defect_prediction.deployment",
            "```",
            "",
            "The correct similarity module is `defect_prediction.source_similarity`; there is no `source_similarity4` runner.",
            "",
            "Configurations and protocols:",
            "",
            "- [Original baseline configuration](../config/experiment.json)",
            "- [Fixed model-comparison configuration](../config/model_comparison.json) and [protocol](../docs/MODEL_COMPARISON_PROTOCOL.md)",
            "- [Similarity-selection configuration](../config/source_similarity.json) and [protocol](../docs/SOURCE_SIMILARITY_PROTOCOL.md)",
            "- [Dataset manifest](../config/data_manifest.json) and [dataset documentation](../docs/DATASET.md)",
            "",
            "Research models and reports remain historical. The API continues to serve the separately packaged, checksum-pinned `logistic-regression-jedit-holdout-v1` artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    lopo_path: Path,
    source_validation_path: Path,
    model_comparison_path: Path,
    similarity_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    model_full = _read_json(model_comparison_path)
    similarity_full = _read_json(similarity_path)
    baseline = build_baseline_summary(lopo_path, source_validation_path)
    model = build_model_comparison_summary(model_full, model_comparison_path)
    similarity = build_similarity_summary(similarity_full, similarity_path)
    project_sets = [
        {row["project"] for row in baseline["projects"]},
        {row["project"] for row in model["projects"]},
        {row["project"] for row in similarity["projects"]},
    ]
    if not project_sets[0] == project_sets[1] == project_sets[2] or len(project_sets[0]) != 7:
        raise RuntimeError("Research reports do not describe the same seven projects.")
    if not similarity_full["baseline_reproduction"]["matches"]:
        raise RuntimeError("Similarity all-source results did not reproduce the baseline.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reports_dir / "baseline" / "summary.json", baseline)
    _write_json(reports_dir / "model_comparison" / "summary.json", model)
    _write_json(reports_dir / "source_similarity" / "summary.json", similarity)
    pd.DataFrame(_model_rows(model)).to_csv(
        reports_dir / "model_comparison" / "per_project_results.csv", index=False
    )
    source_model_report = model_comparison_path.parent / "REPORT.md"
    if source_model_report.exists():
        (reports_dir / "model_comparison" / "REPORT.md").write_text(
            source_model_report.read_text(encoding="utf-8"), encoding="utf-8"
        )
    final_report = _final_report(baseline, model, similarity, model_full)
    (reports_dir / "FINAL_REPORT.md").write_text(final_report, encoding="utf-8")
    return {"baseline": baseline, "model_comparison": model, "source_similarity": similarity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lopo", type=Path, default=Path("artifacts/lopo/lopo_results.json")
    )
    parser.add_argument(
        "--source-validation",
        type=Path,
        default=Path("artifacts/source_validation/source_validation_results.json"),
    )
    parser.add_argument(
        "--model-comparison",
        type=Path,
        default=Path("artifacts/model_comparison/comparison_results.json"),
    )
    parser.add_argument(
        "--source-similarity",
        type=Path,
        default=Path("artifacts/source_similarity/results.json"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = write_reports(
        args.lopo,
        args.source_validation,
        args.model_comparison,
        args.source_similarity,
        args.reports_dir,
    )
    print(
        json.dumps(
            {
                "reports": sorted(result),
                "project_count": result["baseline"]["project_count"],
                "output": args.reports_dir.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
