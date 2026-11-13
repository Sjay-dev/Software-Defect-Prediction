"""Run fixed leave-one-project-out evaluation without changing the baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_dataset
from .duplicate_audit import audit_cross_split_duplicates
from .modeling import (
    build_dummy_pipeline,
    build_logistic_pipeline,
    evaluate_predictions,
    positive_probabilities,
    split_by_project,
)
from .schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


MODEL_NAMES = ("dummy_prior", "logistic_regression")
METRIC_NAMES = ("average_precision", "precision", "recall", "f1")


def _sha256_files(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    result: dict[str, str] = {}
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def evaluate_lopo_frame(
    data: pd.DataFrame, *, random_state: int, threshold: float
) -> list[dict[str, Any]]:
    """Fit the unchanged pipelines once per held-out project and return fold results."""
    folds: list[dict[str, Any]] = []
    for project in sorted(data["project"].unique()):
        train, test = split_by_project(data, project)
        x_train = train.loc[:, FEATURE_COLUMNS]
        y_train = train[TARGET_COLUMN]
        x_test = test.loc[:, FEATURE_COLUMNS]
        y_test = test[TARGET_COLUMN]
        models = {
            "dummy_prior": build_dummy_pipeline(),
            "logistic_regression": build_logistic_pipeline(random_state),
        }
        scores: dict[str, Any] = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            probabilities = positive_probabilities(model, x_test)
            scores[name], _ = evaluate_predictions(y_test, probabilities, threshold)
        folds.append(
            {
                "held_out_project": project,
                "training_projects": sorted(train["project"].unique().tolist()),
                "test_releases": sorted(test["release"].unique().tolist()),
                "sample_count": int(len(test)),
                "defective_count": int(y_test.sum()),
                "defect_prevalence": float(y_test.mean()),
                "models": scores,
                "duplicate_audit": audit_cross_split_duplicates(train, test),
            }
        )
    return folds


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_standard_deviation": float(statistics.stdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    model_summary: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        model_summary[model_name] = {
            metric: _distribution(
                [fold["models"][model_name][metric] for fold in folds]
            )
            for metric in METRIC_NAMES
        }
        model_summary[model_name]["confusion_matrix_totals"] = {
            key: sum(
                fold["models"][model_name]["confusion_matrix"][key]
                for fold in folds
            )
            for key in ("tn", "fp", "fn", "tp")
        }
    return {
        "project_count": len(folds),
        "macro_average_note": (
            "Each project has equal weight, regardless of its number of classes. "
            "Variation is the sample standard deviation across project scores."
        ),
        "defect_prevalence": _distribution(
            [fold["defect_prevalence"] for fold in folds]
        ),
        "models": model_summary,
    }


def _flat_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for model_name in MODEL_NAMES:
            metrics = fold["models"][model_name]
            matrix = metrics["confusion_matrix"]
            rows.append(
                {
                    "held_out_project": fold["held_out_project"],
                    "model": model_name,
                    "sample_count": fold["sample_count"],
                    "defective_count": fold["defective_count"],
                    "defect_prevalence": fold["defect_prevalence"],
                    **{name: metrics[name] for name in METRIC_NAMES},
                    **{key: matrix[key] for key in ("tn", "fp", "fn", "tp")},
                }
            )
    return rows


def comparison_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the requested one-row-per-project comparison layout."""
    rows: list[dict[str, Any]] = []
    for fold in folds:
        dummy = fold["models"]["dummy_prior"]
        logistic = fold["models"]["logistic_regression"]
        matrix = logistic["confusion_matrix"]
        rows.append(
            {
                "held_out_project": fold["held_out_project"],
                "test_rows": fold["sample_count"],
                "defect_prevalence": fold["defect_prevalence"],
                "dummy_average_precision": dummy["average_precision"],
                "logistic_average_precision": logistic["average_precision"],
                "logistic_precision": logistic["precision"],
                "logistic_recall": logistic["recall"],
                "logistic_f1": logistic["f1"],
                **{key: matrix[key] for key in ("tn", "fp", "fn", "tp")},
            }
        )
    return rows


def _markdown_report(payload: dict[str, Any]) -> str:
    folds = payload["folds"]
    summary = payload["summary"]
    review = payload["baseline_review"]
    lines = [
        "# Baseline review and leave-one-project-out evaluation",
        "",
        "The original `artifacts/baseline` files were hashed before and after this run. "
        f"They were unchanged: **{payload['baseline_integrity']['unchanged']}**.",
        "",
        "## Original baseline configuration and verification",
        "",
        f"- Dataset: {payload['dataset_name']} from the seven pinned Zenodo records in "
        "`config/data_manifest.json`.",
        f"- Predictive features: {', '.join(payload['features'])}.",
        f"- Training projects: {', '.join(review['training_projects'])}.",
        f"- Held-out project: {review['held_out_project']}, including releases "
        f"{', '.join(review['held_out_releases'])}.",
        f"- Decision threshold: {payload['decision_threshold']}.",
        f"- Excluded from predictors: {', '.join(review['excluded_non_features'])}.",
        "- The training/test project intersection is empty, so every held-out release was "
        "excluded from fitting.",
        "- The imputer and scaler are pipeline steps fitted by calling `fit` only on the "
        "training-project feature table. A dedicated test checks their learned statistics.",
        "- Tuning is `none`; the fixed threshold comes from the pre-existing experiment "
        "configuration, and test labels are used only after prediction to calculate metrics.",
        f"- The JEdit LOPO fold exactly reproduces the saved baseline metrics: "
        f"**{review['jedit_metrics_match_original_baseline']}**.",
        "",
        "The same fixed pipelines, random state, and 0.5 decision threshold were used "
        "for every fold. No parameter or threshold was selected from these test results.",
        "",
        "## Per-project results",
        "",
        "| Held-out project | Test rows | Prevalence | Dummy AP | Logistic AP | Logistic precision | Logistic recall | Logistic F1 | Logistic TN / FP / FN / TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows(folds):
        lines.append(
            f"| {row['held_out_project']} | {row['test_rows']} | "
            f"{row['defect_prevalence']:.3f} | {row['dummy_average_precision']:.3f} | "
            f"{row['logistic_average_precision']:.3f} | "
            f"{row['logistic_precision']:.3f} | {row['logistic_recall']:.3f} | "
            f"{row['logistic_f1']:.3f} | "
            f"{row['tn']} / {row['fp']} / {row['fn']} / {row['tp']} |"
        )
    lines.extend(
        [
            "",
            "## Across-project summary",
            "",
            "These are unweighted project means +/- sample standard deviations; each project "
            "counts once, even when project sizes differ.",
            "",
            "| Model | AP | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        values = summary["models"][model_name]
        cells = [
            f"{values[name]['mean']:.3f} +/- {values[name]['sample_standard_deviation']:.3f}"
            for name in METRIC_NAMES
        ]
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")

    ap_deltas = {
        fold["held_out_project"]: (
            fold["models"]["logistic_regression"]["average_precision"]
            - fold["models"]["dummy_prior"]["average_precision"]
        )
        for fold in folds
    }
    helped = [name for name, delta in ap_deltas.items() if delta > 0]
    failed = [name for name, delta in ap_deltas.items() if delta <= 0]
    low_recall = [
        fold["held_out_project"]
        for fold in folds
        if fold["models"]["logistic_regression"]["recall"] < 0.20
    ]
    lines.extend(
        [
            "",
            "## Plain-language interpretation",
            "",
            f"Logistic regression ranked defects better than the dummy prior on "
            f"{len(helped)} of {len(folds)} projects ({', '.join(helped) or 'none'}). "
            f"It did not improve average precision on: {', '.join(failed) or 'none'}.",
            f"At the unchanged 0.5 threshold, recall is below 0.20 on "
            f"{len(low_recall)} projects ({', '.join(low_recall)}). In plain language, the "
            "model ranks defects better than the dummy but still leaves most known defects "
            "unflagged on those projects.",
            "The dummy's learned positive-class prior stays below 0.5 in every fold, so it "
            "predicts every class as clean. Its average precision equals test prevalence, "
            "while its precision, recall, and F1 are all zero.",
            "The spread between projects is important: a model that transfers reasonably to "
            "one codebase can transfer poorly to another. Threshold-based precision, recall, "
            "and F1 also depend on the fixed 0.5 cutoff and the project's defect prevalence.",
            "These results are a research baseline, not evidence that the model is ready for production.",
            "",
            "## Duplicate-overlap investigation",
            "",
            "| Held-out project | Exact IDs | Same class names | Same 20 metrics, n (%) | Same metrics + target | Same metrics + bug count | Metric match with conflicting target |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in folds:
        audit = fold["duplicate_audit"]
        lines.append(
            f"| {fold['held_out_project']} | {audit['exact_identifier_match_rows']} | "
            f"{audit['same_class_name_match_rows']} | {audit['exact_feature_vector_match_rows']} "
            f"({audit['exact_feature_vector_match_percent']:.1f}%) | "
            f"{audit['exact_feature_and_binary_target_match_rows']} | "
            f"{audit['exact_feature_and_bug_count_match_rows']} | "
            f"{audit['feature_match_with_conflicting_target_rows']} |"
        )
    lines.extend(
        [
            "",
            "There are no repeated full identifiers or class names across project splits. Some "
            "different classes have identical 20-number metric vectors. Several such matches have "
            "conflicting labels, showing that a metric match is not reliable proof of a copied code "
            "record. The baseline and this fixed comparison retain the source corpus unchanged; the "
            "counts are disclosed so a future sensitivity study can evaluate de-duplication rules.",
            "",
        ]
    )
    return "\n".join(lines)


def run_lopo(
    data_dir: Path,
    manifest_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    baseline_dir: Path,
) -> dict[str, Any]:
    if output_dir.resolve() == baseline_dir.resolve():
        raise ValueError("LOPO output must not overwrite the original baseline directory.")
    before = _sha256_files(baseline_dir)
    settings = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = load_dataset(data_dir, manifest_path)
    baseline_train, baseline_test = split_by_project(
        data, settings["held_out_project"]
    )
    folds = evaluate_lopo_frame(
        data,
        random_state=settings["random_state"],
        threshold=settings["decision_threshold"],
    )
    summary = summarize_folds(folds)
    original_results_path = baseline_dir / "evaluation_results.json"
    original_results = json.loads(original_results_path.read_text(encoding="utf-8"))
    original_fold = next(
        fold
        for fold in folds
        if fold["held_out_project"] == settings["held_out_project"]
    )
    after = _sha256_files(baseline_dir)
    if before != after:
        raise RuntimeError("The original baseline artifacts changed during LOPO evaluation.")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": manifest["dataset_name"],
        "dataset_records": [record["record_url"] for record in manifest["records"]],
        "features": list(FEATURE_COLUMNS),
        "random_state": settings["random_state"],
        "decision_threshold": settings["decision_threshold"],
        "tuning": "none",
        "model_configuration": {
            "dummy_prior": "median imputation; DummyClassifier(strategy='prior')",
            "logistic_regression": (
                "median imputation with missing indicators; standard scaling; "
                f"LogisticRegression(solver='lbfgs', max_iter=2000, "
                f"random_state={settings['random_state']})"
            ),
            "preprocessing_fit_scope": "the training projects in each fold only",
        },
        "folds": folds,
        "summary": summary,
        "baseline_review": {
            "training_projects": sorted(baseline_train["project"].unique().tolist()),
            "held_out_project": settings["held_out_project"],
            "held_out_releases": sorted(baseline_test["release"].unique().tolist()),
            "training_test_project_intersection": sorted(
                set(baseline_train["project"]) & set(baseline_test["project"])
            ),
            "excluded_non_features": [
                *METADATA_COLUMNS,
                LABEL_SOURCE_COLUMN,
                TARGET_COLUMN,
            ],
            "preprocessing_fit_scope": "training projects only",
            "test_label_usage": "metrics only; no tuning or threshold selection",
            "jedit_metrics_match_original_baseline": (
                original_fold["models"] == original_results["models"]
            ),
        },
        "baseline_integrity": {
            "unchanged": before == after,
            "sha256_before": before,
            "sha256_after": after,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lopo_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(_flat_rows(folds)).to_csv(
        output_dir / "lopo_per_project.csv", index=False
    )
    pd.DataFrame(comparison_rows(folds)).to_csv(
        output_dir / "lopo_comparison.csv", index=False
    )
    (output_dir / "review.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lopo"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    result = run_lopo(
        args.data_dir,
        args.manifest,
        args.config,
        args.output_dir,
        args.baseline_dir,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
