from defect_prediction.data import validate_dataset
from defect_prediction.lopo import comparison_rows, evaluate_lopo_frame, summarize_folds


def test_lopo_holds_out_every_project_once_with_all_releases(sample_data):
    data = validate_dataset(sample_data)
    folds = evaluate_lopo_frame(data, random_state=42, threshold=0.5)

    assert {fold["held_out_project"] for fold in folds} == {
        "alpha",
        "beta",
        "heldout",
    }
    for fold in folds:
        assert fold["held_out_project"] not in fold["training_projects"]
        assert fold["test_releases"] == ["1.0", "2.0"]
        assert fold["sample_count"] == 8

    summary = summarize_folds(folds)
    assert summary["project_count"] == 3

    comparison = comparison_rows(folds)
    assert len(comparison) == 3
    assert set(comparison[0]) == {
        "held_out_project",
        "test_rows",
        "defect_prevalence",
        "dummy_average_precision",
        "logistic_average_precision",
        "logistic_precision",
        "logistic_recall",
        "logistic_f1",
        "tn",
        "fp",
        "fn",
        "tp",
    }
