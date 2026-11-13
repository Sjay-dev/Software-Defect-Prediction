from __future__ import annotations

import numpy as np

from defect_prediction.data import validate_dataset
from defect_prediction.source_validation import (
    choose_threshold,
    select_threshold_for_outer_project,
    top_budget_indices,
)


def test_outer_labels_cannot_change_source_selected_threshold(sample_data):
    data = validate_dataset(sample_data)
    original = select_threshold_for_outer_project(
        data, "heldout", random_state=42
    )

    changed = data.copy()
    outer_mask = changed["project"].eq("heldout")
    changed.loc[outer_mask, "defective"] = 1 - changed.loc[outer_mask, "defective"]
    changed.loc[outer_mask, "bug_count"] = changed.loc[outer_mask, "defective"]
    repeated = select_threshold_for_outer_project(
        changed, "heldout", random_state=42
    )

    assert original["selected_threshold"] == repeated["selected_threshold"]
    assert original["threshold_scores"] == repeated["threshold_scores"]
    assert all(
        "heldout" not in fold["training_projects"]
        and fold["validation_project"] != "heldout"
        for fold in original["inner_folds"]
    )


def test_threshold_tie_break_prefers_closest_to_half_then_lower():
    assert choose_threshold({0.2: 0.4, 0.4: 0.7, 0.6: 0.7, 0.8: 0.5}) == 0.4
    assert choose_threshold({0.3: 0.8, 0.5: 0.8, 0.7: 0.8}) == 0.5


def test_top_twenty_percent_uses_exact_ceiling_budget_and_stable_ties(sample_data):
    target = validate_dataset(sample_data)
    target = target[target["project"].eq("heldout")].copy().iloc[:5]
    target.loc[:, "release"] = ["2.0", "1.0", "1.0", "2.0", "1.0"]
    target.loc[:, "class_name"] = ["Z", "B", "A", "A", "C"]
    selected = top_budget_indices(target, np.full(len(target), 0.75), fraction=0.20)

    assert len(selected) == 1
    assert target.loc[selected[0], "release"] == "1.0"
    assert target.loc[selected[0], "class_name"] == "A"

