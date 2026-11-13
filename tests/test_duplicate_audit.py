from defect_prediction.data import validate_dataset
from defect_prediction.duplicate_audit import audit_cross_split_duplicates
from defect_prediction.modeling import split_by_project
from defect_prediction.schema import FEATURE_COLUMNS


def test_duplicate_audit_distinguishes_identifiers_metrics_and_conflicts(sample_data):
    data = validate_dataset(sample_data)
    train, test = split_by_project(data, "heldout")

    # Give one test row the same predictors/label as a training row and another
    # the same predictors but the opposite label. Their identifiers stay different.
    test.loc[test.index[0], FEATURE_COLUMNS] = train.iloc[0].loc[list(FEATURE_COLUMNS)]
    test.loc[test.index[0], "bug_count"] = train.iloc[0]["bug_count"]
    test.loc[test.index[0], "defective"] = train.iloc[0]["defective"]
    test.loc[test.index[1], FEATURE_COLUMNS] = train.iloc[0].loc[list(FEATURE_COLUMNS)]
    test.loc[test.index[1], "bug_count"] = 1
    test.loc[test.index[1], "defective"] = 1

    audit = audit_cross_split_duplicates(train, test)
    assert audit["exact_identifier_match_rows"] == 0
    assert audit["same_class_name_match_rows"] == 0
    assert audit["exact_feature_vector_match_rows"] >= 2
    assert audit["exact_feature_and_binary_target_match_rows"] >= 1
    assert audit["feature_match_with_conflicting_target_rows"] >= 1

