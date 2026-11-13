from __future__ import annotations

import pandas as pd
import pytest

from defect_prediction.data import DataValidationError, validate_dataset
from defect_prediction.schema import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN


def test_validation_builds_binary_target_and_keeps_metadata_out_of_features(sample_data):
    validated = validate_dataset(sample_data)
    assert set(validated[TARGET_COLUMN].unique()) == {0, 1}
    assert (validated[TARGET_COLUMN] == (validated["bug_count"] > 0)).all()
    assert set(METADATA_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert "bug_count" not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_validation_rejects_missing_required_feature(sample_data):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(sample_data.drop(columns=["loc"]))


def test_validation_rejects_non_numeric_feature(sample_data):
    sample_data["wmc"] = sample_data["wmc"].astype("object")
    sample_data.loc[0, "wmc"] = "not-a-number"
    with pytest.raises(DataValidationError, match="non-numeric"):
        validate_dataset(sample_data)


def test_validation_rejects_duplicate_identifiers(sample_data):
    duplicated = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_dataset(duplicated)
