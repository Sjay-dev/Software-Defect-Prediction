import joblib
import numpy as np

from defect_prediction.data import validate_dataset
from defect_prediction.modeling import (
    build_logistic_pipeline,
    positive_probabilities,
    split_by_project,
)
from defect_prediction.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_predictions_are_identical_after_save_and_load(sample_data, tmp_path):
    validated = validate_dataset(sample_data)
    train, test = split_by_project(validated, "heldout")
    model = build_logistic_pipeline(random_state=42)
    model.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])
    before = positive_probabilities(model, test.loc[:, FEATURE_COLUMNS])

    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    after = positive_probabilities(restored, test.loc[:, FEATURE_COLUMNS])

    np.testing.assert_array_equal(before, after)

