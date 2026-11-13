import numpy as np

from defect_prediction.data import validate_dataset
from defect_prediction.modeling import build_logistic_pipeline, split_by_project
from defect_prediction.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_imputation_and_scaling_statistics_come_only_from_training_rows(sample_data):
    data = validate_dataset(sample_data)
    train, test = split_by_project(data, "heldout")
    model = build_logistic_pipeline(random_state=42)
    model.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])

    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    expected_medians = train.loc[:, FEATURE_COLUMNS].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected_medians)
    imputed_training = imputer.transform(train.loc[:, FEATURE_COLUMNS])
    np.testing.assert_allclose(scaler.mean_, imputed_training.mean(axis=0))

    # The test project has different values, but none of them entered these statistics.
    assert not np.isclose(test["wmc"].median(), imputer.statistics_[0])

