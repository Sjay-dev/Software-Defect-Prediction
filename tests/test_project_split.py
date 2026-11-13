from defect_prediction.data import validate_dataset
from defect_prediction.modeling import split_by_project


def test_holdout_contains_every_release_and_never_appears_in_training(sample_data):
    validated = validate_dataset(sample_data)
    train, test = split_by_project(validated, "heldout")

    assert "heldout" not in set(train["project"])
    assert set(test["project"]) == {"heldout"}
    assert set(test["release"]) == {"1.0", "2.0"}
    assert set(train["project"]).isdisjoint(test["project"])
    assert len(train) + len(test) == len(validated)

