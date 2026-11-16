from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient

from defect_prediction.api import create_app
from defect_prediction.data import validate_dataset
from defect_prediction.deployment import package_deployment_model
from defect_prediction.modeling import (
    build_logistic_pipeline,
    positive_probabilities,
    split_by_project,
)
from defect_prediction.prediction import ModelLoadError, load_predictor, load_service_settings
from defect_prediction.schema import FEATURE_COLUMNS, FEATURE_DEFINITIONS, TARGET_COLUMN


def _deployment(tmp_path: Path, sample_data, *, max_batch_size: int = 100):
    data = validate_dataset(sample_data)
    train, test = split_by_project(data, "heldout")
    model = build_logistic_pipeline(random_state=42)
    model.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN])

    source = tmp_path / "evaluation"
    source.mkdir()
    joblib.dump(model, source / "logistic_pipeline.joblib")
    (source / "experiment_config.json").write_text(
        json.dumps(
            {
                "training_projects": ["alpha", "beta"],
                "test_projects": ["heldout"],
                "experiment_name": "test_experiment",
                "tuning": "none",
                "preprocessing_fit_scope": "training projects only",
            }
        ),
        encoding="utf-8",
    )
    (source / "feature_schema.json").write_text(
        json.dumps(
            {
                "predictive_features": [
                    {"name": name, "dtype": "numeric", "definition": FEATURE_DEFINITIONS[name]}
                    for name in FEATURE_COLUMNS
                ]
            }
        ),
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment"
    manifest = package_deployment_model(source, deployment, "test-v1")
    config = tmp_path / "service.json"
    config.write_text(
        json.dumps(
            {
                "model_manifest": str(deployment / "model_manifest.json"),
                "expected_model_sha256": manifest["model_sha256"],
                "expected_model_version": "test-v1",
                "max_batch_size": max_batch_size,
            }
        ),
        encoding="utf-8",
    )
    return config, deployment, model, test


def _record(row, identifier: str = "record-1") -> dict:
    return {
        "record_id": identifier,
        **{
            name: None if np.isnan(row[name]) else float(row[name])
            for name in FEATURE_COLUMNS
        },
    }


def test_api_loads_model_and_reports_exact_provenance(tmp_path, sample_data):
    config, _, _, _ = _deployment(tmp_path, sample_data)
    expected_hash = json.loads(config.read_text(encoding="utf-8"))[
        "expected_model_sha256"
    ]

    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "test-v1",
    }
    assert info.json()["training_projects"] == ["alpha", "beta"]
    assert info.json()["model_sha256"] == expected_hash
    assert info.json()["evaluation_only_projects"] == ["heldout"]
    assert info.json()["features"] == list(FEATURE_COLUMNS)
    assert "not a calibrated probability" in info.json()["score_description"]


def test_api_allows_configured_local_frontend_origin(tmp_path, sample_data):
    config, _, _, _ = _deployment(tmp_path, sample_data)

    with TestClient(create_app(config)) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_missing_configured_reports_do_not_block_predictions(tmp_path, sample_data):
    config, _, _, test = _deployment(tmp_path, sample_data)
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings["historical_reports"] = {
        "lopo_results": "missing-lopo.json",
        "source_validation_results": "missing-source.json",
    }
    config.write_text(json.dumps(settings), encoding="utf-8")

    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict", json={"records": [_record(test.iloc[0])]}
        )
        reports = client.get("/experiment-results")

    assert health.status_code == 200
    assert prediction.status_code == 200
    assert reports.status_code == 503
    assert "Historical reports are unavailable" in reports.json()["detail"]


def test_api_predictions_agree_with_direct_pipeline(tmp_path, sample_data):
    config, _, model, test = _deployment(tmp_path, sample_data)
    rows = test.iloc[:2]
    payload = {
        "records": [
            _record(rows.iloc[0], "first"),
            _record(rows.iloc[1], "second"),
        ]
    }
    direct = positive_probabilities(model, rows.loc[:, FEATURE_COLUMNS])

    with TestClient(create_app(config)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "not a calibrated probability" in body["score_description"]
    np.testing.assert_allclose(
        [item["model_score"] for item in body["predictions"]], direct, rtol=0, atol=0
    )
    assert {item["rank"] for item in body["predictions"]} == {1, 2}
    assert all(item["model_version"] == "test-v1" for item in body["predictions"])


def test_api_rejects_missing_unknown_and_duplicate_inputs(tmp_path, sample_data):
    config, _, _, test = _deployment(tmp_path, sample_data)
    valid = _record(test.iloc[0])
    missing = dict(valid)
    missing.pop("loc")
    unknown = {**valid, "bug_count": 1}

    with TestClient(create_app(config)) as client:
        assert client.post("/predict", json={"records": [missing]}).status_code == 422
        assert client.post("/predict", json={"records": [unknown]}).status_code == 422
        assert (
            client.post("/predict", json={"records": [valid, valid]}).status_code
            == 422
        )


def test_configured_batch_limit_is_enforced(tmp_path, sample_data):
    config, _, _, test = _deployment(tmp_path, sample_data, max_batch_size=1)
    records = [_record(test.iloc[0], "one"), _record(test.iloc[1], "two")]

    with TestClient(create_app(config)) as client:
        response = client.post("/predict", json={"records": records})

    assert response.status_code == 422
    assert response.json()["detail"] == "Batch exceeds configured limit of 1."


def test_model_loader_rejects_tampered_artifact(tmp_path, sample_data):
    config, deployment, _, _ = _deployment(tmp_path, sample_data)
    model_path = deployment / "pipeline.joblib"
    original = model_path.read_bytes()
    model_path.write_bytes(original + b"tampered")
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() != json.loads(
        config.read_text(encoding="utf-8")
    )["expected_model_sha256"]

    try:
        load_predictor(load_service_settings(config))
    except ModelLoadError as error:
        assert "checksum verification failed" in str(error)
    else:
        raise AssertionError("Tampered model was loaded")
