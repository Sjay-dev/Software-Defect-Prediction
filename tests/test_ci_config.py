from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from defect_prediction.ci_config import write_verified_ci_service_config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_ci_config_pins_generated_artifact_without_changing_release_config(tmp_path):
    config_dir = tmp_path / "checkout" / "config"
    report_path = tmp_path / "checkout" / "reports" / "baseline.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}", encoding="utf-8")
    base_config_path = config_dir / "service.json"
    release_hash = "1" * 64
    base_config = {
        "model_manifest": "../release/model_manifest.json",
        "expected_model_sha256": release_hash,
        "expected_model_version": "test-v1",
        "historical_reports": {"baseline_summary": "../reports/baseline.json"},
    }
    _write_json(base_config_path, base_config)

    deployment = tmp_path / "generated"
    model_path = deployment / "pipeline.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"linux-generated-model")
    generated_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_path = deployment / "model_manifest.json"
    _write_json(
        manifest_path,
        {
            "model_version": "test-v1",
            "model_file": "pipeline.joblib",
            "model_sha256": generated_hash,
        },
    )
    output_path = tmp_path / "runner-temp" / "service.json"

    hashes = write_verified_ci_service_config(
        base_config_path, manifest_path, output_path
    )

    generated_config = json.loads(output_path.read_text(encoding="utf-8"))
    assert generated_config["expected_model_sha256"] == generated_hash
    assert generated_config["model_manifest"] == str(manifest_path.resolve())
    assert generated_config["historical_reports"]["baseline_summary"] == str(
        report_path.resolve()
    )
    assert hashes["configured_sha256"] == hashes["manifest_sha256"]
    assert hashes["manifest_sha256"] == hashes["actual_sha256"]
    assert json.loads(base_config_path.read_text(encoding="utf-8")) == base_config
    assert release_hash != generated_hash


def test_ci_config_rejects_manifest_artifact_checksum_mismatch(tmp_path):
    base_config_path = tmp_path / "config" / "service.json"
    _write_json(
        base_config_path,
        {
            "expected_model_version": "test-v1",
            "historical_reports": {},
        },
    )
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "pipeline.joblib").write_bytes(b"corrupt")
    manifest_path = deployment / "model_manifest.json"
    _write_json(
        manifest_path,
        {
            "model_version": "test-v1",
            "model_file": "pipeline.joblib",
            "model_sha256": "0" * 64,
        },
    )
    output_path = tmp_path / "ci-service.json"

    with pytest.raises(RuntimeError, match="manifest=.*actual="):
        write_verified_ci_service_config(
            base_config_path, manifest_path, output_path
        )

    assert not output_path.exists()
