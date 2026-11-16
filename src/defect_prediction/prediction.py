"""Trusted model loading, feature validation, scoring, and ranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .modeling import positive_probabilities
from .schema import FEATURE_COLUMNS


class ModelLoadError(RuntimeError):
    """Raised when a configured deployment artifact is missing or inconsistent."""


@dataclass(frozen=True)
class ServiceSettings:
    manifest_path: Path
    expected_model_sha256: str
    expected_model_version: str
    max_batch_size: int
    cors_origins: tuple[str, ...]
    lopo_results_path: Path | None
    source_validation_results_path: Path | None


def load_service_settings(config_path: Path) -> ServiceSettings:
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    def resolve_optional(value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else (config_path.parent / path).resolve()

    manifest_path = Path(raw["model_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()
    max_batch_size = int(raw["max_batch_size"])
    if not 1 <= max_batch_size <= 100:
        raise ModelLoadError("max_batch_size must be between 1 and 100.")
    cors_origins = tuple(
        raw.get(
            "cors_origins",
            ["http://localhost:5173", "http://127.0.0.1:5173"],
        )
    )
    historical_reports = raw.get("historical_reports", {})
    return ServiceSettings(
        manifest_path=manifest_path,
        expected_model_sha256=str(raw["expected_model_sha256"]),
        expected_model_version=str(raw["expected_model_version"]),
        max_batch_size=max_batch_size,
        cors_origins=cors_origins,
        lopo_results_path=resolve_optional(historical_reports.get("lopo_results")),
        source_validation_results_path=resolve_optional(
            historical_reports.get("source_validation_results")
        ),
    )


class Predictor:
    def __init__(
        self,
        *,
        pipeline: Any,
        manifest: dict[str, Any],
        feature_schema: dict[str, Any],
        max_batch_size: int,
    ) -> None:
        self.pipeline = pipeline
        self.manifest = manifest
        self.feature_schema = feature_schema
        self.max_batch_size = max_batch_size

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    def model_info(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "model_sha256": self.manifest["model_sha256"],
            "model_type": self.manifest["model_type"],
            "score_description": self.manifest["score_description"],
            "training_projects": self.manifest["training_projects"],
            "evaluation_only_projects": self.manifest["evaluation_only_projects"],
            "features": list(FEATURE_COLUMNS),
            "max_batch_size": self.max_batch_size,
            "packaging_method": self.manifest["packaging_method"],
            "decision_threshold": self.manifest.get("decision_threshold"),
            "calibration_status": self.manifest.get("calibration_status"),
        }

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(records) <= self.max_batch_size:
            raise ValueError(
                f"Batch must contain between 1 and {self.max_batch_size} records."
            )
        identifiers = [str(record["record_id"]) for record in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("record_id values must be unique within a batch.")
        features = pd.DataFrame(
            [{name: record[name] for name in FEATURE_COLUMNS} for record in records],
            columns=FEATURE_COLUMNS,
        )
        scores = positive_probabilities(self.pipeline, features)
        order = sorted(
            range(len(records)),
            key=lambda index: (-float(scores[index]), identifiers[index], index),
        )
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        return [
            {
                "record_id": identifiers[index],
                "model_score": float(scores[index]),
                "rank": ranks[index],
                "model_version": self.model_version,
            }
            for index in range(len(records))
        ]


def load_predictor(settings: ServiceSettings) -> Predictor:
    if not settings.manifest_path.is_file():
        raise ModelLoadError(f"Model manifest does not exist: {settings.manifest_path}")
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_version") != settings.expected_model_version:
        raise ModelLoadError("Configured model version does not match the manifest.")
    if manifest.get("model_sha256") != settings.expected_model_sha256:
        raise ModelLoadError("Configured model checksum does not match the manifest.")

    model_path = settings.manifest_path.parent / manifest["model_file"]
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != settings.expected_model_sha256:
        raise ModelLoadError("Model artifact checksum verification failed.")

    schema_path = settings.manifest_path.parent / manifest["feature_schema_file"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_features = tuple(item["name"] for item in schema["predictive_features"])
    if schema_features != FEATURE_COLUMNS:
        raise ModelLoadError("Deployment schema does not match the application schema.")

    # joblib artifacts can execute code while loading. Loading happens only after the
    # explicitly configured checksum above has been verified.
    pipeline = joblib.load(model_path)
    artifact_features = tuple(str(name) for name in pipeline.feature_names_in_)
    if artifact_features != FEATURE_COLUMNS:
        raise ModelLoadError("Pipeline feature schema does not match the training schema.")
    if 1 not in list(pipeline.classes_):
        raise ModelLoadError("Pipeline does not contain the defective class (1).")
    return Predictor(
        pipeline=pipeline,
        manifest=manifest,
        feature_schema=schema,
        max_batch_size=settings.max_batch_size,
    )
