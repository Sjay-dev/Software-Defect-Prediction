"""FastAPI routes for the frozen software-defect scoring pipeline."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from .historical_results import load_research_results
from .prediction import Predictor, load_predictor, load_service_settings
from .schema import FEATURE_COLUMNS


DEFAULT_CONFIG_PATH = Path("config/service.json")
ABSOLUTE_MAX_BATCH_SIZE = 100


class _RecordIdentifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: Annotated[str, Field(min_length=1, max_length=200)]

    @field_validator("record_id")
    @classmethod
    def identifier_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("record_id must not be blank")
        return value


MetricRecord = create_model(
    "MetricRecord",
    __base__=_RecordIdentifier,
    **{
        name: (Annotated[float | None, Field(allow_inf_nan=False)], ...)
        for name in FEATURE_COLUMNS
    },
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: Annotated[
        list[MetricRecord], Field(min_length=1, max_length=ABSOLUTE_MAX_BATCH_SIZE)
    ]

    @field_validator("records")
    @classmethod
    def identifiers_must_be_unique(cls, records: list[Any]) -> list[Any]:
        identifiers = [record.record_id for record in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("record_id values must be unique within a batch")
        return records


class PredictionResult(BaseModel):
    record_id: str
    model_score: Annotated[
        float,
        Field(description="Relative defect-risk model score; not a calibrated probability."),
    ]
    rank: Annotated[int, Field(ge=1)]
    model_version: str


class PredictResponse(BaseModel):
    score_description: str
    predictions: list[PredictionResult]


def _configured_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return Path(os.environ.get("DEFECT_SERVICE_CONFIG", DEFAULT_CONFIG_PATH))


def create_app(config_path: Path | None = None) -> FastAPI:
    settings = load_service_settings(_configured_path(config_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = load_predictor(settings)
        app.state.historical_results = load_research_results(
            settings.baseline_summary_path,
            settings.model_comparison_summary_path,
            settings.source_similarity_summary_path,
        )
        yield

    application = FastAPI(
        title="Software Defect Prediction Service",
        version="1.0.0",
        description=(
            "Ranks metric records with a frozen cross-project model. Scores are model "
            "scores, not calibrated probabilities."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def predictor(request: Request) -> Predictor:
        return request.app.state.predictor

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        loaded = predictor(request)
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": loaded.model_version,
        }

    @application.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        return predictor(request).model_info()

    @application.get("/experiment-results")
    def experiment_results(request: Request) -> dict[str, Any]:
        results = request.app.state.historical_results
        return results

    @application.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
        loaded = predictor(request)
        if len(payload.records) > loaded.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"Batch exceeds configured limit of {loaded.max_batch_size}.",
            )
        records = [record.model_dump() for record in payload.records]
        return {
            "score_description": loaded.manifest["score_description"],
            "predictions": loaded.predict(records),
        }

    return application


app = create_app()
