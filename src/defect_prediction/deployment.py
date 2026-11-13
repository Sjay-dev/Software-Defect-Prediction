"""Package a frozen evaluation pipeline as a separate deployment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


MODEL_VERSION = "logistic-regression-jedit-holdout-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_deployment_model(
    source_dir: Path, output_dir: Path, model_version: str = MODEL_VERSION
) -> dict[str, Any]:
    """Copy, never refit, a known pipeline and record its exact provenance."""
    source_model = source_dir / "logistic_pipeline.joblib"
    source_config = json.loads(
        (source_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    source_schema = source_dir / "feature_schema.json"
    source_hash = _sha256(source_model)
    manifest = {
        "model_version": model_version,
        "model_file": "pipeline.joblib",
        "model_sha256": source_hash,
        "feature_schema_file": "feature_schema.json",
        "model_type": "scikit-learn logistic-regression pipeline with saved preprocessing",
        "score_description": "Relative defect-risk model score; not a calibrated probability.",
        "training_projects": source_config["training_projects"],
        "evaluation_only_projects": source_config["test_projects"],
        "training_experiment": source_config["experiment_name"],
        "training_tuning": source_config["tuning"],
        "preprocessing_fit_scope": source_config["preprocessing_fit_scope"],
        "source_artifact": str(source_model),
        "packaging_method": "Byte-for-byte copy of the saved pipeline; no fitting or retraining.",
    }
    manifest_path = output_dir / "model_manifest.json"
    target_model = output_dir / manifest["model_file"]
    target_schema = output_dir / manifest["feature_schema_file"]

    if output_dir.exists():
        if not (manifest_path.exists() and target_model.exists() and target_schema.exists()):
            raise FileExistsError(
                f"Refusing to overwrite incomplete deployment directory: {output_dir}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest or _sha256(target_model) != source_hash:
            raise FileExistsError(
                f"Refusing to overwrite a different deployment artifact: {output_dir}"
            )
        return existing

    output_dir.mkdir(parents=True)
    shutil.copyfile(source_model, target_model)
    shutil.copyfile(source_schema, target_schema)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if _sha256(target_model) != source_hash:
        raise RuntimeError("Deployment copy checksum does not match its source artifact.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"artifacts/deployment/{MODEL_VERSION}"),
    )
    parser.add_argument("--model-version", default=MODEL_VERSION)
    args = parser.parse_args()
    print(
        json.dumps(
            package_deployment_model(args.source_dir, args.output_dir, args.model_version),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
