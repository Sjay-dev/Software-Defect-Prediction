"""Create an isolated service config for a model reconstructed in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_from(path: Path, configured: str) -> Path:
    value = Path(configured)
    return value.resolve() if value.is_absolute() else (path.parent / value).resolve()


def write_verified_ci_service_config(
    base_config_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, str]:
    """Pin an isolated config to the verified bytes produced by the CI run."""
    base_config_path = base_config_path.resolve()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()

    config = _read_json(base_config_path)
    manifest = _read_json(manifest_path)
    model_path = _resolve_from(manifest_path, str(manifest["model_file"]))
    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_sha256 = str(manifest["model_sha256"])
    if actual_sha256 != manifest_sha256:
        raise RuntimeError(
            "Generated model checksum does not match its deployment manifest: "
            f"manifest={manifest_sha256}, actual={actual_sha256}"
        )
    if manifest.get("model_version") != config.get("expected_model_version"):
        raise RuntimeError(
            "Generated model version does not match the base service configuration."
        )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite CI service config: {output_path}")

    # The config lives outside the checkout, so preserve every relative report path
    # as an absolute path rooted at the tracked base config.
    reports = config.get("historical_reports", {})
    config["historical_reports"] = {
        name: str(_resolve_from(base_config_path, str(value)))
        for name, value in reports.items()
        if value is not None
    }
    config["model_manifest"] = str(manifest_path)
    config["expected_model_sha256"] = actual_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "configured_sha256": str(config["expected_model_sha256"]),
        "manifest_sha256": manifest_sha256,
        "actual_sha256": actual_sha256,
        "service_config": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", type=Path, default=Path("config/service.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/deployment/logistic-regression-jedit-holdout-v1/"
            "model_manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_verified_ci_service_config(
                args.base_config, args.manifest, args.output
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
