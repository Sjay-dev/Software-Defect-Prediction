"""Download-independent loading, normalization, and validation of source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from .schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    NORMALIZED_COLUMNS,
    SOURCE_HEADER,
    TARGET_COLUMN,
)


class DataValidationError(ValueError):
    """Raised when source data cannot safely be used by the experiment."""


_MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none"}


def _read_release(path: Path, canonical_project: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = tuple(next(csv.reader(source)))
    if header != SOURCE_HEADER:
        raise DataValidationError(
            f"{path.name} has an unexpected header. Expected {list(SOURCE_HEADER)}, "
            f"got {list(header)}."
        )

    frame = pd.read_csv(path)
    if len(frame.columns) != len(SOURCE_HEADER):
        raise DataValidationError(f"{path.name} has the wrong number of columns.")
    frame.columns = [*METADATA_COLUMNS, *FEATURE_COLUMNS, LABEL_SOURCE_COLUMN]

    source_projects = frame["project"].dropna().astype(str).str.strip()
    if source_projects.empty:
        raise DataValidationError(f"{path.name} has no project identifier.")
    frame["project"] = canonical_project
    return frame


def _coerce_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        raw = result[column]
        converted = pd.to_numeric(raw, errors="coerce")
        normalized = raw.astype("string").str.strip().str.lower()
        invalid = raw.notna() & converted.isna() & ~normalized.isin(_MISSING_TOKENS)
        if invalid.any():
            examples = raw.loc[invalid].astype(str).head(3).tolist()
            raise DataValidationError(
                f"Column '{column}' contains non-numeric values, for example {examples}."
            )
        result[column] = converted
    return result


def validate_dataset(
    frame: pd.DataFrame, *, require_multiple_projects: bool = True
) -> pd.DataFrame:
    """Return a validated copy or raise a clear error describing unsafe data."""
    required = {*METADATA_COLUMNS, *FEATURE_COLUMNS, LABEL_SOURCE_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("The dataset has no rows.")

    result = frame.copy()
    for column in METADATA_COLUMNS:
        blank = result[column].isna() | result[column].astype(str).str.strip().eq("")
        if blank.any():
            raise DataValidationError(f"Metadata column '{column}' contains missing values.")
        result[column] = result[column].astype(str).str.strip()

    if require_multiple_projects and result["project"].nunique() < 2:
        raise DataValidationError("Cross-project evaluation requires at least two projects.")

    duplicated = result.duplicated(list(METADATA_COLUMNS), keep=False)
    if duplicated.any():
        sample = result.loc[duplicated, list(METADATA_COLUMNS)].head(3).to_dict("records")
        raise DataValidationError(f"Duplicate project/release/class identifiers: {sample}")

    result = _coerce_numeric(result, (*FEATURE_COLUMNS, LABEL_SOURCE_COLUMN))
    if result[LABEL_SOURCE_COLUMN].isna().any():
        raise DataValidationError("bug_count contains missing values; labels cannot be inferred.")
    if (result[LABEL_SOURCE_COLUMN] < 0).any():
        raise DataValidationError("bug_count must be non-negative.")
    if (result[LABEL_SOURCE_COLUMN] % 1 != 0).any():
        raise DataValidationError("bug_count must contain whole-number counts.")
    if result[list(FEATURE_COLUMNS)].isna().all(axis=0).any():
        empty = result[list(FEATURE_COLUMNS)].columns[
            result[list(FEATURE_COLUMNS)].isna().all(axis=0)
        ].tolist()
        raise DataValidationError(f"Feature columns are entirely missing: {empty}")

    inferred_target = (result[LABEL_SOURCE_COLUMN] > 0).astype("int8")
    if TARGET_COLUMN in result:
        existing = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
        if existing.isna().any() or not existing.isin([0, 1]).all():
            raise DataValidationError("defective must contain only binary values 0 and 1.")
        if not existing.astype("int8").equals(inferred_target):
            raise DataValidationError(
                "defective is inconsistent with the rule bug_count > 0."
            )
    result[TARGET_COLUMN] = inferred_target
    return result.loc[:, NORMALIZED_COLUMNS]


def load_dataset(data_dir: Path, manifest_path: Path) -> pd.DataFrame:
    """Load every release pinned by the manifest and validate the combined table."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    releases: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for record in manifest["records"]:
        for filename in record["files"]:
            path = data_dir / filename
            if not path.exists():
                missing_files.append(filename)
                continue
            releases.append(_read_release(path, record["project"]))
    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} dataset files are missing ({preview}). "
            "Run: python -m defect_prediction.download"
        )
    return validate_dataset(pd.concat(releases, ignore_index=True))

