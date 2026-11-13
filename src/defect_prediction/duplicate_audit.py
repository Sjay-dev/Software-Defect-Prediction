"""Investigate duplicate-like overlap between project-level train/test splits."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .schema import (
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMN,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


def _signatures(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    column_list = list(columns)
    normalized = frame.loc[:, column_list].astype("object")
    normalized = normalized.where(normalized.notna(), "<MISSING>")
    return pd.Series(
        list(normalized.itertuples(index=False, name=None)), index=frame.index
    )


def _example_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_signatures: pd.Series,
    test_signatures: pd.Series,
    *,
    conflicting_target: bool = False,
) -> list[dict[str, Any]]:
    train_lookup: dict[tuple[Any, ...], list[Any]] = {}
    for index, signature in train_signatures.items():
        train_lookup.setdefault(signature, []).append(index)

    examples: list[dict[str, Any]] = []
    for test_index, signature in test_signatures.items():
        candidates = train_lookup.get(signature, [])
        if conflicting_target:
            candidates = [
                index
                for index in candidates
                if int(train.at[index, TARGET_COLUMN])
                != int(test.at[test_index, TARGET_COLUMN])
            ]
        if not candidates:
            continue
        train_index = candidates[0]
        examples.append(
            {
                "train": {
                    column: train.at[train_index, column] for column in METADATA_COLUMNS
                },
                "test": {
                    column: test.at[test_index, column] for column in METADATA_COLUMNS
                },
                "train_bug_count": int(train.at[train_index, LABEL_SOURCE_COLUMN]),
                "test_bug_count": int(test.at[test_index, LABEL_SOURCE_COLUMN]),
                "train_target": int(train.at[train_index, TARGET_COLUMN]),
                "test_target": int(test.at[test_index, TARGET_COLUMN]),
            }
        )
        if len(examples) == 3:
            break
    return examples


def audit_cross_split_duplicates(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    """Count several kinds of overlap without assuming metric matches are code copies."""
    train_ids = set(_signatures(train, METADATA_COLUMNS))
    test_ids = _signatures(test, METADATA_COLUMNS)

    train_names = set(train["class_name"])
    class_name_matches = test["class_name"].isin(train_names)

    train_features = _signatures(train, FEATURE_COLUMNS)
    test_features = _signatures(test, FEATURE_COLUMNS)
    train_feature_set = set(train_features)
    feature_matches = test_features.isin(train_feature_set)

    feature_target_columns = (*FEATURE_COLUMNS, TARGET_COLUMN)
    train_feature_targets = _signatures(train, feature_target_columns)
    test_feature_targets = _signatures(test, feature_target_columns)
    feature_target_matches = test_feature_targets.isin(set(train_feature_targets))

    feature_bug_columns = (*FEATURE_COLUMNS, LABEL_SOURCE_COLUMN)
    train_feature_bugs = _signatures(train, feature_bug_columns)
    test_feature_bugs = _signatures(test, feature_bug_columns)
    feature_bug_matches = test_feature_bugs.isin(set(train_feature_bugs))

    train_targets_by_feature: dict[tuple[Any, ...], set[int]] = {}
    for index, signature in train_features.items():
        train_targets_by_feature.setdefault(signature, set()).add(
            int(train.at[index, TARGET_COLUMN])
        )
    conflicting = pd.Series(
        [
            (1 - int(test.at[index, TARGET_COLUMN]))
            in train_targets_by_feature.get(signature, set())
            for index, signature in test_features.items()
        ],
        index=test.index,
    )

    test_rows = len(test)
    feature_match_rows = int(feature_matches.sum())
    return {
        "definition_note": (
            "A metric-vector match means all 20 predictors are numerically identical; "
            "it does not by itself prove that two rows are copies of the same source code."
        ),
        "test_rows": test_rows,
        "exact_identifier_match_rows": int(test_ids.isin(train_ids).sum()),
        "same_class_name_match_rows": int(class_name_matches.sum()),
        "same_class_name_unique_names": int(
            test.loc[class_name_matches, "class_name"].nunique()
        ),
        "exact_feature_vector_match_rows": feature_match_rows,
        "exact_feature_vector_match_percent": (
            100.0 * feature_match_rows / test_rows if test_rows else 0.0
        ),
        "exact_feature_and_binary_target_match_rows": int(
            feature_target_matches.sum()
        ),
        "exact_feature_and_bug_count_match_rows": int(feature_bug_matches.sum()),
        "feature_match_with_conflicting_target_rows": int(conflicting.sum()),
        "feature_match_examples": _example_pair(
            train, test, train_features, test_features
        ),
        "conflicting_target_examples": _example_pair(
            train,
            test,
            train_features,
            test_features,
            conflicting_target=True,
        ),
    }

