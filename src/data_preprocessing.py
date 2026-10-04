"""Schema-adaptive loading and first-pass data quality checks."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class DatasetError(ValueError):
    """Raised when the supplied dataset cannot support analysis."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a delimited text dataset, choosing the separator with most columns."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    if path.stat().st_size == 0:
        raise DatasetError(
            f"Dataset is empty (0 bytes): {path}. Add the actual CSV before running analysis."
        )

    candidates = []
    errors = []
    for separator in (",", ";", "\t", "|"):
        try:
            frame = pd.read_csv(path, sep=separator, low_memory=False)
            candidates.append(frame)
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
            errors.append(str(exc))
    if not candidates:
        raise DatasetError(f"Could not parse dataset {path}. Parser errors: {'; '.join(errors)}")

    frame = max(candidates, key=lambda item: item.shape[1])
    if frame.empty or frame.shape[1] < 2:
        raise DatasetError(
            f"Dataset parsed as {frame.shape[0]} rows and {frame.shape[1]} column(s). "
            "Check that this is the intended delimited data file."
        )

    frame.columns = [_normalize_column(column) for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise DatasetError("Column names are not unique after normalization.")
    return frame


def _normalize_column(value: object) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return name or "unnamed_column"


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows and report missing/unknown values without hiding them."""
    cleaned = frame.copy()
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    unknown_counts = {}
    for column in cleaned.select_dtypes(include=["object", "string", "category"]).columns:
        count = int(cleaned[column].astype("string").str.strip().str.casefold().eq("unknown").sum())
        if count:
            unknown_counts[column] = count

    domain_rules = {
        "age": lambda values: values.le(0) | values.gt(120),
        "day": lambda values: values.lt(1) | values.gt(31),
        "campaign": lambda values: values.lt(1),
        "pdays": lambda values: values.lt(-1),
        "previous": lambda values: values.lt(0),
        "duration": lambda values: values.lt(0),
    }
    invalid_counts = {}
    for column, rule in domain_rules.items():
        if column in cleaned:
            values = pd.to_numeric(cleaned[column], errors="coerce")
            count = int(rule(values).fillna(False).sum())
            if count:
                invalid_counts[column] = count

    iqr_outlier_counts = {}
    for column in cleaned.select_dtypes(include="number").columns:
        values = cleaned[column].dropna()
        if values.empty:
            continue
        lower_quartile, upper_quartile = values.quantile([0.25, 0.75])
        interquartile_range = upper_quartile - lower_quartile
        if interquartile_range > 0:
            lower_bound = lower_quartile - 1.5 * interquartile_range
            upper_bound = upper_quartile + 1.5 * interquartile_range
            count = int(((values < lower_bound) | (values > upper_bound)).sum())
            if count:
                iqr_outlier_counts[column] = count

    quality = {
        "rows_before_deduplication": int(len(frame)),
        "duplicate_rows_removed": duplicate_count,
        "rows_after_deduplication": int(len(cleaned)),
        "missing_values_by_column": cleaned.isna().sum().astype(int).to_dict(),
        "unknown_values_by_column": unknown_counts,
        "invalid_values_by_column": invalid_counts,
        "potential_iqr_outliers_by_column": iqr_outlier_counts,
        "data_types": cleaned.dtypes.astype(str).to_dict(),
    }
    return cleaned, quality
