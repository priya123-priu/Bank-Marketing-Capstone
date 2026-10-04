"""Small, conditional features for the bank marketing schema."""

from __future__ import annotations

import pandas as pd


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "age" in result.columns:
        age = pd.to_numeric(result["age"], errors="coerce")
        result["age_group"] = pd.cut(
            age,
            bins=[0, 25, 35, 45, 55, 65, float("inf")],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
            include_lowest=True,
        ).astype("object")
    if "campaign" in result.columns:
        contacts = pd.to_numeric(result["campaign"], errors="coerce")
        result["campaign_contact_band"] = pd.cut(
            contacts,
            bins=[-float("inf"), 1, 3, 5, float("inf")],
            labels=["1", "2-3", "4-5", "6+"],
        ).astype("object")
    if "pdays" in result.columns:
        days = pd.to_numeric(result["pdays"], errors="coerce")
        result["previously_contacted"] = days.ne(-1).where(days.notna()).astype("boolean")
    return result


def modeling_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Exclude the target and post-contact duration to reduce leakage for targeting."""
    excluded = {target, "duration"}
    return frame.drop(columns=[column for column in excluded if column in frame.columns])
