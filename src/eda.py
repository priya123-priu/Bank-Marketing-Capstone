"""Dataset profiling and compact, reproducible exploratory plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def profile_dataset(frame: pd.DataFrame, output_dir: str | Path, target: str | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = frame.select_dtypes(include="number")
    profile = {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": list(frame.columns),
        "data_types": frame.dtypes.astype(str).to_dict(),
        "missing_values_by_column": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_values_by_column": frame.nunique(dropna=False).astype(int).to_dict(),
        "numeric_summary": numeric.describe().to_dict(),
        "target": target if target in frame.columns else None,
        "target_distribution": (
            frame[target].astype("string").value_counts(dropna=False).astype(int).to_dict()
            if target in frame.columns else None
        ),
    }
    (output_dir / "data_profile.json").write_text(
        json.dumps(profile, indent=2, default=str), encoding="utf-8"
    )

    sns.set_theme(style="whitegrid")
    if target in frame.columns:
        fig, axis = plt.subplots(figsize=(7, 4))
        order = frame[target].astype("string").value_counts().index
        sns.countplot(data=frame, x=target, order=order, ax=axis, color="#2f7c78")
        axis.set(title=f"Distribution of {target}", xlabel=target, ylabel="Records")
        fig.tight_layout()
        fig.savefig(output_dir / "target_distribution.png", dpi=150)
        plt.close(fig)

    if not numeric.empty:
        numeric.hist(figsize=(12, 8), bins=30, color="#4b8f8a", edgecolor="white")
        plt.tight_layout()
        plt.savefig(output_dir / "numeric_distributions.png", dpi=150)
        plt.close()

    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    categorical = [column for column in categorical if column != target and frame[column].nunique() <= 20]
    for column in categorical[:8]:
        counts = frame[column].astype("string").fillna("<missing>").value_counts().head(15)
        fig, axis = plt.subplots(figsize=(8, max(3, len(counts) * 0.28)))
        counts.sort_values().plot.barh(ax=axis, color="#d6864b")
        axis.set(title=f"{column}: most frequent values", xlabel="Records", ylabel=column)
        fig.tight_layout()
        fig.savefig(output_dir / f"category_{column}.png", dpi=150)
        plt.close(fig)
    return profile
