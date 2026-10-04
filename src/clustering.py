"""Data-driven K-means segmentation using available numeric customer features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .feature_engineering import modeling_features


def segment_customers(
    frame: pd.DataFrame,
    target: str,
    output_dir: str | Path,
    random_state: int = 42,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = modeling_features(frame, target).select_dtypes(include="number")
    candidates = candidates.dropna(axis=1, how="all")
    candidates = candidates.loc[:, candidates.nunique(dropna=True) > 1]
    if candidates.shape[1] == 0 or len(frame) < 3:
        return {"status": "not_calculated", "reason": "Insufficient varying numeric features or rows."}

    values = candidates.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median()).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    sample_size = min(len(scaled), 5000)
    rng = np.random.default_rng(random_state)
    sample_indices = np.sort(rng.choice(len(scaled), size=sample_size, replace=False))
    sample = scaled[sample_indices]
    scores = []
    upper_k = min(7, sample_size - 1)
    for clusters in range(2, upper_k + 1):
        estimator = KMeans(n_clusters=clusters, n_init=10, random_state=random_state)
        labels = estimator.fit_predict(sample)
        if len(np.unique(labels)) > 1:
            scores.append({"clusters": clusters, "silhouette": float(silhouette_score(sample, labels))})
    if not scores:
        return {"status": "not_calculated", "reason": "No valid cluster count for silhouette evaluation."}

    best = max(scores, key=lambda item: item["silhouette"])
    final_model = KMeans(n_clusters=best["clusters"], n_init=10, random_state=random_state)
    frame_with_clusters = frame.copy()
    frame_with_clusters["cluster"] = final_model.fit_predict(scaled)
    frame_with_clusters.to_csv(output_dir / "customer_segments.csv", index=False)
    joblib.dump({"model": final_model, "scaler": scaler, "features": candidates.columns.tolist()}, output_dir / "kmeans.joblib")
    target_values = frame[target].astype("string").str.strip().str.casefold()
    target_classes = sorted(target_values.dropna().unique().tolist())
    positive_class = next(
        (value for value in target_classes if value in {"yes", "1", "true", "subscribed"}),
        target_classes[-1] if len(target_classes) == 2 else None,
    )
    cluster_profiles = {}
    for cluster_id, group in frame_with_clusters.groupby("cluster", sort=True):
        profile = {"records": int(len(group))}
        if positive_class is not None:
            profile["target_positive_class"] = positive_class
            profile["target_positive_rate"] = float(
                target_values.loc[group.index].eq(positive_class).mean()
            )
        for column in candidates.columns:
            profile[f"mean_{column}"] = float(group[column].mean())
        cluster_profiles[str(int(cluster_id))] = profile
    return {
        "status": "calculated",
        "features": candidates.columns.tolist(),
        "candidate_scores": scores,
        "selected_clusters": int(best["clusters"]),
        "selected_silhouette": best["silhouette"],
        "cluster_sizes": frame_with_clusters["cluster"].value_counts().sort_index().astype(int).to_dict(),
        "cluster_profiles": cluster_profiles,
    }
