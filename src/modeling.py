"""Leakage-aware binary classification with preprocessing inside each pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import classification_metrics
from .feature_engineering import modeling_features


def train_models(frame: pd.DataFrame, target: str, output_dir: str | Path, random_state: int = 42) -> dict:
    if target not in frame:
        raise ValueError(f"Target column '{target}' is not present. Available columns: {list(frame.columns)}")
    data = frame.loc[frame[target].notna()].copy()
    labels = data[target].astype("string").str.strip().str.casefold()
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Binary classification requires exactly two target classes; found {classes}.")
    positive = next((value for value in classes if value in {"yes", "1", "true", "subscribed"}), classes[-1])
    y = labels.eq(positive).astype(int)
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two observations for a stratified split.")

    X = modeling_features(data, target)
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.columns.difference(numeric_columns).tolist()
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No usable predictor columns remain after excluding the target and duration.")
    preprocessing = ColumnTransformer(transformers, remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        ),
    }
    results = {}
    fitted = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        results[name] = classification_metrics(y_test, predictions, probabilities)
        fitted[name] = pipeline

    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["roc_auc"] or 0))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], output_dir / "best_model.joblib")
    return {
        "status": "calculated",
        "target": target,
        "classes": classes,
        "positive_class": positive,
        "excluded_features": [column for column in ["duration"] if column in data.columns],
        "test_rows": int(len(y_test)),
        "metrics": results,
        "selected_model": best_name,
        "saved_model": str(output_dir / "best_model.joblib"),
    }
