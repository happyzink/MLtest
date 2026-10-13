"""Exploratory Data Analysis service: correlation, feature importance, mutual information."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def compute_correlations(df: pd.DataFrame, features: list[str]) -> dict:
    """Compute Pearson and Spearman correlation matrices for numeric features."""
    numeric_df = df[features].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"pearson": {}, "spearman": {}}

    pearson = numeric_df.corr(method="pearson").round(4).to_dict()
    spearman = numeric_df.corr(method="spearman").round(4).to_dict()
    return {"pearson": pearson, "spearman": spearman}


def compute_feature_importance(df: pd.DataFrame, features: list[str], target: str, task_type: str) -> dict:
    """Compute tree-based feature importance using a quick Random Forest."""
    X = df[features].copy()
    y = df[target].copy()

    # Encode categoricals
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    X = X.fillna(X.median(numeric_only=True))

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)

    model.fit(X, y)
    importances = model.feature_importances_
    return {feat: round(float(imp), 4) for feat, imp in zip(features, importances)}


def compute_mutual_info(df: pd.DataFrame, features: list[str], target: str, task_type: str) -> dict:
    """Compute mutual information scores between features and target."""
    X = df[features].copy()
    y = df[target].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    X = X.fillna(X.median(numeric_only=True))

    if task_type == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    return {feat: round(float(mi), 4) for feat, mi in zip(features, mi_scores)}
