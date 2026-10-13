"""
ML Engine: algorithm registry, training with proper parameter handling,
and prediction output for scatter plots.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, calinski_harabasz_score,
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    LogisticRegression, SGDClassifier,
    BayesianRidge, HuberRegressor, OrthogonalMatchingPursuit,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import (
    KMeans, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

from app.services.data_service import get_split_ratio

# ---------------------------------------------------------------------------
# Algorithm Registry
# ---------------------------------------------------------------------------

REGRESSION_ALGOS = {
    "LinearRegression": {"cls": LinearRegression, "defaults": {}, "needs_scaling": False},
    "Ridge": {"cls": Ridge, "defaults": {"alpha": 1.0}, "needs_scaling": True},
    "Lasso": {"cls": Lasso, "defaults": {"alpha": 1.0}, "needs_scaling": True},
    "ElasticNet": {"cls": ElasticNet, "defaults": {"alpha": 1.0, "l1_ratio": 0.5}, "needs_scaling": True},
    "DecisionTree": {"cls": DecisionTreeRegressor, "defaults": {"max_depth": 10}, "needs_scaling": False},
    "RandomForest": {"cls": RandomForestRegressor, "defaults": {"n_estimators": 100, "max_depth": 10}, "needs_scaling": False},
    "SVR": {"cls": SVR, "defaults": {"C": 1.0, "kernel": "rbf"}, "needs_scaling": True},
    "KNN": {"cls": KNeighborsRegressor, "defaults": {"n_neighbors": 5}, "needs_scaling": True},
    "XGBoost": {"cls": None, "defaults": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}, "needs_scaling": False},
    "LightGBM": {"cls": None, "defaults": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}, "needs_scaling": False},
    "CatBoost": {"cls": None, "defaults": {"iterations": 100, "depth": 6, "learning_rate": 0.1}, "needs_scaling": False},
    "AdaBoost": {"cls": AdaBoostRegressor, "defaults": {"n_estimators": 50, "learning_rate": 1.0}, "needs_scaling": False},
    "GBDT": {"cls": GradientBoostingRegressor, "defaults": {"n_estimators": 100, "max_depth": 3}, "needs_scaling": False},
    "MLP": {"cls": MLPRegressor, "defaults": {"hidden_layer_sizes": (100,), "max_iter": 500}, "needs_scaling": True},
    "Huber": {"cls": HuberRegressor, "defaults": {"epsilon": 1.35}, "needs_scaling": True},
    "BayesianRidge": {"cls": BayesianRidge, "defaults": {}, "needs_scaling": True},
    "SGD": {"cls": SGDRegressor, "defaults": {"max_iter": 1000}, "needs_scaling": True},
    "OrthogonalMatchingPursuit": {"cls": OrthogonalMatchingPursuit, "defaults": {}, "needs_scaling": True},
}

CLASSIFICATION_ALGOS = {
    "LogisticRegression": {"cls": LogisticRegression, "defaults": {"max_iter": 1000}, "needs_scaling": True},
    "SVM": {"cls": SVC, "defaults": {"C": 1.0, "kernel": "rbf", "probability": True}, "needs_scaling": True},
    "DecisionTree": {"cls": DecisionTreeClassifier, "defaults": {"max_depth": 10}, "needs_scaling": False},
    "RandomForest": {"cls": RandomForestClassifier, "defaults": {"n_estimators": 100, "max_depth": 10}, "needs_scaling": False},
    "KNN": {"cls": KNeighborsClassifier, "defaults": {"n_neighbors": 5}, "needs_scaling": True},
    "GaussianNB": {"cls": GaussianNB, "defaults": {}, "needs_scaling": False},
    "MultinomialNB": {"cls": MultinomialNB, "defaults": {}, "needs_scaling": False},
    "XGBoost": {"cls": None, "defaults": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}, "needs_scaling": False},
    "LightGBM": {"cls": None, "defaults": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}, "needs_scaling": False},
    "CatBoost": {"cls": None, "defaults": {"iterations": 100, "depth": 6, "learning_rate": 0.1}, "needs_scaling": False},
    "AdaBoost": {"cls": AdaBoostClassifier, "defaults": {"n_estimators": 50, "learning_rate": 1.0}, "needs_scaling": False},
    "GBDT": {"cls": GradientBoostingClassifier, "defaults": {"n_estimators": 100, "max_depth": 3}, "needs_scaling": False},
    "MLP": {"cls": MLPClassifier, "defaults": {"hidden_layer_sizes": (100,), "max_iter": 500}, "needs_scaling": True},
    "LDA": {"cls": LinearDiscriminantAnalysis, "defaults": {}, "needs_scaling": True},
    "QDA": {"cls": QuadraticDiscriminantAnalysis, "defaults": {}, "needs_scaling": True},
    "SGD": {"cls": SGDClassifier, "defaults": {"max_iter": 1000}, "needs_scaling": True},
    "ExtraTrees": {"cls": ExtraTreesClassifier, "defaults": {"n_estimators": 100, "max_depth": 10}, "needs_scaling": False},
}

UNSUPERVISED_ALGOS = {
    "KMeans": {"cls": KMeans, "defaults": {"n_clusters": 3}, "needs_scaling": True},
    "DBSCAN": {"cls": DBSCAN, "defaults": {"eps": 0.5, "min_samples": 5}, "needs_scaling": True},
    "Hierarchical": {"cls": None, "defaults": {"n_clusters": 3}, "needs_scaling": True},
    "SpectralClustering": {"cls": SpectralClustering, "defaults": {"n_clusters": 3}, "needs_scaling": True},
    "GMM": {"cls": GaussianMixture, "defaults": {"n_components": 3}, "needs_scaling": False},
    "PCA": {"cls": PCA, "defaults": {"n_components": 2}, "needs_scaling": True},
    "TSNE": {"cls": None, "defaults": {"n_components": 2, "perplexity": 30}, "needs_scaling": True},
    "IsolationForest": {"cls": IsolationForest, "defaults": {"n_estimators": 100, "contamination": 0.1}, "needs_scaling": False},
    "OneClassSVM": {"cls": OneClassSVM, "defaults": {"kernel": "rbf", "nu": 0.1}, "needs_scaling": True},
    "LOF": {"cls": LocalOutlierFactor, "defaults": {"n_neighbors": 20, "novelty": True}, "needs_scaling": True},
    "AffinityPropagation": {"cls": AffinityPropagation, "defaults": {"damping": 0.5}, "needs_scaling": True},
    "MeanShift": {"cls": MeanShift, "defaults": {}, "needs_scaling": True},
}

ALL_ALGOS = {
    "regression": REGRESSION_ALGOS,
    "classification": CLASSIFICATION_ALGOS,
    "unsupervised": UNSUPERVISED_ALGOS,
}


def _get_algo_class(name: str, task_type: str):
    entry = ALL_ALGOS[task_type][name]
    if entry["cls"] is not None:
        return entry["cls"]
    if name == "XGBoost":
        from xgboost import XGBRegressor, XGBClassifier
        return XGBRegressor if task_type == "regression" else XGBClassifier
    if name == "LightGBM":
        from lightgbm import LGBMRegressor, LGBMClassifier
        return LGBMRegressor if task_type == "regression" else LGBMClassifier
    if name == "CatBoost":
        from catboost import CatBoostRegressor, CatBoostClassifier
        return CatBoostRegressor if task_type == "regression" else CatBoostClassifier
    if name == "Hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering
    if name == "TSNE":
        from sklearn.manifold import TSNE as _TSNE
        return _TSNE
    raise ValueError(f"Unknown algorithm: {name}")


def _validate_params(algo_cls, params: dict) -> dict:
    """Validate params against estimator's get_params(), not inspect.signature.

    This fixes the XGBoost bug where inspect.signature only shows
    objective and **kwargs, causing all useful params to be dropped.
    """
    try:
        instance = algo_cls()
        valid_keys = set(instance.get_params(deep=True).keys())
    except Exception:
        return params

    validated = {}
    rejected = []
    for k, v in params.items():
        if k in valid_keys:
            validated[k] = v
        else:
            rejected.append(k)

    if rejected:
        raise ValueError(f"Unknown parameters for {algo_cls.__name__}: {rejected}. Valid: {sorted(valid_keys)[:20]}")
    return validated


def _prepare_data(df: pd.DataFrame, features: list[str], target: str, task_type: str):
    """Prepare data with proper Pipeline-based preprocessing."""
    X = df[features].copy()
    y = df[target].copy() if target else None

    # Encode categorical columns
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Fill NaN with median
    numeric_fill = X.median(numeric_only=True)
    X = X.fillna(numeric_fill)

    if task_type == "unsupervised":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, None, None, None, scaler, None, None

    # Encode target if classification
    target_le = None
    if task_type == "classification" and y.dtype == "object":
        target_le = LabelEncoder()
        y = pd.Series(target_le.fit_transform(y.astype(str)), name=target)

    test_size, split_desc = get_split_ratio(len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, None, split_desc, target_le


def _compute_regression_metrics(y_true, y_pred) -> dict:
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else None
    return {
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": len(y_true),
    }


def _compute_metrics(y_true, y_pred, task_type: str) -> dict:
    if task_type == "regression":
        return _compute_regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        avg = "weighted" if len(np.unique(y_true)) > 2 else "binary"
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
            "n": len(y_true),
        }
        return metrics
    return {}


def train_model(df, features, target, task_type, algorithm, params=None):
    """Train a model and return metrics + predictions for scatter plots."""
    X_train, X_test, y_train, y_test, scaler, split_desc, target_le = _prepare_data(
        df, features, target, task_type
    )

    algo_cls = _get_algo_class(algorithm, task_type)
    algo_entry = ALL_ALGOS[task_type][algorithm]
    merged_params = {**algo_entry["defaults"], **(params or {})}

    # CRITICAL FIX: use get_params() validation instead of inspect.signature
    effective_params = _validate_params(algo_cls, merged_params)

    model = algo_cls(**effective_params)

    if task_type == "unsupervised":
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_train)
        else:
            model.fit(X_train)
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)

        metrics = {}
        n_labels = len(set(labels)) - (1 if -1 in labels else 0)
        if n_labels >= 2:
            mask = labels != -1
            if mask.sum() >= 2:
                metrics["silhouette"] = float(silhouette_score(X_train[mask], labels[mask]))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_train[mask], labels[mask]))
        metrics["n_clusters"] = n_labels
        return {
            "metrics": metrics, "params": effective_params,
            "split_ratio": "unsupervised", "train_size": len(X_train), "test_size": 0,
            "train_predictions": [], "test_predictions": [],
            "train_metrics": {}, "test_metrics": {},
        }

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = _compute_metrics(y_train, y_train_pred, task_type)
    test_metrics = _compute_metrics(y_test, y_test_pred, task_type)

    # Primary metrics (for display)
    metrics = test_metrics.copy()

    # Build prediction arrays for scatter plots
    train_predictions = [
        {"y_true": float(yt), "y_pred": float(yp)}
        for yt, yp in zip(y_train.values if hasattr(y_train, 'values') else y_train,
                          y_train_pred)
    ]
    test_predictions = [
        {"y_true": float(yt), "y_pred": float(yp)}
        for yt, yp in zip(y_test.values if hasattr(y_test, 'values') else y_test,
                          y_test_pred)
    ]

    return {
        "metrics": metrics,
        "params": effective_params,
        "split_ratio": split_desc,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def get_primary_score(metrics: dict, task_type: str) -> float:
    if task_type == "regression":
        val = metrics.get("r2")
        return val if val is not None else -float("inf")
    elif task_type == "classification":
        return metrics.get("f1", 0.0)
    return metrics.get("silhouette", 0.0)
