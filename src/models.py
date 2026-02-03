from __future__ import annotations

import inspect
from typing import Any

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline

MODEL_ALIASES = {
    "hist_gb": "hgb",
    "hist_gradient_boosting": "hgb",
    "histgradientboosting": "hgb",
    "lgb": "lgbm",
    "lightgbm": "lgbm",
    "xgboost": "xgb",
    "catboost": "cat",
    "random_forest": "rf",
    "extra_trees": "extra_trees",
    "extratrees": "extra_trees",
}

DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "ridge": {
        "alpha": 1.0,
    },
    "elasticnet": {
        "alpha": 1.0,
        "l1_ratio": 0.1,
    },
    "hgb": {
        "max_depth": 8,
        "learning_rate": 0.05,
        "max_iter": 300,
    },
    "rf": {
        "n_estimators": 300,
        "max_depth": 16,
        "n_jobs": -1,
        "random_state": 42,
    },
    "extra_trees": {
        "n_estimators": 300,
        "max_depth": 16,
        "n_jobs": -1,
        "random_state": 42,
    },
    "lgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgb": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    },
    "cat": {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
    },
}


def normalize_model_name(name: str | None) -> str:
    if not name:
        return "ridge"
    key = name.strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(key, key)


def resolve_model_params(
    model_name: str | None,
    model_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    name = normalize_model_name(model_name)
    params = dict(DEFAULT_MODEL_PARAMS.get(name, {}))
    if model_params:
        params.update(model_params)
    return name, params


def _build_estimator(model_name: str, params: dict[str, Any]):
    if model_name == "ridge":
        return Ridge(**params)
    if model_name == "elasticnet":
        return ElasticNet(**params)
    if model_name == "hgb":
        return HistGradientBoostingRegressor(**params)
    if model_name == "rf":
        return RandomForestRegressor(**params)
    if model_name == "extra_trees":
        return ExtraTreesRegressor(**params)
    if model_name == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "lightgbm is not installed. Add it to requirements.md or pip install lightgbm."
            ) from exc
        return lgb.LGBMRegressor(**params)
    if model_name == "xgb":
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is not installed. Add it to requirements.md or pip install xgboost."
            ) from exc
        return xgb.XGBRegressor(**params)
    if model_name == "cat":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "catboost is not installed. Add it to requirements.md or pip install catboost."
            ) from exc
        return CatBoostRegressor(**params)
    raise ValueError(f"Unknown model '{model_name}'. Supported: {', '.join(sorted(DEFAULT_MODEL_PARAMS))}")


def build_model(
    model_name: str | None,
    model_params: dict[str, Any] | None = None,
    impute: bool = True,
) -> Pipeline:
    name, params = resolve_model_params(model_name, model_params)
    estimator = _build_estimator(name, params)
    if impute:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    return estimator


def supports_sample_weight(model) -> bool:
    estimator = model
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    try:
        sig = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False
    return "sample_weight" in sig.parameters


def build_fit_params(model, sample_weight):
    if sample_weight is None:
        return {}
    if not supports_sample_weight(model):
        return {}
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        return {"model__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def supported_models() -> list[str]:
    return sorted(DEFAULT_MODEL_PARAMS)
