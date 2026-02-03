from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from src.config import load_json_config, resolve_arg
from src.data import FEATURE_PREFIX, TARGET_COL, WEIGHT_COL, get_data_dir, load_train
from src.metrics import weighted_zero_mean_r2
from src.models import build_fit_params, build_model, resolve_model_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory.")
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Number of train parquet parts to load (0 loads all parts).",
    )
    parser.add_argument("--valid-days", type=int, default=None, help="Number of date_id values to use for validation.")
    parser.add_argument("--alpha", type=float, default=None, help="Ridge regularization strength.")
    parser.add_argument("--feature-prefix", type=str, default=None)
    parser.add_argument("--feature-list", type=str, default=None, help="Path to a file listing feature columns.")
    parser.add_argument("--drop-all-null-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--weight-col", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Model name (ridge, hgb, lgbm, xgb, cat, rf, extra_trees).")
    parser.add_argument(
        "--model-params",
        type=str,
        default=None,
        help="JSON string or path to a JSON file with model params.",
    )
    parser.add_argument("--model-out", type=str, default=None, help="Optional path to save the model.")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers for parquet loading.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for metrics/plots.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot-sample", type=int, default=None, help="Max rows for scatter plot.")
    return parser.parse_args()


def time_split(df: pd.DataFrame, date_col: str, valid_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df[date_col].unique())
    if valid_days <= 0 or valid_days >= len(unique_dates):
        raise ValueError("valid_days must be > 0 and less than the number of unique dates")
    valid_date_set = set(unique_dates[-valid_days:])
    is_valid = df[date_col].isin(valid_date_set)
    return df.loc[~is_valid].copy(), df.loc[is_valid].copy()


def load_feature_list(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    parts = [item.strip() for item in text.replace(",", "\n").splitlines()]
    return [item for item in parts if item]


def parse_model_params(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("model-params must be JSON or a path to a JSON file") from exc
    raise TypeError("model-params must be a dict, JSON string, or path to JSON file")


def save_model(
    model,
    feature_cols: list[str],
    model_name: str,
    model_params: dict[str, Any],
    target: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "features": feature_cols,
        "model_name": model_name,
        "model_params": model_params,
        "target": target,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def train_model(
    data_dir: Path,
    max_parts: int | None,
    num_workers: int,
    show_progress: bool,
    feature_prefix: str,
    feature_list: str | None,
    drop_all_null: bool,
    target: str,
    weight_col: str,
    valid_days: int,
    model_name: str,
    model_params: dict[str, Any] | None,
):
    df = load_train(
        data_dir=data_dir,
        max_parts=max_parts,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    if feature_list:
        feature_cols = load_feature_list(feature_list)
    else:
        feature_cols = sorted([col for col in df.columns if col.startswith(feature_prefix)])

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"Warning: dropping {len(missing)} feature columns not present in data.")
        feature_cols = [col for col in feature_cols if col in df.columns]

    if not feature_cols:
        raise ValueError("No feature columns found. Check feature prefix or feature list.")

    df = df.dropna(subset=[target, weight_col])

    train_df, valid_df = time_split(df, date_col="date_id", valid_days=valid_days)

    if drop_all_null:
        non_null_mask = train_df[feature_cols].notna().any(axis=0)
        dropped = [col for col, keep in non_null_mask.items() if not keep]
        feature_cols = [col for col in feature_cols if non_null_mask[col]]
        if dropped:
            print(f"Dropped {len(dropped)} all-null features: {', '.join(dropped)}")

    model = build_model(model_name, model_params)

    fit_params = build_fit_params(model, train_df[weight_col].to_numpy())
    model.fit(train_df[feature_cols], train_df[target], **fit_params)

    valid_pred = model.predict(valid_df[feature_cols])
    score = weighted_zero_mean_r2(valid_df[target], valid_pred, valid_df[weight_col])

    return {
        "model": model,
        "feature_cols": feature_cols,
        "score": score,
        "valid_df": valid_df,
        "valid_pred": valid_pred,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }


def save_scatter(y_true, y_pred, output_path: Path, sample: int, rng_seed: int = 42) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample is not None and len(y_true) > sample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(y_true), size=sample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    combined = np.concatenate([y_true, y_pred])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return

    low, high = np.nanpercentile(finite, [1, 99])
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=4, alpha=0.2)
    plt.plot([low, high], [low, high], "--", color="tab:red", linewidth=1)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Validation predictions")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)

    data_dir_value = resolve_arg(args.data_dir, config, "data_dir", None)
    data_dir = Path(data_dir_value) if data_dir_value else get_data_dir()
    max_parts = resolve_arg(args.max_parts, config, "max_parts", 2)
    valid_days = resolve_arg(args.valid_days, config, "valid_days", 5)
    alpha = resolve_arg(args.alpha, config, "alpha", 1.0)
    feature_prefix = resolve_arg(args.feature_prefix, config, "feature_prefix", FEATURE_PREFIX)
    feature_list = resolve_arg(args.feature_list, config, "feature_list", None)
    drop_all_null = resolve_arg(args.drop_all_null_features, config, "drop_all_null_features", True)
    target = resolve_arg(args.target, config, "target", TARGET_COL)
    weight_col = resolve_arg(args.weight_col, config, "weight_col", WEIGHT_COL)
    model_name = resolve_arg(args.model, config, "model", "ridge")
    raw_model_params = resolve_arg(args.model_params, config, "model_params", None)
    model_params = parse_model_params(raw_model_params)
    model_out = resolve_arg(args.model_out, config, "model_out", None)
    num_workers = resolve_arg(args.num_workers, config, "num_workers", 1)
    show_progress = resolve_arg(args.progress, config, "progress", True)
    report_dir = resolve_arg(args.report_dir, config, "report_dir", "outputs")
    plot = resolve_arg(args.plot, config, "plot", True)
    plot_sample = resolve_arg(args.plot_sample, config, "plot_sample", 20000)

    max_parts = max_parts if max_parts and max_parts > 0 else None
    resolved_name, resolved_params = resolve_model_params(model_name, model_params)
    if resolved_name == "ridge" and model_params is None:
        resolved_params = {"alpha": alpha}
    model_params = resolved_params

    results = train_model(
        data_dir=data_dir,
        max_parts=max_parts,
        num_workers=num_workers,
        show_progress=show_progress,
        feature_prefix=feature_prefix,
        feature_list=feature_list,
        drop_all_null=drop_all_null,
        target=target,
        weight_col=weight_col,
        valid_days=valid_days,
        model_name=resolved_name,
        model_params=model_params,
    )

    model = results["model"]
    feature_cols = results["feature_cols"]
    score = results["score"]
    valid_df = results["valid_df"]
    valid_pred = results["valid_pred"]

    print(f"Validation weighted zero-mean R2: {score:.6f}")

    if model_out:
        model_path = Path(model_out)
        save_model(model, feature_cols, resolved_name, model_params or {}, target, model_path)
        print(f"Saved model to {model_path}")

    if report_dir:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        metrics = {
            "model": resolved_name,
            "model_params": model_params or {},
            "weighted_zero_mean_r2": float(score),
            "train_rows": results["train_rows"],
            "valid_rows": results["valid_rows"],
            "features": int(len(feature_cols)),
            "valid_days": int(valid_days),
            "max_parts": int(max_parts or 0),
        }
        (report_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        if plot:
            save_scatter(
                valid_df[target].to_numpy(),
                valid_pred,
                report_path / "valid_scatter.png",
                sample=plot_sample,
            )


if __name__ == "__main__":
    main()
