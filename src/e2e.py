from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_json_config, resolve_arg
from src.data import FEATURE_PREFIX, TARGET_COL, WEIGHT_COL, get_data_dir
from src.models import resolve_model_params
from src.predict import generate_submission
from src.train import parse_model_params, save_model, save_scatter, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model and generate a submission.")
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
    parser.add_argument("--model-out", type=str, default=None, help="Path to save the trained model.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the submission CSV.")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers for parquet loading.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for metrics/plots.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot-sample", type=int, default=None, help="Max rows for scatter plot.")
    return parser.parse_args()


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
    output = resolve_arg(args.output, config, "output", None)
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

    if model_out is None:
        model_out = f"models/{resolved_name}.pkl"
    if output is None:
        output = f"submissions/{resolved_name}_submission.csv"

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

    model_path = Path(model_out)
    save_model(model, feature_cols, resolved_name, model_params or {}, target, model_path)
    print(f"Saved model to {model_path}")

    output_df = generate_submission(
        model=model,
        feature_cols=feature_cols,
        data_dir=data_dir,
        feature_prefix=feature_prefix,
        feature_list=feature_list,
        num_workers=num_workers,
        show_progress=show_progress,
        target=target,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

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
            "submission": str(output_path),
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
