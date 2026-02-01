from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.config import load_json_config, resolve_arg
from src.data import FEATURE_PREFIX, TARGET_COL, get_data_dir, load_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for the mock test set.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory.")
    parser.add_argument("--model", type=str, default=None, help="Path to a saved model file.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--feature-prefix", type=str, default=None)
    parser.add_argument("--feature-list", type=str, default=None, help="Optional feature list file.")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers for parquet loading.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_feature_list(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    parts = [item.strip() for item in text.replace(",", "\n").splitlines()]
    return [item for item in parts if item]


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)

    data_dir_value = resolve_arg(args.data_dir, config, "data_dir", None)
    data_dir = Path(data_dir_value) if data_dir_value else get_data_dir()
    model_path_value = resolve_arg(args.model, config, "model", None)
    output = resolve_arg(args.output, config, "output", "submissions/mock_submission.csv")
    feature_prefix = resolve_arg(args.feature_prefix, config, "feature_prefix", FEATURE_PREFIX)
    feature_list = resolve_arg(args.feature_list, config, "feature_list", None)
    target = resolve_arg(args.target, config, "target", TARGET_COL)
    num_workers = resolve_arg(args.num_workers, config, "num_workers", 1)
    show_progress = resolve_arg(args.progress, config, "progress", True)

    if not model_path_value:
        raise ValueError("Model path is required. Use --model or set model in the config.")

    with Path(model_path_value).open("rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    feature_cols = payload.get("features")

    test_df = load_test(data_dir=data_dir, num_workers=num_workers, show_progress=show_progress)
    if feature_list:
        feature_cols = load_feature_list(feature_list)
    if feature_cols is None:
        feature_cols = sorted([col for col in test_df.columns if col.startswith(feature_prefix)])

    missing = [col for col in feature_cols if col not in test_df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns in test data.")

    preds = model.predict(test_df[feature_cols])

    if "row_id" in test_df.columns:
        row_ids = test_df["row_id"]
    else:
        row_ids = pd.Series(range(len(test_df)), name="row_id")

    output_df = pd.DataFrame({"row_id": row_ids, target: preds})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
