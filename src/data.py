from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

FEATURE_PREFIX = "feature_"
TARGET_COL = "responder_6"
WEIGHT_COL = "weight"

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    return get_repo_root() / "data"


def list_parquet_parts(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            files = sorted(path.rglob("*.parquet"))
        return files
    return [path]


def _read_parquet(path: Path, columns: Iterable[str] | None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def load_parquet_parts(
    path: Path,
    columns: Iterable[str] | None = None,
    max_parts: int | None = None,
    num_workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    parts = list_parquet_parts(path)
    if max_parts is not None:
        parts = parts[:max_parts]
    if not parts:
        raise FileNotFoundError(f"No parquet files found at {path}")

    if num_workers is None or num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        iterable = tqdm(parts, desc="Loading parquet", disable=not show_progress)
        frames = [_read_parquet(part, columns) for part in iterable]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        frames = [None] * len(parts)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_read_parquet, part, columns): idx for idx, part in enumerate(parts)
            }
            iterable = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading parquet",
                disable=not show_progress,
            )
            for future in iterable:
                idx = futures[future]
                frames[idx] = future.result()
    return pd.concat(frames, ignore_index=True)


def load_train(
    data_dir: Path | None = None,
    columns: Iterable[str] | None = None,
    max_parts: int | None = None,
    num_workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    data_root = data_dir or get_data_dir()
    return load_parquet_parts(
        data_root / "train.parquet",
        columns=columns,
        max_parts=max_parts,
        num_workers=num_workers,
        show_progress=show_progress,
    )


def load_test(
    data_dir: Path | None = None,
    columns: Iterable[str] | None = None,
    num_workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    data_root = data_dir or get_data_dir()
    return load_parquet_parts(
        data_root / "test.parquet",
        columns=columns,
        num_workers=num_workers,
        show_progress=show_progress,
    )
