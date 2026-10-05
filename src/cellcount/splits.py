"""Reproducible train/valid/test splits, stratified by count and grouped by near-duplicates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_splits(
    images: pd.DataFrame,
    group_col: str = "dup_group",
    stratify_col: str = "count_bin",
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.Series:
    """Assign each image to train/valid/test.

    Images in the same ``group_col`` (near-duplicates) always land in the same split, and each
    split keeps roughly the same distribution of ``stratify_col``. Fractions are rounded to the
    nearest 5%.
    """
    assert abs(sum(fractions) - 1) < 1e-6, "fractions must sum to 1"
    n_folds = 20
    n_train, n_valid = (round(f * n_folds) for f in fractions[:2])

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(images), dtype=int)
    y = images[stratify_col].astype(str)
    for k, (_, idx) in enumerate(skf.split(images, y, groups=images[group_col])):
        fold[idx] = k

    split = np.where(fold < n_train, "train", np.where(fold < n_train + n_valid, "valid", "test"))
    return pd.Series(split, index=images.index, name="new_split")


def save_splits(images: pd.DataFrame, path: Path, split_col: str = "new_split") -> Path:
    """Write the split assignment so every collaborator trains and evaluates on the same images."""
    cols = [c for c in ["uid", "split", "file_name", "source", "frame", "count"] if c in images] + [split_col]
    images[cols].rename(columns={"split": "roboflow_split", split_col: "split"}).to_csv(path, index=False)
    return Path(path)


def load_splits(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
