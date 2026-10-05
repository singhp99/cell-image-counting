"""Plotting helpers for viewing images with their boxes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from .eda import read_rgb


def show_image(row: pd.Series, boxes: pd.DataFrame | None = None, ax=None, title: str | None = None):
    """Draw one image (a row of the images DataFrame), optionally with its boxes."""
    ax = ax or plt.gca()
    ax.imshow(read_rgb(row["path"]))
    if boxes is not None:
        for b in boxes.loc[boxes["uid"] == row["uid"]].itertuples():
            ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, lw=0.8, ec="yellow"))
    ax.set_title(title if title is not None else f"{row['uid']}  n={row['count']}", fontsize=8)
    ax.axis("off")
    return ax


def show_samples(
    images: pd.DataFrame,
    boxes: pd.DataFrame | None = None,
    n: int = 8,
    by: str | None = None,
    ncols: int = 4,
    seed: int = 0,
):
    """Grid of random images. With ``by``, draws one row of ``ncols`` samples per value of that column."""
    if by is None:
        sample = images.sample(min(n, len(images)), random_state=seed)
        nrows = -(-len(sample) // ncols)
        rows = [sample.iloc[i * ncols:(i + 1) * ncols] for i in range(nrows)]
        labels = [None] * nrows
    else:
        groups = sorted(images[by].dropna().unique())
        rows = [images[images[by] == g].sample(min(ncols, (images[by] == g).sum()), random_state=seed) for g in groups]
        labels = [f"{by}={g}" for g in groups]
        nrows = len(rows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for r, (row_df, label) in enumerate(zip(rows, labels)):
        for c in range(ncols):
            ax = axes[r, c]
            if c < len(row_df):
                show_image(row_df.iloc[c], boxes, ax=ax)
            else:
                ax.axis("off")
        if label:
            axes[r, 0].text(-0.05, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def show_pairs(pairs: pd.DataFrame, images: pd.DataFrame, n: int = 4):
    """Show near-duplicate pairs side by side (output of ``eda.find_duplicates``)."""
    pairs = pairs.head(n)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6, 3 * len(pairs)), squeeze=False)
    by_uid = images.set_index("uid", drop=False)
    for i, p in enumerate(pairs.itertuples()):
        show_image(by_uid.loc[p.uid_a], ax=axes[i, 0])
        show_image(by_uid.loc[p.uid_b], ax=axes[i, 1], title=f"{p.uid_b}  dist={p.distance}")
    fig.tight_layout()
    return fig
