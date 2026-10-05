"""Image, box, overlap and duplicate statistics for exploratory data analysis."""

from __future__ import annotations

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from tqdm.auto import tqdm


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_stats(images: pd.DataFrame) -> pd.DataFrame:
    """Per-image colour, brightness, blur and perceptual-hash statistics (one pass over the files).

    ``blur`` is the variance of the Laplacian: lower means blurrier.
    """
    rows = []
    for uid, path in tqdm(zip(images["uid"], images["path"]), total=len(images), desc="image stats"):
        rgb = read_rgb(path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean = rgb.reshape(-1, 3).mean(axis=0)
        std = rgb.reshape(-1, 3).std(axis=0)
        rows.append(
            {
                "uid": uid,
                "mean_r": mean[0], "mean_g": mean[1], "mean_b": mean[2],
                "std_r": std[0], "std_g": std[1], "std_b": std[2],
                "brightness": gray.mean(),
                "contrast": gray.std(),
                "blur": cv2.Laplacian(gray, cv2.CV_64F).var(),
                "phash": str(imagehash.phash(Image.fromarray(rgb))),
            }
        )
    return pd.DataFrame(rows)


def pairwise_iou(xywh: np.ndarray) -> np.ndarray:
    """IoU matrix for boxes given as rows of ``x, y, w, h``."""
    x1, y1 = xywh[:, 0], xywh[:, 1]
    x2, y2 = x1 + xywh[:, 2], y1 + xywh[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = iw * ih
    area = xywh[:, 2] * xywh[:, 3]
    union = area[:, None] + area - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def overlap_stats(boxes: pd.DataFrame, iou_threshold: float = 0.1) -> pd.DataFrame:
    """Per-image overlap: max IoU and the fraction of boxes overlapping another box above the threshold."""
    rows = []
    for uid, g in boxes.groupby("uid"):
        if len(g) < 2:
            rows.append({"uid": uid, "max_iou": 0.0, "frac_overlapping": 0.0})
            continue
        iou = pairwise_iou(g[["x", "y", "w", "h"]].to_numpy(float))
        np.fill_diagonal(iou, 0)
        rows.append(
            {
                "uid": uid,
                "max_iou": iou.max(),
                "frac_overlapping": (iou.max(axis=1) > iou_threshold).mean(),
            }
        )
    return pd.DataFrame(rows)


def find_duplicates(stats: pd.DataFrame, images: pd.DataFrame, max_distance: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """Near-duplicate image pairs by perceptual-hash Hamming distance.

    Returns ``(pairs, groups)``: ``pairs`` lists each near-duplicate pair with its splits and
    whether it crosses splits (leakage); ``groups`` maps every uid to a duplicate-group id, for
    grouped splitting.
    """
    df = stats[["uid", "phash"]].merge(images[["uid", "split"]], on="uid")
    hashes = np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1).astype(bool)
    # Hamming distance = number of differing bits.
    dist = (bits[:, None, :] != bits[None, :, :]).sum(axis=2)
    ia, ib = np.where(np.triu(dist <= max_distance, k=1))

    pairs = pd.DataFrame(
        {
            "uid_a": df["uid"].to_numpy()[ia],
            "uid_b": df["uid"].to_numpy()[ib],
            "split_a": df["split"].to_numpy()[ia],
            "split_b": df["split"].to_numpy()[ib],
            "distance": dist[ia, ib],
        }
    )
    pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]

    # Union-find to merge pairs into groups.
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(ia, ib):
        parent[find(a)] = find(b)
    groups = pd.Series([find(i) for i in range(len(df))], index=df["uid"], name="dup_group")
    return pairs, groups


def background_groups(stats: pd.DataFrame, k: int = 3, seed: int = 0) -> pd.Series:
    """Cluster images by mean RGB colour. Labels are ordered from darkest (0) to brightest."""
    X = stats[["mean_r", "mean_g", "mean_b"]].to_numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    order = np.argsort(km.cluster_centers_.sum(axis=1))
    relabel = {old: new for new, old in enumerate(order)}
    return pd.Series([relabel[label] for label in km.labels_], index=stats["uid"], name="bg_group")


def count_bins(counts: pd.Series, edges: list[float] | None = None) -> pd.Series:
    """Bin cell counts into labelled ranges. Default edges are the quartiles of ``counts``."""
    if edges is None:
        edges = sorted(set(np.quantile(counts, [0, 0.25, 0.5, 0.75, 1]).round().astype(int)))
    edges = list(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(counts, bins=edges)
