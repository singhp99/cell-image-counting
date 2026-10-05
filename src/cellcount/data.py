"""Download the Roboflow dataset and load its COCO annotations into DataFrames."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

SPLITS = ("train", "valid", "test")
COCO_FILE = "_annotations.coco.json"

# Our fork of https://universe.roboflow.com/cell-counting/cell-counting-zeqo5 (500 original images,
# classes Live/Dead). The public project's version 1 has ~1,200 images only because the 350 train
# images were augmented 3x and everything was stretched to 640x640, so we use the raw fork export.
ROBOFLOW_WORKSPACE = "dingshan-deng"
ROBOFLOW_PROJECT = "cell-counting-zeqo5-7svoy"  # project ID (display name: "cell counting")
RAW_DIRNAME = "roboflow-cell-counting-raw-coco"


def _is_present(out: Path) -> bool:
    return any((out / s / COCO_FILE).exists() for s in SPLITS)


def download_export(url: str, dest: Path, dirname: str = RAW_DIRNAME, overwrite: bool = False) -> Path:
    """Download and unzip a Roboflow signed export link (the "Raw URL" of a COCO export)."""
    out = Path(dest) / dirname
    if not overwrite and _is_present(out):
        print(f"Dataset already present at {out}")
        return out

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        with urllib.request.urlopen(url) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as zf:
            zf.extractall(out)
    print(f"Downloaded dataset to {out}")
    return out


def download_roboflow(
    api_key: str,
    dest: Path,
    version: int,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    fmt: str = "coco",
    overwrite: bool = False,
) -> Path:
    """Download a generated dataset version through the Roboflow API.

    The fork has no generated version yet; create one in the Roboflow UI (Generate, with no
    preprocessing or augmentation) before using this.
    """
    out = Path(dest) / f"roboflow-cell-counting-v{version}-{fmt}"
    if not overwrite and _is_present(out):
        print(f"Dataset already present at {out}")
        return out

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    rf.workspace(workspace).project(project).version(version).download(fmt, location=str(out), overwrite=True)
    return out


def load_coco(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every split of a Roboflow COCO export.

    Returns ``(images, boxes)``. ``images`` has one row per image (``uid`` = "<split>/<id>")
    with its path, size, acquisition ``source`` and ``frame`` (parsed from the original file
    name) and cell ``count``. ``boxes`` has one row per annotation with the
    COCO box ``x, y, w, h`` plus centre ``cx, cy`` and ``area``.
    """
    root = Path(root)
    image_rows, box_rows = [], []
    for split in SPLITS:
        ann_file = root / split / COCO_FILE
        if not ann_file.exists():
            continue
        coco = json.loads(ann_file.read_text())

        # Roboflow adds a placeholder category (supercategory "none") that is the parent
        # of the real classes; it never labels a cell.
        cats = {c["id"]: c for c in coco["categories"]}
        parents = {c["supercategory"] for c in cats.values()}
        placeholder = {cid for cid, c in cats.items() if c["supercategory"] == "none" and c["name"] in parents}

        for img in coco["images"]:
            image_rows.append(
                {
                    "uid": f"{split}/{img['id']}",
                    "split": split,
                    "image_id": img["id"],
                    "file_name": img["file_name"],
                    "path": str(root / split / img["file_name"]),
                    "width": img["width"],
                    "height": img["height"],
                }
            )
        for ann in coco["annotations"]:
            if ann["category_id"] in placeholder:
                continue
            x, y, w, h = ann["bbox"]
            box_rows.append(
                {
                    "uid": f"{split}/{ann['image_id']}",
                    "split": split,
                    "category": cats[ann["category_id"]]["name"],
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    if not image_rows:
        raise FileNotFoundError(f"No {COCO_FILE} found under {root}/{{{','.join(SPLITS)}}}")

    images = pd.DataFrame(image_rows)
    # Roboflow renames files to "<original>_jpg.rf.<hash>.jpg"; originals look like
    # "2022_12_7_1_A-creport_181": capture session/well ("source") and frame number.
    images["original_name"] = images["file_name"].str.replace(r"_(jpe?g|png|tiff?)\.rf\..*$", "", regex=True)
    parts = images["original_name"].str.extract(r"^(?P<source>.+?)-.*?(?P<frame>\d+)$")
    images["source"] = parts["source"]
    images["frame"] = pd.to_numeric(parts["frame"])
    boxes = pd.DataFrame(box_rows, columns=["uid", "split", "category", "x", "y", "w", "h"])
    boxes["cx"] = boxes["x"] + boxes["w"] / 2
    boxes["cy"] = boxes["y"] + boxes["h"] / 2
    boxes["area"] = boxes["w"] * boxes["h"]

    counts = boxes.groupby("uid").size()
    images["count"] = images["uid"].map(counts).fillna(0).astype(int)
    return images, boxes
