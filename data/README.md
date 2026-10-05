# Data

Datasets are **not committed**. Everything in this folder except this README is gitignored. On Colab, data lives on Google Drive at `MyDrive/cell-image-counting/data/` instead.

## Roboflow Cell Counting (primary)

| | |
|---|---|
| Source | Our fork (`dingshan-deng/cell-counting-zeqo5-7svoy`) of [cell-counting/cell-counting-zeqo5](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5) |
| Export | Raw dataset, COCO JSON, **no preprocessing or augmentation** |
| Contents | 500 images, 640×640 RGB; 26,715 boxes; classes `Live`, `Dead` |
| Roboflow split | train 350 / valid 100 / test 50 (we use `splits/splits.csv` instead) |
| License | CC BY 4.0 |
| Local folder | `data/roboflow-cell-counting-raw-coco/{train,valid,test}/` (images + `_annotations.coco.json`) |

Download it with `notebooks/00_setup_and_download.ipynb`. That notebook reads the `ROBOFLOW_EXPORT_URL` secret: from `.env` locally (see `.env.example`), or from Colab Secrets.

Don't use the public project's version 1 (about 1,200 images). It is the same 500 images, with the training images augmented 3× and everything stretched to 640×640.

## Quick setup for collaborators

```bash
conda env create -f environment.yml && conda activate cellcount
python -m ipykernel install --user --name cellcount --display-name "Python (cellcount)"
nbstripout --install          # strip notebook outputs on commit
cp .env.example .env          # add ROBOFLOW_EXPORT_URL (ask the team)
```

On Colab, open a notebook from GitHub, add `ROBOFLOW_EXPORT_URL` (and `GH_TOKEN` if the repo is private) under Secrets, and Run all. The first cell mounts Drive, clones the repo, and installs the `cellcount` utils.
