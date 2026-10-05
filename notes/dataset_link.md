# Dataset Download Links

## Roboflow dataset

[Roboflow Cell Counting](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5) | **Primary, in use** | 500 unique images (the public version's ~1,200 includes 3× augmented training copies)

Our fork: workspace `dingshan-deng`, project ID `cell-counting-zeqo5-7svoy` (display name "cell counting"), raw COCO export. The fork has no generated version yet, so the API route (`download_roboflow`) only works after someone generates one.

> **Never paste API keys or signed download links into this repo.**
> - Local: put `ROBOFLOW_EXPORT_URL=...` (and optionally `ROBOFLOW_API_KEY=...`) in `.env` (gitignored, template in `.env.example`).
> - Colab: add the same names under Secrets (key icon in the left sidebar).

Download with the repo utils (works locally and on Colab), or just run `notebooks/00_setup_and_download.ipynb`:

```python
from cellcount.config import get_paths, get_secret
from cellcount.data import download_export

download_export(get_secret("ROBOFLOW_EXPORT_URL"), get_paths().data)
```
