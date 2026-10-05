# Project Plan

This is a working roadmap. Edit it freely as the team makes decisions. Project scope and KPIs are in [`project.md`](project.md).

---

## Suggestions

### 1. Define "accuracy" before the first experiment

For a count, exact-match accuracy is too strict to be useful: an image with 200 cells will almost never be counted exactly right. Pick and report these metrics everywhere:

| Metric | Why |
|---|---|
| **MAE** (mean absolute error, in cells) | Main metric, easy to interpret. |
| **RMSE** | Penalizes large misses, which matter most for dense images. |
| **MAPE** or relative error | Makes errors comparable between sparse and dense images. |
| **±10% accuracy** | Share of images whose predicted count is within 10% of the truth. This replaces "accuracy" in the KPIs. |
| **R²** | Already a KPI. On its own it can hide a systematic bias, so always report it with MAE. |

Always break these results down by **count bin** (for example 0–25, 25–100, 100+) and by **background color**. A good average can hide failures on dense images.

### 2. Confirm the annotation format first

Roboflow Universe projects usually ship **bounding boxes** (object detection), and the count is the number of boxes per image. Before doing anything else, confirm which export format and version we use, because it decides which model families we can try:

- boxes → detection models, or density maps built from box centers
- counts only → regression models only

### 3. Watch for data leakage from Roboflow augmentation

Roboflow dataset versions often include **augmented copies** (flips, rotations, brightness changes) that are already spread across train, validation, and test. If the same base image appears in two splits, the test scores will look better than they really are.

- Check for near-duplicates with a perceptual hash (for example the `imagehash` package) before trusting the provided splits.
- If we find duplicates, export the **raw, unaugmented** version and make our own split, grouped by base image.
- Do augmentation ourselves, in the training pipeline only.

### 4. Agree on a repository layout early

Everyone works in one notebook right now, which leads to merge conflicts quickly in a shared repo. Suggested layout:

```
data/            # gitignored; each person downloads it locally (data/README.md says how)
notebooks/       # one notebook per person or topic, e.g. 01_eda_<name>.ipynb
src/             # shared code: dataset loading, metrics, models
notes/           # project.md, plan.md, meeting notes
requirements.txt # pinned dependencies
```

- Keep the Roboflow API key in `.env` (already gitignored), never in a notebook.
- Clear notebook outputs before committing (`nbstripout` does this automatically) so diffs stay readable.
- Fix one random seed and save the split as a file (`splits.csv`) so everyone uses the same train/validation/test images.

---

## Roadmap

### Phase 0: Setup

- [x] Conda env `cellcount` (`environment.yml`) and installable utils package `src/cellcount/` (`pyproject.toml`). PyTorch will be added when modeling starts; Colab already ships it.
- [x] Download notebook `notebooks/00_setup_and_download.ipynb`, which reads `ROBOFLOW_EXPORT_URL` from `.env` locally or from Colab Secrets.
- [x] `data/README.md` with the dataset source, export format, license, and collaborator setup.
- [x] Colab workflow: notebooks clone the repo from GitHub; data and outputs go to a shared Google Drive folder.
- [ ] Agree on branch naming (current pattern: `vX.Y.Z-<stage>-<topic>`) and that changes reach `main` only through pull requests.

### Phase 1: Exploratory data analysis (done, pending team review)

Questions to answer, with the plot or table that answers each:

| Question | Analysis |
|---|---|
| What do the images look like? | Grid of samples, one per background color, with boxes drawn on. |
| Image sizes and aspect ratios? | Histogram of width and height. Decide the input size and whether to resize or tile. |
| How are counts distributed? | Histogram of counts per image; min, max, median, skew. Very skewed counts → stratify the split by count bin. |
| How big are the cells? | Distribution of box sizes. Tiny cells mean we cannot downscale aggressively. |
| How much do cells overlap? | Pairwise box overlap (IoU) per image; share of images with heavy overlap. |
| How do colors vary? | Mean and standard deviation per channel for each background group. Decide whether to normalize, convert to grayscale, or keep RGB. |
| Are there duplicates or leakage? | Perceptual-hash near-duplicates within and across the provided splits. |
| Are some labels bad? | Manually check outliers: images with 0 boxes, extreme counts, or truncated boxes. |

**Deliverable:** `notebooks/01_eda.ipynb`. Its last section, "Findings → preprocessing decisions", answers the three milestone questions, and the notebook writes the shared split to `splits/splits.csv`.

Headline results:
- 500 unique images from only **5 acquisition sources**, with counts of 23–91 per image.
- Count and background colour both depend heavily on the source.
- Cells barely overlap (0.1%), so dense or overlapping cases must come from the Phase 4 datasets.
- Faint, out-of-focus cells are often unlabeled.
- Split: 350/75/75, stratified by source × count, plus leave-one-source-out cross-validation for generalization.

Splits: **70/15/15** stratified by source × count bin (`splits/splits.csv`). With 350 training images, use pretrained backbones and heavy augmentation, and report leave-one-source-out cross-validation alongside the main split.

### Phase 2: Baselines

Build the cheapest baselines first, so every later model has something to beat.

1. **Predict the mean count** of the training set. This is the floor, with R² ≈ 0.
2. **Classical image processing:** Otsu threshold, then watershed, then connected-component count (scikit-image). No training required.
3. **Pretrained, no training:** run Cellpose or StarDist and count the masks they find.
4. **Fine-tuned detector:** YOLOv8 or Faster R-CNN trained on our boxes; count = number of detections.

Record every baseline in a single results table using the Suggestion 1 metrics and count bins.

### Phase 3: Our model

Candidate approaches, roughly in order of effort:

- **Direct count regression:** a pretrained CNN backbone (ResNet or EfficientNet) with a regression head. Train on log(count + 1) if the counts are skewed.
- **Density-map regression:** turn box centers into Gaussian density maps and train a fully convolutional network (U-Net or CSRNet style); the count is the sum of the map. This is the standard way to handle dense and overlapping cells.
- **Detection + counting:** the fine-tuned detector from Phase 2, tuned further (anchor sizes, non-max suppression threshold) for small, crowded objects.

Ways to improve any of them:

- Augmentation: flips, 90° rotations, color jitter, Gaussian blur and noise.
- Tiling: split large images into tiles, count each, then sum.
- Error analysis: look at the worst 20 images after every experiment.

### Phase 4: Robustness and generalization

- **Synthetic Cell Images and Masks:** measure how error grows with blur level; try training with blur augmentation.
- **Cross-dataset test:** train on Roboflow, then test on the synthetic set (and the reverse) to measure generalization.
- **CoNIC:** test whether the model transfers to histology nuclei, a different imaging method.

### Phase 5: Efficiency and reporting

- [ ] Measure inference time per image on CPU and GPU, plus memory use and model size.
- [ ] Compare with manual counting time, using a published per-image estimate or by timing ourselves on a few images.
- [ ] Final report: results table, error plots by count bin, failure examples, and recommendations for stakeholders.

---

## Open decisions for the team

- [x] Which Roboflow dataset version and export format do we use? → The fork's raw COCO export (500 images). The public v1 is augmented and resized.
- [ ] Count Live + Dead together, or also predict the dead fraction (viability)?
- [ ] Which metrics count as success? (proposal: MAE + ±10% accuracy + R²)
- [ ] Should images be RGB or grayscale? (decide after the EDA)
- [ ] Which framework: plain PyTorch, PyTorch Lightning, or Ultralytics for detection?
- [x] Where do we train? → Google Colab, with data on a shared Drive folder.
- [ ] Who owns which phase or task?
