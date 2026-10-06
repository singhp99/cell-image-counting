# Project Plan

This is a working roadmap. Edit it freely as the team makes decisions. Background, stakeholders, and KPIs are in [`project.md`](project.md); dataset details are in [`../data/README.md`](../data/README.md).

**Contents**
1. [Goal and scope](#1-goal-and-scope)
2. [Status at a glance](#2-status-at-a-glance)
3. [What the data tells us](#3-what-the-data-tells-us)
4. [Evaluation](#4-evaluation)
5. [Generalization strategy](#5-generalization-strategy)
6. [Roadmap](#6-roadmap)
7. [Team workflow](#7-team-workflow)
8. [Open decisions](#8-open-decisions)

---

## 1. Goal and scope

**Headline:** automated Live/Dead cell counting with viability %, built to hold up on images from new sessions, labs, and microscopes.

The tool works like a bird-identification app: find every cell, label each one, then count.

```
Input : microscope image (any size or shape)
          ↓
Detect   : a box around each cell
Classify : Live or Dead (Dead cells take up the blue trypan stain)
          ↓
Output: annotated image + Live: 52 · Dead: 3 · Total: 55 · Viability: 94.5% + quality warnings
```

It differs from a bird app in where the difficulty lies. Bird apps have a few large objects and thousands of look-alike classes. Here there are 23–91 small (~28 px), near-identical cells and two easily separated classes. **The hard part is counting every cell exactly once on images that look different from the training data**, not identifying what a cell is.

| Scope | Work |
|---|---|
| **Core** | Live/Dead detector with tiling and scale/photometric augmentation; per-class counts and viability %; leave-one-session-out test plus stress-test curves |
| **Stretch 1** | Compare with a two-stage pipeline: Cellpose (pretrained on varied microscopy) finds cells, then a small colour classifier labels Live/Dead |
| **Stretch 2** | Test set from a different lab or microscope; warnings for inputs the model can't handle |
| **Stretch 3** | Density-map model for crowded and blurry images (synthetic dataset); public datasets to teach "what is a cell"; per-lab fine-tuning |
| **Stretch 4** | Demo app (upload an image → annotated image + counts), e.g. Gradio on Colab |
| **Out of scope** | Cell types beyond Live/Dead (no labels). CoNIC's 6 nucleus types come from stained tissue slices, a different kind of microscopy; treat it as a separate project. |

---

## 2. Status at a glance

| Phase | Status | Deliverable |
|---|---|---|
| 0. Setup | ✅ Done | `environment.yml`, `src/cellcount/`, `notebooks/00_setup_and_download.ipynb`, `data/README.md` |
| 1. EDA | ✅ Done, team review pending | `notebooks/01_eda.ipynb`, `splits/splits.csv` |
| 2. Baselines | ⏭️ Next | `notebooks/02_baselines.ipynb`, results table |
| 3. Core model | Planned | `notebooks/03_detector.ipynb` |
| 4. Generalization | Planned | stress-test curves, leave-one-session-out results, external test set |
| 5. Product and report | Planned | inference pipeline, demo, efficiency numbers, final report |

---

## 3. What the data tells us

From `notebooks/01_eda.ipynb` (full findings in its last section):

- **500 unique images**, all 640×640, with 26,715 boxes: **Live 25,691 / Dead 1,024 (~4%)**.
  - The public Roboflow version (~1,200 images) is the same images with augmented copies; don't use it.
- **One lab, one day:** every image is from `2022_12_7`. There are only 5 acquisition sessions (`1_A`, `1_B`, `2_A`, `2_B`, `3_A`), 100 frames each.
  - Each session has its own background colour and typical count (means of 37–71 cells).
  - A model can therefore learn "background → count" instead of counting cells.
- **Cells are uniform:** about 28 px across, and they barely overlap (0.1% of cells).
  - This set can't test crowded or overlapping cells; that needs the synthetic set.
- **Label policy:** faint, out-of-focus cells are often unlabeled, so the model learns the annotator's idea of a countable cell.
- **The data is smaller than it looks for some methods.** A detector learns from 26,715 labelled cells, but a model that predicts one number per image learns from only 350 training images. This favours detection.

---

## 4. Evaluation

### Metrics

Exact-match "accuracy" is too strict for counts, so use the following metrics, computed on `splits/splits.csv`:

| Metric | Applies to | Why |
|---|---|---|
| **MAE** (mean absolute error, in cells) | Live, Dead, and Total separately | Main metric, easy to interpret. Split by class, because Dead is only ~4% and would otherwise be hidden. |
| **Viability error** (percentage points) | Live / (Live + Dead) | The number a lab actually uses. |
| **±10% accuracy** | Total | Share of images whose count is within 10% of the truth; replaces "accuracy" in the KPIs. |
| **R²** | Total | Already a KPI. On its own it can hide a systematic bias, so always report it with MAE. |
| **RMSE, relative error** | Total | Highlight large misses, and make errors comparable between sparse and dense images. |
| **mAP@0.5** (diagnostic only) | Detectors | Checks whether boxes land on real cells, not just whether the totals happen to match. |

Always break results down by **session**, **count bin**, and **class**. A good average can hide failures.

### Three levels of testing

The model is only as good as its hardest honest test.

1. **Random split** (`split` column of `splits/splits.csv`): in-distribution performance, used for development.
2. **Leave one session out:** train on 4 sessions and test on the 5th, using the `source` column. The gap between this score and the random-split score shows how much the model relies on recognizing the session.
3. **Stress tests:** apply each change to the test set at increasing strength, and plot error against strength:
   - rescaling (0.25–4×)
   - crops to irregular or round shapes
   - blur (σ 0–5 px)
   - colour and white-balance shifts
   - noise and JPEG compression

   The curves answer "what inputs can we handle?"
4. **External test set (stretch):** 20–30 images from another lab or microscope, hand-labelled in Roboflow. This is the only real proof of generalization.

---

## 5. Generalization strategy

Real inputs will not match the training set. They may be larger or smaller, not rectangular, and from labs with different stains, lighting, focus, and cameras.

### 5.1 Different image sizes and shapes

**What matters is how many pixels a cell covers, not the image size.** Bring cells to a consistent size and let the model work on fixed-size pieces:

```
any image
  → mask the usable area   (round microscope view, black borders, counting-chamber grid)
  → estimate cell size → rescale so cells ≈ 28 px
  → cut into overlapping 640×640 tiles   (overlap ≥ 2 cell widths)
  → detect Live/Dead in each tile
  → merge tiles, drop cells counted twice where tiles overlap
  → keep only cells whose centre is inside the usable area
  → counts + viability % + quality warnings
```

- **Never stretch images** to change their aspect ratio; pad or tile instead.
- **Non-rectangular inputs:** pad to a rectangle, and count only cells whose centre lies inside the real image area. Detect that area automatically (dark round borders are easy to threshold) or let the user draw it.
- **Cell-size estimate at prediction time:** run the detector at a few zoom levels and keep the one with the most confident detections. If the user supplies microscope settings (magnification, µm per pixel), use those instead.
- **Tiling and merging:** use an existing tool such as SAHI rather than writing our own.
- **Training:** use random crops (256–640 px) and random zoom (0.5–2×), so the model never assumes a fixed frame or cell size. Also cut random round or irregular shapes out of training images and remove the labels of cells that fall outside, so the model learns to ignore padding.

### 5.2 Different image quality between labs

**Augmentation:** make training images more varied than real ones will be. Albumentations covers all of these:

| What differs between labs | Augmentation that simulates it |
|---|---|
| Stain strength, white balance | Brightness, contrast, gamma, saturation, *small* hue shifts |
| Uneven lighting | Random light gradients and darker corners |
| Focus | Gaussian and defocus blur at several strengths |
| Camera | Noise, JPEG compression, lower resolution followed by upscaling |
| Microscope orientation | Flips and 90° rotations |

> ⚠️ **Live vs Dead depends on the blue colour.** Keep hue shifts small and never convert images fully to grayscale. Shift the background colour freely, but keep "blue cell vs clear cell" intact.

**Preprocessing:** subtract the background (remove a heavily blurred copy of the image) and normalize each image on its own, so lighting differences are removed before the model sees the image.

### 5.3 More varied data (the biggest lever)

Augmentation only stretches the 5 sessions we have. In order of effort:

1. **Start from a model already trained on many kinds of cell images.**
   - Cellpose finds the cells, then a small classifier labels each one Live or Dead from its colour.
   - Because the Live/Dead difference is mostly colour, this split may transfer to new labs better than one end-to-end model trained on our 5 sessions.
2. **Add public cell datasets for "what is a cell".**
   - LIVECell, for example, is large and covers 8 cell types; it has no Live/Dead labels.
   - Use those datasets only to teach the model what a cell is, and our data only to teach Live vs Dead.
3. **Per-lab fine-tuning:** a new lab corrects the counts on 5–10 of its own images, and we fine-tune briefly.

### 5.4 Flag inputs we can't handle

Don't return a confident wrong count. Warn when:
- the estimated cell size is far outside the training range
- the average detection confidence is low
- the image is very blurry
- the image is mostly background

---

## 6. Roadmap

### Phase 0: Setup ✅

- [x] Conda env `cellcount` (`environment.yml`) and the installable utils package `src/cellcount/` (`pyproject.toml`).
- [x] `notebooks/00_setup_and_download.ipynb`, which reads `ROBOFLOW_EXPORT_URL` from `.env` or Colab Secrets.
- [x] `data/README.md` with the source, format, license, and collaborator setup.
- [x] Colab workflow: notebooks clone the repo from GitHub; data and outputs go to a shared Google Drive folder.

### Phase 1: Exploratory data analysis ✅

- [x] `notebooks/01_eda.ipynb` covers images, sizes, counts, cell sizes, overlap, colour and background, blur, duplicates, label outliers, and the split.
- [x] Shared split `splits/splits.csv`: 350/75/75, stratified by session × count bin.
- [ ] Team review of the findings and preprocessing decisions.

### Phase 2: Baselines

The cheapest baselines come first, so every later model has something to beat. Every result goes into a single table using the Section 4 metrics.

- [ ] Add a `src/cellcount/metrics.py` that computes the Section 4 metrics, so every notebook reports the same numbers.
- [ ] **Predict the mean count** of the training set. This is the floor, with R² ≈ 0.
- [ ] **Classical image processing:** background subtraction, Otsu threshold, watershed, then count. Classify Live/Dead by the blue intensity inside each cell. Expect over-counting, because faint cells are unlabeled.
- [ ] **Pretrained with no training:** Cellpose, counting the masks it finds, plus the colour rule for Live/Dead. This is also the starting point for Stretch 1.

### Phase 3: Core model (Live/Dead detector)

- [ ] YOLO (Ultralytics) trained on our boxes with 2 classes, starting from pretrained weights.
- [ ] Training recipe:
  - random crops and zoom (§5.1)
  - photometric and blur augmentation with limited hue shifts (§5.2)
  - early stopping on the validation set
- [ ] Prediction pipeline: tiling and merging (SAHI), plus the usable-area mask (§5.1).
- [ ] Tune the confidence threshold on the validation set to minimize count MAE, not just to maximize mAP.
- [ ] Error analysis after each experiment: look at the worst 20 images and the cells missed or counted twice.
- [ ] Optional comparison: a count-regression CNN, to show why detection is the better fit (Section 3).

### Phase 4: Generalization

- [ ] Leave-one-session-out cross-validation (5 folds) for the best baseline and the core model.
- [ ] Stress-test curves (§4): scale, shape, blur, colour, and noise.
- [ ] *Stretch 1:* Cellpose plus colour classifier vs the trained detector, on all three test levels.
- [ ] *Stretch 2:* collect and label an external test set from another lab; add the warnings from §5.4.
- [ ] *Stretch 3:*
  - Synthetic Cell Images and Masks: error against blur level, and crowded scenes.
  - A density-map model (U-Net or CSRNet style) for crowded images, compared with the detector.
  - Public datasets for "what is a cell".

### Phase 5: Product and report

- [ ] `src/cellcount/predict.py`: one function from any image to counts, viability %, annotated image, and warnings.
- [ ] Efficiency: inference time per image (CPU and Colab GPU), memory use, and model size, compared with counting by hand.
- [ ] *Stretch 4:* Gradio demo.
- [ ] Final report:
  - results table (all metrics × all test levels)
  - stress-test curves
  - failure examples
  - recommendations for stakeholders

---

## 7. Team workflow

```
data/            # gitignored; download with notebooks/00 (data/README.md says how)
notebooks/       # NN_topic.ipynb; each starts with the shared Colab bootstrap cell
src/cellcount/   # all reusable code (data, eda, viz, splits; metrics, models, predict to come)
splits/          # splits.csv: the shared, committed train/valid/test assignment
notes/           # project.md, plan.md, dataset_link.md, meeting notes
outputs/         # gitignored; figures, checkpoints, results (on Drive when using Colab)
```

- **Setup:** see `data/README.md`. In short: `conda env create -f environment.yml`, register the kernel, `nbstripout --install`, then copy `.env.example` to `.env`.
- **Secrets** (`ROBOFLOW_EXPORT_URL`, `ROBOFLOW_API_KEY`, `GH_TOKEN`) go only in `.env` or Colab Secrets, never in notebooks or notes.
- **Shared code:** put reusable logic in `src/cellcount/`, not copied between notebooks. Add new dependencies to `pyproject.toml` with lower bounds only.
- **Shared split:** always train and evaluate on `splits/splits.csv`, never on Roboflow's split.
- **Branches:** `vX.Y.Z-<stage>-<topic>`; changes reach `main` only through pull requests.

---

## 8. Open decisions

- [x] Dataset version → the fork's raw COCO export (500 images).
- [x] Where do we train? → Google Colab, with data on a shared Drive folder.
- [x] RGB or grayscale? → **RGB**, because Dead cells are recognized by their blue stain.
- [ ] Confirm the core scope: a Live/Dead detector with viability %? (proposed in Section 1)
- [ ] Which metrics count as success? (proposal: per-class MAE, viability error, ±10% accuracy, R²; see Section 4)
- [ ] Detector framework: Ultralytics YOLO (proposed) or torchvision Faster R-CNN?
- [ ] Can anyone get 20–30 images from another lab or microscope for the external test set?
- [ ] Who owns which phase or task?
- [ ] Agree on the branch and pull-request rules (Section 7).
