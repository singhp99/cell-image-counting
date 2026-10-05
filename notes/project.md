# Cell Image Counting

## Scope

Many processes in medical research are now automated, but counting cells in microscopy images is still mostly done by hand, which is slow and tedious. We aim to build a deep learning model that takes a microscopy image and outputs an estimated cell count.

We will train the model on images from a variety of cell sources so that it applies broadly to different kinds of cell samples. We will evaluate it thoroughly on the hard cases:

- high cell density
- overlapping cells
- poor imaging quality (blur, noise, uneven backgrounds)

## Data sources

We want datasets that cover several cell types and imaging methods, so that the training data reflects the variation the model will meet in practice. We are still looking for more relevant datasets.

| Dataset | Status | Size | Description |
|---|---|---|---|
| [Roboflow Cell Counting](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5) | **Primary, in use** | 500 images (the ~1,200 figure counted augmented copies) | 640×640 brightfield images from 5 sources, with 23–91 cells each, boxed as Live or Dead. See `data/README.md`. |
| Synthetic Cell Images and Masks | Planned | 20,400 images | Black-and-white synthetic cell images with different amounts of blur applied. Each image is labeled with its cell count and blur level; the unblurred images are the ground truth. |
| CoNIC | Planned | n/a | Challenge dataset for segmenting, classifying, and counting six types of nuclei in histology images. |

## Stakeholders

- **Pathology laboratories and medical companies:** automated counting for diagnostics and quality control.
- **Research labs (academic and private):** cell culture monitoring, disease research, and other experiments that rely on counts.

How interested each group is will depend on how well the model performs on their kinds of images.

## Key performance indicators

We will start from available open-source models and use their performance as the **baseline**, then look for ways to improve on it.

| KPI | What it measures |
|---|---|
| Count accuracy | How close the predicted count is to the true count (see the metric definitions in `plan.md`). |
| R² | How much of the variance in true counts the predictions explain. |
| Compute efficiency | Inference time and resources per image, compared with counting by hand. |

## Current milestone: exploratory data analysis and preprocessing

- [ ] Upload or download the dataset into the workspace.
- [ ] Do basic exploratory data analysis suited to image data.
- [ ] Record the preprocessing decisions:
  1. **Inclusion/exclusion:** should all images go into the model, or do we need criteria for leaving some out?
  2. **Transformations:** what preprocessing must happen before images go into the model?
  3. **Splits:** how much data goes into train, validation, and test? Is that enough to keep a deep model from overfitting? (Deep models overfit much more easily than traditional machine learning.)

The detailed roadmap is in [`plan.md`](plan.md).
