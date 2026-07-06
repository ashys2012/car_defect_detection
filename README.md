# CarDD damage detector + active learning check

 YOLOv11 damage detector built against a public dataset, with an actual train/val/test discipline, plus a working test of whether the active-learning idea from the design doc holds up in practice.

## Layout

```
download_data.py                    pulls CarDD via kagglehub, copies it locally
train_baseline.py                   fine-tunes yolo11n.pt on CarDD
evaluate.py                         per-class test report, worst class first
active_learning/mc_dropout.py       MC-Dropout + BaaL uncertainty scoring for YOLO
run_active_learning_experiment.py   active selection vs random selection, head to head
results/                            small evidence files (plots, logs, csv) from the runs below
```

## Setup

```bash
uv venv --python 3.10
source .venv/bin/activate
# match torch to your CUDA driver (check nvidia-smi first), e.g.:
uv pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
python download_data.py   # needs a Kaggle API token - see the docstring
```

## Dataset

Used the Kaggle mirror `gabrielfcarvalho/cardd-with-yolo-annotations-images-labels` rather than the official CarDD release, because the official one sits behind a licence request form on cardd-ustc.github.io. The mirror has the same images, already converted to YOLO box format.

Split is 2816 / 810 / 374 (train/val/test). I checked and this matches the official CarDD paper's published split exactly, so I kept it as-is instead of reshuffling — the paper's authors presumably already made sure it's scene/vehicle-disjoint, and a fresh random split without vehicle IDs could easily leak near-duplicate photos of the same car across train and test without me noticing. The test set only gets touched once, for the final numbers below; `val` is what training uses for early stopping and checkpoint selection.

Two things about this dataset that are worth being upfront about:

CarDD's actual 6 labels are `dent, scratch, crack, glass_shatter, lamp_broken, tire_flat`.

**There's no "clean car" class.** Every single image has at least one damage box. The design doc calls for a No Defect / background class (Sec. 3), and CarDD just doesn't have one. A real pilot would need photos of undamaged panels collected separately.

Class balance in the training set, for reference: scratch 2560, dent 1806, lamp_broken 494, glass_shatter 475, crack 651, tire_flat 225. crack and tire_flat are the thin end of the wedge, and it shows in the results below.

## Baseline results

`python train_baseline.py --epochs 100`, yolo11n.pt fine-tuned on the full 2816-image train set.

| | mAP50 | mAP50-95 |
|---|---|---|
| **test set** | **0.705** | **0.552** |

Per class (worst to best), from `evaluate.py`:

| class | AP50 | AP50-95 |
|---|---|---|
| crack | 0.367 | 0.200 |
| dent | 0.582 | 0.330 |
| scratch | 0.593 | 0.327 |
| lamp_broken | 0.817 | 0.693 |
| tire_flat | 0.884 | 0.836 |
| glass_shatter | 0.988 | 0.927 |

crack is clearly the weak point, and it's not a mystery why: it's the rarest class in training (651 instances vs 2560 for scratch) and visually it's a thin, low-contrast line that's easy to confuse with glare or a reflection on a phone photo. A nano model with no special handling for this isn't going to be great at it, and that's fine — the brief is explicit that raw mAP isn't the thing being graded here, the process is. Plots and the full per-epoch curve are in `results/`.

## Active Learning Implementation

Active Learning works with a small labelling budget, using the model's own uncertainty (MC Dropout) to decide which vehicles to label next, instead of picking at random. BaaL is the library used for this, but BaaL's uncertainty heuristics (BALD, entropy) are built for classification — one image, one fixed-size probability vector that sums to 1. YOLO11 has no dropout layers anywhere in the network, so for BaaL to switch on — MC Dropout has to be added, not just enabled.

Added in `active_learning/mc_dropout.py` :

1. Added BaaL's `Dropout2d` into just the classification branch of the Detect head. Left the box-regression branch without active learning.
2. Ran 8 stochastic passes per image on the same input tensor, reading the raw per-anchor class scores before NMS each time.
3. For each class, built a proper two-way distribution (`p`, `1-p`) per anchor per pass and fed that into BaaL's actual `BALD().compute_score()`. Feeding it the raw multi-label tensor directly would have made BaaL silently softmax-normalize across classes that aren't mutually exclusive, which would have been wrong.
4. Took the worst class per anchor, then averaged the top 20 anchors, to get one uncertainty number per image without background anchors drowning out the signal.

Then the actual test: split the training images into a 422-image seed set and a 2394-image pool. Trained on seed only, used that model to rank the pool by uncertainty, then trained two more models — one on seed + the 422 most uncertain pool images, one on seed + 422 random pool images. Same size, same hyperparameters, only the selection method differs; both scored on the untouched test set.

In this particular case, 15% of the total samples were labelled.

| training set | n images | mAP50 | mAP50-95 |
|---|---|---|---|
| seed only | 422 | 0.540 | 0.414 |
| seed + random | 844 | 0.610 | 0.473 |
| seed + active (uncertainty-selected) | 844 | 0.616 | 0.475 |

Active selection beat random by **+0.0055 mAP50**.

## What I'd do with more time

- Run a second and third active-learning round — one round only tells whether the method works.
- Try segmentation instead of boxes. CarDD ships instance masks, and crack in particular is probably better described by a thin mask than a box. Stuck to detection for the pilot.
- Extend the MC-Dropout bridge to the box-regression branch too — right now it only measures class uncertainty.
- Fix the two dataset gaps above (no background class, no mirror/panel_damage).
