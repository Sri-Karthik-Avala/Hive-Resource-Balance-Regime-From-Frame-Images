# Hive Resource Balance Regime from Frame Images

| | |
| --- | --- |
| Final rank | #7 |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-08-12 |

## Problem statement

### Overview

A colony-monitoring system receives one image of a single hive frame at a time. The image can contain cells associated with developing brood, stored food, open comb, and the surrounding frame. Your task is to infer the frame's **resource balance regime** from visual evidence alone. The regime is an image-level operational summary, not the name of one visible object.

The four regimes describe the dominant composition of the frame: brood-dominant, forage-dominant, mixed-resource, or sparse/uncertain. A useful model must learn fine-grained visual differences between visually similar cells and remain useful when the acquisition style changes between training and test images. The public training split has 979 images; the private test split has 245 images from eight held-out acquisition/sequence groups.

The operational label policy is composition-based: brood-dominant and forage-dominant require at least half of the non-frame cell evidence to belong to the corresponding resource family; sparse/uncertain is used for an empty-evidence share of at least 60% or no cell evidence; all remaining frames are mixed-resource.

### Dataset

### File descriptions

- `train.csv` -- Labeled image index with one row per training image.
- `test.csv` -- Unlabeled image index with one row per test image.
- `train/` -- 320x224 JPEG images referenced by `train.csv`.
- `test/` -- 320x224 JPEG images referenced by `test.csv`.
- `sample_submission.csv` -- Submission template with a valid random regime for each test image.

### Column descriptions

- `id` (string) -- Stable 12-character identifier for one image.
- `image_path` (string) -- Relative path to the image under the supplied public image folders.
- `target` (string, train only) -- One of `brood_dominant`, `forage_dominant`, `mixed_resource`, or `sparse_uncertain`.

### Evaluation

Submissions are scored using **Composition-Weighted Macro F1 (CWM-F1)**, which is maximized. It combines overall regime quality, performance on the two decision-critical regimes, and balanced per-regime recall:

```
ALL_REGIMES = [

    "brood_dominant", "forage_dominant", "mixed_resource", "sparse_uncertain"

]

overall_f1 = macro_f1(y_true, y_pred, labels=ALL_REGIMES)

critical_f1 = macro_f1(y_true, y_pred,

                       labels=["brood_dominant", "sparse_uncertain"])

balanced_accuracy = mean(

    recall(y_true, y_pred, regime) for regime in ALL_REGIMES

)

score = (0.55  *overall_f1 + 0.25*  critical_f1

         + 0.20 * balanced_accuracy)
```

All component scores are in `[0, 1]`, so the final score is also in `[0, 1]`.

### Submission

Submit a CSV file with one prediction for every row in `test.csv`.

- `id` (string) -- The 12-character identifier from `test.csv`.
- `target` (string) -- One of `brood_dominant`, `forage_dominant`, `mixed_resource`, or `sparse_uncertain`.

Example:

```
id,target

04bd67054490,brood_dominant

024dea6e68a3,mixed_resource
```

### Requirements

- The file must contain exactly one row for every image in `test.csv`.
- Every `id` from `test.csv` must appear exactly once.
- Every prediction must be one of the four allowed regime strings.
- File format: `.csv` only, with exact column names `id,target`.

### What Not To Use

- Do not recover image labels or cell annotations from files outside the supplied public folders; that bypasses the image-learning task.
- Do not use filename metadata, row order, or reverse maps from the hashed image IDs to infer a regime.
- Do not use a checkpoint trained specifically on this challenge's images or annotations. A general-purpose image backbone is allowed only when it is trained or fine-tuned as part of the submitted solution.
- Do not submit a fixed color-threshold or annotation-count lookup in place of a trained image model; the challenge evaluates visual generalization under a held-out acquisition group.
