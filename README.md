# Pasture Biomass from Photographs — CSIRO Image2Biomass

Predicting five dry-matter biomass measurements (grams) from a single overhead
photograph of a 70×30 cm pasture quadrat, using frozen DINOv3 features.

**Weighted log-space R²: 0.6861** (5-fold grouped cross-validation, 357 images)

---

## Result

| target | weight | R² (log) | contribution |
|---|---|---|---|
| Dry_Green_g | 0.10 | 0.7125 | 0.0712 |
| Dry_Dead_g | 0.10 | 0.4591 | 0.0459 |
| Dry_Clover_g | 0.10 | 0.7615 | 0.0761 |
| GDM_g | 0.20 | 0.7026 | 0.1405 |
| Dry_Total_g | 0.50 | 0.7045 | 0.3522 |
| **weighted total** | | | **0.6861** |

Reference points on the same metric:

| | score |
|---|---|
| predicting the training mean | −0.0404 |
| **this model (frozen backbone, no fine-tuning)** | **0.6861** |
| public DINOv3 baseline (competition leaderboard) | 0.70 |
| 4th place, gold (competition private leaderboard) | 0.66 |

The competition closed in January 2026. The number above is cross-validation on
the 357 public training images, **not** a leaderboard score, and the two are not
directly comparable — the winning team's score dropped from 0.74 public to 0.64
private, so a held-out CV figure on this dataset should be read as optimistic.

---

## The metric

The competition scores a weighted sum of per-target R², computed on log(1+y):

```
score = 0.1·R²(Green) + 0.1·R²(Dead) + 0.1·R²(Clover)
      + 0.2·R²(GDM)   + 0.5·R²(Total)
```

Two consequences drive every design decision in this repo.

**Dry_Total is half the score.** It contributes 0.3522 of the 0.6861 above —
51% from one target. Improving Dry_Clover from 0.76 to a perfect 1.00 would add
0.024; the same improvement on Dry_Total would add 0.148.

**log(1+y) reweights the samples.** The derivative of log(1+y) is 1/(1+y), so a
1 g error on a 1 g sample costs 50× what it costs on a 100 g sample. Sparse
quadrats matter far more than the gram-scale distribution suggests.

An earlier version of this project reported *pooled* R² — a single R² across all
1,785 rows in grams, with all five targets in one pool. That metric is not the
competition's and it is optimistic: it credits the model for knowing that clover
averages ~6 g while total averages ~45 g, so predicting each target's mean scores
about 0.20 on it. The same predictions that score **0.6861** here score **0.7304**
pooled. The gap is not a constant offset — at an earlier stage of this work it was
0.11 — so old numbers cannot be converted, only re-scored.

---

## Main finding: the field metadata is redundant

`train.csv` includes two field measurements, `Pre_GSHH_NDVI` (a greenness index)
and `Height_Ave_cm` (mean canopy height). Neither is present in `test.csv`.

An earlier iteration of this project, using ConvNeXt-Tiny at 224 px, found that
metadata alone reached most of the achievable accuracy and that the image added
little on top. That made the missing test-time metadata look like a hard blocker.

Measured here, out-of-fold, with a stronger encoder:

| | R² |
|---|---|
| NDVI predicted from the image alone | 0.8841 |
| canopy height predicted from the image alone | 0.8303 |

And feeding those predicted values back into the biomass head:

| | score |
|---|---|
| image features only | 0.6839 |
| image features + predicted metadata | 0.6872 |
| difference | **+0.0033** |

The metadata is recoverable from the photograph, and supplying it adds nothing
measurable. It was never carrying independent information — it was a shortcut to
something already visible in the image, and a sufficiently strong encoder reaches
it directly. The blocker dissolves rather than requiring a workaround.

(This is the auxiliary / privileged-information approach used by the 4th-place
solution, which reported a similarly small gain of about +0.01.)

---

## Method

**Backbone.** `vit_huge_plus_patch16_dinov3.lvd1689m` — 840.5M parameters, frozen,
forward pass only. Images resized 2000×1000 → 800×800, giving 50×50 = 2500 patch
tokens of 1280 dimensions each, plus 5 prefix tokens (1 CLS + 4 register).
Position embeddings are interpolated from the 256 px pretraining resolution.

**Two-phase design.** Feature extraction runs once (21 minutes on a Kaggle P100,
1,428 forward passes for 357 images × 4 flip views) and is cached to a 7.3 MB
array. Every subsequent experiment trains a small MLP head on that cache in
seconds rather than hours. This is what made the experiment count below
affordable.

**Head.** LayerNorm → Linear(1280, 512) → GELU → Dropout(0.5) → Linear(512, 5).
Targets are log1p-transformed then standardised. Loss is per-target squared error
weighted by the competition weights `[0.1, 0.1, 0.1, 0.2, 0.5]` — since
R² = 1 − SSE/SS_tot, minimising weighted squared error on log1p targets directly
maximises the scored quantity.

**Validation.** `StratifiedGroupKFold`, stratified on `State` and grouped on
`Sampling_Date`. Images from one sampling session share site, growth stage,
species, weather and lighting; splitting them across folds lets the model
recognise the session rather than estimate biomass.

**Measurement floor.** Adjacent epochs of a single training run on this dataset
differ by up to 0.077, so five-fold CV resolves roughly ±0.025. Differences
smaller than that are reported as ties throughout, not ranked.

---

## What was tested

**Pooling — the one result that clears the noise floor.**

| feature | dims | score |
|---|---|---|
| patch-token mean | 1280 | **0.6755** |
| CLS + patch mean, flip-averaged | 2560 | 0.6737 |
| CLS + patch mean, all 4 views concatenated | 10240 | 0.6605 |
| CLS + patch mean | 2560 | 0.6547 |
| CLS token | 1280 | 0.6086 |

Averaging the 2500 patch tokens beats the CLS token by 0.067. Biomass is a
texture-density measurement across the whole frame, which is what patch-mean
pooling computes; the CLS token summarises image-level semantics instead.

Nothing more elaborate helped. Concatenating all four flip views cost 6× the
compute (571 s vs 89 s) for no gain.

**Head capacity — a flat plateau with one real signal.**

The 3×3 grid over hidden width and dropout spans only 0.021, so no cell is
meaningfully best. But dropout 0.5 beat 0.3 beat 0.1 at *all three* widths —
three independent runs agreeing in direction is signal where any single 0.01 gap
is not. The model wants more regularisation than the grid offered, which is what
357 training images against 1280 features predicts. Hidden width had no effect.

---

## Limitations

- Cross-validation on 357 images, not a leaderboard score. Read as optimistic.
- The backbone is frozen; fine-tuning is untested here and would be expected to
  improve on this.
- Images are squashed 2000×1000 → 800×800, distorting the 2:1 aspect ratio.
- `Dry_Dead_g` (0.4591) is the weakest target, but its 0.10 weight caps any
  achievable gain at about 0.04, so it was not pursued.
- The quadrat's cardboard backing is visible in some frames. The 4th-place team
  reported +0.01 from cropping it manually; no automatic method is implemented
  here, and manual cropping is not possible on a hidden test set.

---

## Repository

```
README.md              this file
csiro_metric.py        competition metric, grouped folds, mean baseline
notebook.ipynb         feature extraction + head experiments, end to end
frozen_config.json     winning configuration and per-target scores
oof_frozen.npy         out-of-fold predictions (357 × 5, grams)
```

`csiro_metric.py` is standalone — `weighted_log_r2`, `make_folds` and
`mean_baseline` can be dropped into any notebook on this competition.

Competition data is not redistributed here; it is available from the
[competition page](https://www.kaggle.com/competitions/csiro-biomass).

---

## Credits

Design decisions taken from the publicly published 4th-place solution
([Jatin-Mehra119](https://github.com/Jatin-Mehra119/Kaggle-CSIRO-4th-Position-Solution-)):
800 px input resolution, freezing the first 50% of transformer blocks, the
weighted loss matching the metric weights, and the auxiliary metadata-prediction
idea evaluated above.

Dataset: CSIRO, via the Kaggle *Image2Biomass Prediction* competition.
