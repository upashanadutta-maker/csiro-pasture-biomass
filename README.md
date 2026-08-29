# Pasture Biomass Prediction with DINOv3

An end-to-end computer vision and ML engineering project for predicting pasture biomass from field photographs using frozen DINOv3 visual representations and Ridge regression.

The project is based on the CSIRO pasture biomass dataset and focuses on leakage-aware validation, transfer learning, reproducible inference, automated testing, and continuous integration.

## Problem

Given a pasture image, predict five biomass quantities:

- `Dry_Clover_g`
- `Dry_Dead_g`
- `Dry_Green_g`
- `GDM_g`
- `Dry_Total_g`

The training data contains **357 independent pasture images**.

The original training CSV is in long format, with five target rows per image. For modeling, it is converted into one row per image with the five biomass targets represented as separate columns.

## Validation Strategy

Images collected during the same field collection session may share environmental and visual characteristics.

To reduce leakage between related observations, independent collection groups were defined using:

```text
State + Sampling_Date
```

The dataset contains **30 collection groups**.

Groups were assigned to three folds using state-stratified splitting at the group level.

| Fold | Images | Collection Groups |
|---|---:|---:|
| 0 | 127 | 10 |
| 1 | 91 | 10 |
| 2 | 139 | 10 |

No collection group appears in both training and validation data within a fold.

Performance is measured using the competition's globally weighted R² metric.

## Modeling Experiments

| Model | 3-Fold Weighted R² |
|---|---:|
| Per-target mean baseline | 0.2354 |
| Handcrafted image features + Ridge | 0.3409 |
| Handcrafted image features + Random Forest | 0.3671 |
| DINOv3-Small, 256×512 | 0.6643 |
| DINOv3-Base, 256×512 | 0.6847 |
| DINOv3-Large, 256×512 | 0.6911 |
| DINOv3-Large, 384×768, Ridge α=300 | 0.6945 |
| **DINOv3-Large, 384×768, Ridge α=100** | **0.6969** |

The largest improvement came from replacing manually engineered image statistics with pretrained visual representations.

Experiments with a nonlinear neural regression head showed greater overfitting than the regularized linear model, demonstrating that a more complex downstream model was not automatically better.

## Final Model

```text
Pasture image
      ↓
Resize to 384 × 768
      ↓
Frozen DINOv3-Large
      ↓
Patch-token representations
      ↓
Mean pooling
      ↓
1024-dimensional embedding
      ↓
StandardScaler
      ↓
Ridge Regression (alpha=100)
      ↓
5 biomass predictions
```

### Configuration

- Backbone: `vit_large_patch16_dinov3.lvd1689m`
- Input resolution: `384 × 768`
- Original 2:1 image aspect ratio preserved
- Backbone: frozen
- Pooling: mean of patch tokens
- Embedding dimension: 1024
- Regressor: Ridge regression
- Ridge alpha: 100
- Targets: five direct biomass outputs
- Grouped cross-validation weighted R²: **0.6969**

The reported `0.6969` is a cross-validation/model-selection estimate and is **not described as an untouched test-set score**.

## Why DINOv3?

Only 357 independent training images are available, so training a large vision model end-to-end would create substantial overfitting risk.

Instead, a pretrained DINOv3 backbone is used as a frozen feature extractor.

This provides rich visual representations while requiring only a lightweight regression layer to be fitted on the biomass dataset.

Experiments also showed that:

- DINOv3 representations substantially outperformed handcrafted image features.
- Base outperformed Small.
- Large outperformed Base.
- Mean pooling over patch tokens outperformed CLS pooling.
- Increasing input resolution provided a modest additional improvement.

## Project Structure

```text
csiro-pasture-biomass/
│
├── artifacts/
│   └── biomass_dinov3_large_ridge.joblib
│
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── predictor.py
│   └── inference.py
│
├── tests/
│   └── test_predictor.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── requirements.txt
├── smoke_test.py
└── README.md
```

## Inference

The production-facing inference interface accepts an image path and returns all five biomass predictions.

```python
from src.inference import BiomassInferencePipeline

pipeline = BiomassInferencePipeline(
    artifact_path="artifacts/biomass_dinov3_large_ridge.joblib",
    device="cuda"
)

predictions = pipeline.predict(
    "path/to/pasture_image.jpg"
)

print(predictions)
```

The returned structure is:

```python
{
    "Dry_Clover_g": ...,
    "Dry_Dead_g": ...,
    "Dry_Green_g": ...,
    "GDM_g": ...,
    "Dry_Total_g": ...
}
```

## Saved Model Artifact

The lightweight regression artifact contains:

- fitted `StandardScaler`
- trained Ridge model
- target ordering
- DINO backbone name
- input resolution
- pooling configuration
- embedding dimension
- selected Ridge alpha
- validation metadata

The artifact is only about **46 KB** because the pretrained DINOv3 backbone itself is loaded through `timm` at runtime rather than being stored inside the artifact.

## Training–Inference Parity

A parity check was performed between the experimental notebook pipeline and the reusable production feature extractor.

For the same image:

```text
Embedding max absolute difference : 0.0
Embedding mean absolute difference: 0.0
Prediction max difference         : 0.0
```

This confirms that the modular inference implementation reproduces the preprocessing and feature extraction used during model development.

## Testing

The project includes automated Pytest coverage for the prediction layer.

Tests verify that:

- a valid 1024-dimensional embedding produces five finite biomass predictions
- incorrect embedding dimensions are rejected
- one-dimensional malformed inputs are rejected
- multiple embeddings are rejected by the single-image prediction interface

Run the tests with:

```bash
python -m pytest -q
```

Current result:

```text
4 passed
```

## Continuous Integration

GitHub Actions automatically runs the lightweight test suite on pushes and pull requests.

The CI workflow intentionally tests the regression/prediction layer without downloading the full DINOv3 GPU stack, keeping automated validation fast and inexpensive.

## Key Findings

- Validation design matters substantially when images originate from related collection sessions.
- Hand-engineered color and vegetation statistics contain useful signal but are much weaker than pretrained visual representations.
- Increasing backbone capacity from DINOv3-Small to Base to Large improved validation performance.
- Patch-mean pooling performed better than CLS pooling.
- Increasing resolution from `256×512` to `384×768` provided only a modest gain.
- Physical output constraints were tested but did not improve the strongest DINOv3-Large model.
- Strong representation learning allowed a simple regularized linear regressor to outperform more complex downstream neural modeling.
- Model complexity was increased only when validation evidence supported it.
