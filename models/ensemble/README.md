# Ensemble Meta-Models

This directory stores trained ensemble meta-models for advanced fake voice detection.

## Purpose

These models are **optional** and provide advanced ensemble methods like stacking, which combine predictions from multiple base detectors (deep learning, signal features, behavioral, linguistic, anomaly) into a final score.

## What Goes Here

When trained, this directory will contain:

```
models/ensemble/
├── stacking.pkl              # Stacking ensemble (meta-learner)
├── weighted_ensemble.pkl     # Weighted ensemble configuration
└── voting_ensemble.pkl        # Voting ensemble configuration
```

## Training Ensemble Models

### Option 1: Programmatic Training

```python
from detection.ensemble import EnsembleDetector
import numpy as np

# Create ensemble detector
ensemble = EnsembleDetector(ensemble_method="stacking")

# Prepare training data
# predictions: (n_samples, n_models) array of base model predictions
# labels: (n_samples,) array of true labels (0=genuine, 1=fake)
predictions = np.array([
    [dl_score, signal_score, behavioral_score, linguistic_score, anomaly_score]
    for each_sample...
])
labels = np.array([0, 1, 0, ...])  # 0=genuine, 1=fake

# Train stacking meta-learner
ensemble.train_stacking(
    training_predictions=predictions,
    labels=labels,
    meta_learner="logistic"  # or "xgboost", "lightgbm"
)

# Save the trained ensemble
from backend.model_loader import ModelLoader
model_loader = ModelLoader()
model_loader.save_model(ensemble, "stacking", model_type="ensemble")
```

### Option 2: Using ModelLoader

```python
from backend.model_loader import ModelLoader
from detection.ensemble import EnsembleDetector

model_loader = ModelLoader()

# Train and save
ensemble = EnsembleDetector(ensemble_method="stacking")
# ... train on your data ...
model_loader.save_model(ensemble, "stacking", model_type="ensemble")

# Load later
ensemble = model_loader.load_ensemble_model("stacking")
```

## Meta-Learners

The stacking ensemble uses a meta-learner to combine base predictions:

- **`logistic`**: Logistic Regression (default, fast, good for small datasets)
- **`xgboost`**: XGBoost (requires `xgboost` package, better for large datasets)
- **`lightgbm`**: LightGBM (requires `lightgbm` package, fast and accurate)

## Requirements

- **Training data**: You need labeled examples (genuine vs fake audio)
- **Base model predictions**: Get predictions from all base detectors
- **Labels**: True labels (0=genuine, 1=fake)

## How It Works

1. **Base models** make predictions on training samples:
   - Deep Learning (ResNet18)
   - Signal Features
   - Behavioral Analysis
   - Linguistic Analysis
   - Anomaly Detection

2. **Meta-learner** learns how to combine these predictions optimally

3. **Stacking ensemble** uses the meta-learner to make final predictions

## Using Trained Ensemble

```python
from backend.model_loader import ModelLoader

model_loader = ModelLoader()
ensemble = model_loader.load_ensemble_model("stacking")

# Use in detection
result = ensemble.detect(
    deep_learning_score=0.8,
    behavioral_score=0.6,
    signal_score=0.7,
    linguistic_score=0.5,
    anomaly_score=0.9
)
```

## Current Status

**Empty**: Ensemble models not yet trained. The system uses default weighted ensemble.

## Default Behavior

Without trained ensemble models, the system uses:
- **Weighted ensemble**: Combines base scores with fixed weights
- **Dynamic weights**: Adjusts weights based on confidence levels
- **Voting**: Soft voting when multiple methods agree

These work well, but trained stacking can improve accuracy by learning optimal combinations.

## Notes

- Stacking requires labeled training data (genuine vs fake)
- Meta-learners like XGBoost/LightGBM need additional dependencies
- Trained ensembles are specific to your data distribution
- The default weighted ensemble works well for most cases

