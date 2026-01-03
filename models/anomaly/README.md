# Anomaly Detection Models

This directory stores trained anomaly detection models for improved fake voice detection.

## Purpose

These models are **optional** but improve detection accuracy when trained on genuine audio samples. The system works without them using heuristic-based detection.

## What Goes Here

When trained, this directory will contain:

```
models/anomaly/
├── isolation_forest.pkl      # Isolation Forest detector
├── one_class_svm.pkl         # One-Class SVM detector
├── lof.pkl                   # Local Outlier Factor detector
└── anomaly_ensemble.pkl      # Combined ensemble detector
```

## Training the Models

### Option 1: Using the Training Script

```bash
# Train on a directory of genuine audio files
python train_anomaly_models.py /path/to/genuine/audio --output models/anomaly
```

### Option 2: Using the Download Script

```bash
# Train anomaly models via download script
python models/download_models.py --train-anomaly /path/to/genuine/audio
```

### Option 3: Programmatic Training

```python
from train_anomaly_models import train_models

# Train on genuine audio directory
train_models(
    genuine_audio_dir="/path/to/genuine/audio",
    output_dir="models/anomaly"
)
```

## Requirements

- **Genuine audio samples**: Provide a directory with genuine (non-synthetic) speech samples
- **More samples = better models**: Aim for at least 50-100 samples for good results
- **Audio formats**: MP3, WAV, AAC, M4A, FLAC

## How It Works

1. **Extract features** from all genuine audio samples
2. **Train three detectors**:
   - Isolation Forest: Detects outliers in feature space
   - One-Class SVM: Learns boundary of normal speech
   - Local Outlier Factor: Detects local anomalies
3. **Combine into ensemble**: Weighted combination of all three methods

## Using Trained Models

The system automatically loads trained models if they exist in this directory. No code changes needed!

```python
from detection.anomaly_detection import AnomalyDetector

# Automatically loads trained models if available
detector = AnomalyDetector()

# Or load explicitly
detector = AnomalyDetector.load("models/anomaly/anomaly_ensemble.pkl")
```

## Current Status

**Empty**: Models not yet trained. The system uses heuristic-based detection.

To train models, provide genuine audio samples and run the training script.

## Notes

- Models are saved as pickle files (`.pkl`)
- Models are specific to the training data - retrain if your audio characteristics change
- These models complement (don't replace) the deep learning ResNet18 model
- Training takes a few minutes depending on dataset size

