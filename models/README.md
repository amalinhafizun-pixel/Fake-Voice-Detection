# Model Weights

This directory contains model weights for the Fake Voice Detection System.

## Directory Structure

```
models/
├── resnet18/        # ResNet18 spoof detection model
│   ├── resnet18_spectrogram_weights.pth
│   └── resnet18_spectrogram_full.pth
├── whisper/         # Whisper speech recognition (auto-downloaded)
├── anomaly/         # Trained anomaly detection models (optional)
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── lof.pkl
│   └── anomaly_ensemble.pkl
└── ensemble/        # Trained ensemble meta-models (optional)
    └── stacking.pkl
```

## Downloading Models

### Option 1: Automatic Download (Recommended)

Run the download script:

```bash
python models/download_models.py
```

### Option 2: Manual Download

#### AASIST Weights

1. Download from the official AASIST repository or HuggingFace
2. Place the weights file at: `models/aasist/weights.pth`

**Sources:**
- Official: https://github.com/clovaai/aasist
- HuggingFace (if available): Search for "aasist" models

#### Whisper Models

Whisper models are automatically downloaded on first use by the `openai-whisper` library.
They are cached in `~/.cache/whisper/` by default.

To pre-download:
```bash
python -c "import whisper; whisper.load_model('base')"
```

## Model Information

### ResNet18

- **Purpose**: Audio spoof detection (deep learning)
- **Architecture**: ResNet-18 adapted for mel spectrograms
- **Input**: Mel spectrograms (64 mel bands, 22050Hz)
- **Output**: Spoof probability (0-1)
- **Size**: ~45 MB
- **Training**: Train with `pretrain/train_model.py` on your dataset

### Whisper

| Model | Size | Parameters | Notes |
|-------|------|------------|-------|
| tiny | 39 MB | 39M | Fastest, less accurate |
| base | 74 MB | 74M | Good balance (default) |
| small | 244 MB | 244M | Better accuracy |
| medium | 769 MB | 769M | High accuracy |
| large | 1550 MB | 1550M | Best accuracy |

### Anomaly Detection Models

These are trained using scikit-learn and saved as pickle files.
They are **optional** and will be created automatically if training data is provided.

**Location**: `models/anomaly/`

**Training**: See `models/anomaly/README.md` for instructions.

**Current Status**: Empty - system uses heuristic-based detection.

### Ensemble Meta-Models

Trained ensemble models (like stacking) that combine base detector predictions.
These are **optional** - the system uses default weighted ensemble without them.

**Location**: `models/ensemble/`

**Training**: See `models/ensemble/README.md` for instructions.

**Current Status**: Empty - system uses default weighted ensemble.

## Notes

- Models are not included in the repository due to size
- First run may take longer as models are downloaded
- Ensure sufficient disk space for Whisper models
- GPU memory requirements vary by model size

