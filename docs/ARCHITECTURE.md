# System Architecture

This document describes the architecture of the Fake Voice Detection System.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                  (PyQt6 GUI / Terminal CLI)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       Main Controller                            │
│                        (main.py)                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    Backend Device Manager                        │
│               (MLX / MPS / CUDA / ROCm / CPU)                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     Audio Processor                              │
│            (Load, Resample, Normalize, Preprocess)               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Signal-Level   │ │  Deep Learning  │ │   Behavioral    │
│    Features     │ │     (AASIST)    │ │    Analysis     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │          ┌────────┴────────┐          │
         │          ▼                 ▼          │
         │  ┌─────────────┐   ┌─────────────┐   │
         │  │  Linguistic │   │   Anomaly   │   │
         │  │  Analysis   │   │  Detection  │   │
         │  └──────┬──────┘   └──────┬──────┘   │
         │         │                 │          │
         └─────────┴────────┬────────┴──────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Ensemble Scoring                              │
│        (Weighted / Stacking / Voting / Dynamic Weights)          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Model Calibration                             │
│          (Platt Scaling / Isotonic / Temperature)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    Detection Result
```

## Module Organization

### Backend (`backend/`)

- **device_manager.py**: Hardware detection, backend selection, device initialization
- **model_loader.py**: Model loading, weight management, caching

### Audio (`audio/`)

- **processor.py**: Audio file loading, format conversion, preprocessing

### Detection (`detection/`)

- **signal_features.py**: MFCC, spectral flux, pitch, HNR, phase
- **deep_learning.py**: AASIST model implementation and inference
- **behavioral.py**: Temporal patterns, pauses, breathing
- **linguistic.py**: Whisper transcription, NLP analysis
- **anomaly_detection.py**: Isolation Forest, One-Class SVM, LOF
- **feature_engineering.py**: Feature selection, normalization, PCA
- **ensemble.py**: Score combination methods
- **calibration.py**: Probability calibration

### ML Utilities (`ml/`)

- **feature_selector.py**: Feature selection algorithms
- **hyperparameter_tuning.py**: Optuna optimization
- **cross_validation.py**: CV utilities
- **model_evaluation.py**: Metrics and evaluation

### UI (`ui/`)

- **main_window.py**: Main application window
- **file_upload.py**: File upload widget
- **results_display.py**: Results visualization

### Terminal (`terminal/`)

- **cli.py**: Command-line interface

## Data Flow

### 1. Input Stage
```
Audio File → Audio Processor → Preprocessed Waveform
                ↓
         (16kHz, mono, normalized)
```

### 2. Feature Extraction Stage
```
Preprocessed Waveform
        │
        ├─→ Signal Feature Extractor → Signal Features
        │
        ├─→ Deep Learning Detector → DL Score
        │
        ├─→ Behavioral Analyzer → Behavioral Score
        │
        ├─→ Linguistic Analyzer → Linguistic Score
        │
        └─→ Anomaly Detector → Anomaly Score
```

### 3. Scoring Stage
```
All Scores → Ensemble Detector → Combined Score
                   │
                   ▼
            Model Calibrator → Calibrated Score
                   │
                   ▼
             Risk Assessment
```

### 4. Output Stage
```
Detection Result
       │
       ├─→ GUI Display
       │
       ├─→ Terminal Output
       │
       └─→ File Export (JSON/CSV)
```

## Component Interactions

### Backend Selection
1. Parse CLI arguments (--MSilicon, --CUDA, --ROCm)
2. Detect available hardware
3. Initialize appropriate backend
4. Configure model loading path

### Model Loading
1. Check for local weights
2. Download if missing (with user consent)
3. Load to appropriate device
4. Cache for subsequent use

### Ensemble Scoring
```python
final_score = (
    0.35 * deep_learning_score +
    0.20 * behavioral_score +
    0.15 * signal_score +
    0.15 * linguistic_score +
    0.15 * anomaly_score
)
```

## Design Decisions

### Why Multiple Detection Methods?
- No single method is perfect
- Different methods catch different spoofing techniques
- Ensemble provides robustness

### Why Backend Abstraction?
- Support diverse hardware (Apple Silicon, NVIDIA, AMD)
- Graceful fallback to CPU
- Optimize for available resources

### Why Calibration?
- Raw scores may not be well-calibrated
- Calibrated probabilities improve decision-making
- Better threshold selection for risk levels

## Extensibility

### Adding New Detection Methods
1. Create new module in `detection/`
2. Implement analysis function returning score 0-1
3. Add to ensemble weights in `config.py`
4. Update ensemble detector

### Adding New Backends
1. Implement backend check in `device_manager.py`
2. Add initialization logic
3. Handle model conversion if needed
4. Update CLI arguments

