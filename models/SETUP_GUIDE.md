# Model Setup Guide

Complete guide for downloading and setting up all required models.

## Table of Contents

1. [AASIST Weights](#aasist-weights)
2. [Whisper Models](#whisper-models)
3. [Anomaly Detection Models](#anomaly-detection-models)
4. [Ensemble Models](#ensemble-models)

---

## AASIST Weights

### What is AASIST?

AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) is the main deep learning model for spoof detection. It won the ASVspoof 2021 challenge.

### Where to Download

#### Option 1: Official AASIST Repository (Recommended)

1. **Visit the repository:**
   ```
   https://github.com/clovaai/aasist
   ```

2. **Navigate to releases or check the README:**
   - Look for "Pre-trained models" section
   - Check for download links or instructions
   - May be in a "models" or "weights" folder

3. **Download the weights file:**
   - Usually named `weights.pth`, `model.pth`, or `AASIST.pth`
   - May be in a zip file that needs extraction

4. **Place the file:**
   ```bash
   # Create directory if needed
   mkdir -p models/aasist
   
   # Move or copy the downloaded file
   cp /path/to/downloaded/weights.pth models/aasist/weights.pth
   ```

#### Option 2: HuggingFace

1. **Search for AASIST on HuggingFace:**
   ```
   https://huggingface.co/models?search=aasist
   ```

2. **Select a model repository:**
   - Look for official or verified models
   - Check model cards for download instructions

3. **Download using HuggingFace CLI:**
   ```bash
   # Install huggingface-hub if needed
   pip install huggingface-hub
   
   # Download model (example - replace with actual model name)
   huggingface-cli download clovaai/aasist --local-dir models/aasist
   ```

4. **Or download manually:**
   - Click "Files and versions" tab
   - Download the `.pth` or `.pt` file
   - Place in `models/aasist/weights.pth`

#### Option 3: Direct Download (if available)

Some researchers share direct download links. Check:
- Paper repositories (arXiv, research papers)
- Conference proceedings
- GitHub issues/discussions

### Verification

After downloading, verify the file:

```bash
# Check file exists
ls -lh models/aasist/weights.pth

# Should be 2-10 MB typically
# If file is much smaller (< 1MB), it might be corrupted
```

### If You Can't Find AASIST Weights

**Don't worry!** The system will still work:

1. The code includes a **fallback detection method** using signal features
2. It will automatically use feature-based detection if AASIST weights are missing
3. You can still use all other detection methods (behavioral, linguistic, anomaly)

The system will log a warning but continue to function.

---

## Whisper Models

### Automatic Download (Recommended)

Whisper models are **automatically downloaded** when you first use the linguistic analyzer:

```python
# This will auto-download on first use
from detection.linguistic import LinguisticAnalyzer
analyzer = LinguisticAnalyzer(whisper_model="base")
result = analyzer.analyze(waveform, sample_rate)
```

### Manual Pre-download

To pre-download a Whisper model:

```bash
# Using Python
python -c "import whisper; whisper.load_model('base')"

# Or use the download script
python models/download_models.py --whisper-size base
```

### Model Sizes

| Size | Parameters | Disk Space | Speed | Accuracy |
|------|------------|------------|-------|----------|
| tiny | 39M | ~39 MB | Fastest | Lower |
| base | 74M | ~74 MB | Fast | Good (default) |
| small | 244M | ~244 MB | Medium | Better |
| medium | 769M | ~769 MB | Slow | High |
| large | 1550M | ~1.5 GB | Slowest | Best |

**Recommendation:** Start with `base` for good balance.

### Cache Location

Models are cached in:
```
~/.cache/whisper/
```

To change location:
```python
import os
os.environ['WHISPER_CACHE_DIR'] = '/custom/path'
```

---

## Anomaly Detection Models

### What Are They?

Anomaly detection models (Isolation Forest, One-Class SVM, LOF) learn patterns from "normal" (genuine) speech to detect outliers (synthetic speech).

### Do You Need Them?

**Short answer: No, not required for basic use.**

The system will:
- Use default parameters if models aren't trained
- Still provide anomaly scores
- Work effectively without pre-trained models

### When to Train Them

Train anomaly models if you:
- Have a dataset of genuine speech samples
- Want improved accuracy on your specific use case
- Are processing many files and want consistency

### How to Train

#### Step 1: Prepare Training Data

Collect genuine (non-synthetic) audio files:

```python
# Example: Prepare training data
from audio.processor import AudioProcessor
from detection.signal_features import SignalFeatureExtractor
import numpy as np

processor = AudioProcessor()
extractor = SignalFeatureExtractor()

# Load genuine audio samples
genuine_files = [
    "genuine1.wav",
    "genuine2.wav",
    "genuine3.wav",
    # ... more genuine samples
]

# Extract features
features_list = []
for file_path in genuine_files:
    audio = processor.load(file_path)
    features = extractor.extract(audio.waveform, audio.sample_rate)
    feature_vector = features.to_vector()
    features_list.append(feature_vector)

# Convert to numpy array
X_train = np.array(features_list)
```

#### Step 2: Train the Models

```python
from detection.anomaly_detection import AnomalyDetector
from backend.model_loader import ModelLoader

# Create detector
detector = AnomalyDetector(
    methods=['isolation_forest', 'one_class_svm', 'lof']
)

# Fit on genuine data
detector.fit(X_train)

# Save models
model_loader = ModelLoader()
model_loader.save_model(detector, 'anomaly_ensemble', model_type='anomaly')

# Or save individual models
from detection.anomaly_detection import IsolationForestDetector, OneClassSVMDetector

iso_forest = IsolationForestDetector()
iso_forest.fit(X_train)
model_loader.save_model(iso_forest, 'isolation_forest', model_type='anomaly')
```

#### Step 3: Use Trained Models

```python
from backend.model_loader import ModelLoader

model_loader = ModelLoader()
detector = model_loader.load_anomaly_model('anomaly_ensemble')

# Use for detection
result = detector.predict(feature_vector)
```

### Training Script Example

Create `train_anomaly_models.py`:

```python
#!/usr/bin/env python3
"""Train anomaly detection models on genuine speech data."""

import sys
from pathlib import Path
from audio.processor import AudioProcessor
from detection.signal_features import SignalFeatureExtractor
from detection.anomaly_detection import AnomalyDetector
from backend.model_loader import ModelLoader
import numpy as np

def train_models(genuine_audio_dir: str):
    """Train anomaly models on genuine audio."""
    
    audio_dir = Path(genuine_audio_dir)
    processor = AudioProcessor()
    extractor = SignalFeatureExtractor()
    model_loader = ModelLoader()
    
    # Collect all audio files
    audio_files = []
    for ext in ['.mp3', '.wav', '.aac', '.flac']:
        audio_files.extend(audio_dir.glob(f'*{ext}'))
    
    if not audio_files:
        print(f"No audio files found in {genuine_audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print("Extracting features...")
    
    # Extract features
    features_list = []
    for i, file_path in enumerate(audio_files, 1):
        print(f"  [{i}/{len(audio_files)}] {file_path.name}")
        try:
            audio = processor.load(file_path)
            features = extractor.extract(audio.waveform, audio.sample_rate)
            feature_vector = features.to_vector()
            features_list.append(feature_vector)
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if not features_list:
        print("No features extracted. Check audio files.")
        return
    
    X_train = np.array(features_list)
    print(f"\nTraining on {len(X_train)} samples...")
    
    # Train detector
    detector = AnomalyDetector()
    detector.fit(X_train)
    
    # Save
    model_loader.save_model(detector, 'anomaly_ensemble', model_type='anomaly')
    print(f"\n✓ Models saved to models/anomaly/")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_anomaly_models.py <genuine_audio_directory>")
        sys.exit(1)
    
    train_models(sys.argv[1])
```

Run it:
```bash
python train_anomaly_models.py /path/to/genuine/audio/files
```

---

## Ensemble Models

### What Are They?

Ensemble meta-models (like stacking) learn how to best combine the base detection methods.

### Do You Need Them?

**No, not required.** The system uses weighted averaging by default, which works well.

### When to Train Them

Train ensemble models if you:
- Have labeled training data (genuine vs. synthetic)
- Want to optimize the combination of detection methods
- Need maximum accuracy for your specific use case

### How to Train

#### Step 1: Prepare Labeled Data

You need:
- Audio files labeled as genuine (0) or synthetic (1)
- Or predictions from base models with known labels

```python
# Example: Prepare training data
import numpy as np
from audio.processor import AudioProcessor
from detection.deep_learning import DeepLearningDetector
from detection.behavioral import BehavioralAnalyzer
from detection.signal_features import SignalFeatureExtractor
from detection.linguistic import LinguisticAnalyzer
from detection.anomaly_detection import AnomalyDetector

processor = AudioProcessor()
dl_detector = DeepLearningDetector()
behavioral = BehavioralAnalyzer()
signal_extractor = SignalFeatureExtractor()
linguistic = LinguisticAnalyzer()
anomaly_detector = AnomalyDetector()

# Your labeled data
audio_files = [
    ("genuine1.wav", 0),
    ("genuine2.wav", 0),
    ("synthetic1.wav", 1),
    ("synthetic2.wav", 1),
    # ... more labeled samples
]

# Extract predictions
predictions = []
labels = []

for file_path, label in audio_files:
    audio = processor.load(file_path)
    
    # Get predictions from each method
    dl_result = dl_detector.detect(audio.waveform, audio.sample_rate)
    behavioral_result = behavioral.analyze(audio.waveform)
    signal_features = signal_extractor.extract(audio.waveform)
    signal_score = signal_extractor.compute_anomaly_score(signal_features)
    linguistic_result = linguistic.analyze(audio.waveform, audio.sample_rate)
    feature_vector = signal_features.to_vector()
    anomaly_result = anomaly_detector.predict(feature_vector)
    
    # Combine predictions
    pred_vector = np.array([
        dl_result.spoof_probability,
        behavioral_result.anomaly_score,
        signal_score,
        linguistic_result.anomaly_score,
        anomaly_result.anomaly_score
    ])
    
    predictions.append(pred_vector)
    labels.append(label)

X_train = np.array(predictions)
y_train = np.array(labels)
```

#### Step 2: Train Stacking Ensemble

```python
from detection.ensemble import EnsembleDetector

# Create ensemble
ensemble = EnsembleDetector(ensemble_method='weighted')

# Train stacking meta-learner
ensemble.train_stacking(
    training_predictions=X_train,
    labels=y_train,
    meta_learner='logistic'  # or 'xgboost', 'lightgbm'
)

# Save
from backend.model_loader import ModelLoader
model_loader = ModelLoader()
model_loader.save_model(ensemble, 'stacking', model_type='ensemble')
```

---

## Quick Start Summary

### Minimum Setup (Works Immediately)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pre-download Whisper (optional, auto-downloads anyway)
python -c "import whisper; whisper.load_model('base')"

# 3. Run the app (AASIST will use fallback if weights missing)
python main.py --PyQT
```

### Full Setup (Best Accuracy)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download AASIST weights (see instructions above)
#    Place in: models/aasist/weights.pth

# 3. Pre-download Whisper
python models/download_models.py --whisper-size base

# 4. (Optional) Train anomaly models if you have genuine audio
python train_anomaly_models.py /path/to/genuine/audio

# 5. Run the app
python main.py --PyQT
```

---

## Troubleshooting

### AASIST Weights Not Found

**Symptom:** Warning message about missing AASIST weights

**Solution:**
- System will use fallback detection (still works!)
- Or download weights following instructions above
- Check file is named `weights.pth` in `models/aasist/`

### Whisper Download Fails

**Symptom:** Error downloading Whisper model

**Solution:**
```bash
# Check internet connection
# Try manual download
pip install --upgrade openai-whisper
python -c "import whisper; whisper.load_model('base')"
```

### Anomaly Models Not Working

**Symptom:** Anomaly detection always returns same score

**Solution:**
- This is normal if models aren't trained
- Train models on your genuine audio data
- Or use default parameters (still functional)

---

## Need Help?

- Check the main [README.md](../docs/README.md)
- See [INSTALLATION.md](../docs/INSTALLATION.md) for setup
- Review [EXAMPLES.md](../docs/EXAMPLES.md) for usage

