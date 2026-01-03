# Usage Examples

Practical examples for using the Fake Voice Detection System.

## Basic Examples

### Analyze Single File (GUI)

```bash
# Launch GUI
python main.py --PyQT

# Then:
# 1. Drag and drop audio file
# 2. Click "Analyze Audio"
# 3. View results
```

### Analyze Single File (Terminal)

```bash
python main.py --terminal --file audio.mp3
```

**Expected Output:**
```
✓ Analysis complete

============================================================
DETECTION RESULTS
============================================================

  Overall Score: 73%
  Risk Level:    HIGH
  Confidence:    HIGH

----------------------------------------
Component Scores:
  deep_learning        [████████████████░░░░] 78%
  behavioral           [██████████████░░░░░░] 68%
  signal               [████████████████░░░░] 75%
  linguistic           [██████████████░░░░░░] 70%
  anomaly              [██████████████░░░░░░] 65%

----------------------------------------
Interpretation:
  Strong evidence of AI-generated or synthetic speech.

============================================================
```

### JSON Output

```bash
python main.py --terminal --file audio.mp3 --json
```

**Expected Output:**
```json
{
  "overall_score": 0.73,
  "risk_level": "HIGH",
  "confidence": "HIGH",
  "component_scores": {
    "deep_learning": 0.78,
    "behavioral": 0.68,
    "signal": 0.75,
    "linguistic": 0.70,
    "anomaly": 0.65
  },
  "interpretation": "Strong evidence of AI-generated or synthetic speech."
}
```

## Backend Examples

### Apple Silicon

```bash
# Use MLX/MPS
python main.py --MSilicon --terminal --file audio.mp3
```

### NVIDIA GPU

```bash
# Use CUDA
python main.py --CUDA --terminal --file audio.mp3
```

### AMD GPU

```bash
# Use ROCm
python main.py --ROCm --terminal --file audio.mp3
```

## Batch Processing

### Process Directory (Terminal)

```bash
python main.py --terminal

# Select option 2: Batch analyze directory
# Enter path: /path/to/audio/files
```

**Expected Output:**
```
Found 5 audio files.

[1/5] Processing: sample1.mp3
✓ Analysis complete

[2/5] Processing: sample2.wav
✓ Analysis complete

...

============================================================
BATCH ANALYSIS SUMMARY
============================================================

  Total files:     5
  High risk:       2
  Medium risk:     1
  Low risk:        2

Export results to JSON? (y/n): y
Results saved to: /path/to/audio/files/analysis_results.json
```

## Python API Examples

### Basic Detection

```python
from audio.processor import AudioProcessor
from detection.ensemble import EnsembleDetector
from detection.signal_features import SignalFeatureExtractor
from detection.deep_learning import DeepLearningDetector
from detection.behavioral import BehavioralAnalyzer
from detection.linguistic import LinguisticAnalyzer
from detection.anomaly_detection import AnomalyDetector

# Load audio
processor = AudioProcessor()
audio = processor.load("audio.mp3")

# Signal features
signal_extractor = SignalFeatureExtractor(sample_rate=audio.sample_rate)
signal_features = signal_extractor.extract(audio.waveform)
signal_score = signal_extractor.compute_anomaly_score(signal_features)

# Deep learning
dl_detector = DeepLearningDetector()
dl_result = dl_detector.detect(audio.waveform, audio.sample_rate)

# Behavioral
behavioral = BehavioralAnalyzer(sample_rate=audio.sample_rate)
behavioral_result = behavioral.analyze(audio.waveform)

# Linguistic
linguistic = LinguisticAnalyzer()
linguistic_result = linguistic.analyze(audio.waveform, audio.sample_rate)

# Anomaly detection
feature_vector = signal_features.to_vector()
anomaly_detector = AnomalyDetector()
anomaly_result = anomaly_detector.predict(feature_vector)

# Ensemble
ensemble = EnsembleDetector()
result = ensemble.detect(
    deep_learning_score=dl_result.spoof_probability,
    behavioral_score=behavioral_result.anomaly_score,
    signal_score=signal_score,
    linguistic_score=linguistic_result.anomaly_score,
    anomaly_score=anomaly_result.anomaly_score
)

print(f"Overall Score: {result.overall_score:.2%}")
print(f"Risk Level: {result.risk_level}")
print(f"Interpretation: {result.interpretation}")
```

### Batch Processing Script

```python
from pathlib import Path
import json
from terminal.cli import TerminalInterface, CLIConfig

# Configure
config = CLIConfig(verbose=False, json_output=False)
interface = TerminalInterface(config)

# Process directory
audio_dir = Path("audio_files")
results = []

for audio_file in audio_dir.glob("*.mp3"):
    print(f"Processing: {audio_file.name}")
    result = interface.analyze_file(str(audio_file))
    result['file'] = str(audio_file)
    results.append(result)

# Summary
high_risk = sum(1 for r in results if r.get('risk_level') == 'HIGH')
print(f"\nTotal: {len(results)}, High Risk: {high_risk}")

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Feature Extraction Only

```python
from audio.processor import AudioProcessor
from detection.signal_features import SignalFeatureExtractor

processor = AudioProcessor()
audio = processor.load("audio.mp3")

extractor = SignalFeatureExtractor(sample_rate=audio.sample_rate)
features = extractor.extract(audio.waveform)

# Get feature dictionary
feature_dict = features.to_dict()
print(json.dumps(feature_dict, indent=2))

# Get feature vector for ML
feature_vector = features.to_vector()
print(f"Feature vector shape: {feature_vector.shape}")
```

### Custom Weights

```python
from detection.ensemble import EnsembleDetector

# Custom weights
custom_weights = {
    'deep_learning': 0.50,  # Increase DL weight
    'behavioral': 0.15,
    'signal': 0.15,
    'linguistic': 0.10,
    'anomaly': 0.10,
}

ensemble = EnsembleDetector(weights=custom_weights)
result = ensemble.detect(
    deep_learning_score=0.8,
    behavioral_score=0.6,
    signal_score=0.5,
    linguistic_score=0.7,
    anomaly_score=0.4
)

print(f"Score with custom weights: {result.overall_score:.2%}")
```

## Expected Results by Audio Type

### Genuine Human Speech

```
Overall Score: 15%
Risk Level: LOW
Confidence: HIGH

Component Scores:
- Deep Learning: 12%
- Behavioral: 18%
- Signal: 15%
- Linguistic: 14%
- Anomaly: 16%

Interpretation: Audio appears to be genuine human speech.
```

### AI-Generated Speech (ElevenLabs, etc.)

```
Overall Score: 82%
Risk Level: HIGH
Confidence: HIGH

Component Scores:
- Deep Learning: 89%
- Behavioral: 75%
- Signal: 80%
- Linguistic: 78%
- Anomaly: 85%

Interpretation: Strong evidence of AI-generated or synthetic speech.
```

### Voice Cloning

```
Overall Score: 68%
Risk Level: HIGH
Confidence: MEDIUM

Component Scores:
- Deep Learning: 72%
- Behavioral: 60%
- Signal: 70%
- Linguistic: 65%
- Anomaly: 68%

Interpretation: High likelihood of synthetic speech based on deep learning, signal.
```

### Borderline Case

```
Overall Score: 45%
Risk Level: MEDIUM
Confidence: LOW

Component Scores:
- Deep Learning: 50%
- Behavioral: 40%
- Signal: 45%
- Linguistic: 42%
- Anomaly: 48%

Interpretation: Mixed indicators - manual review recommended.
```

