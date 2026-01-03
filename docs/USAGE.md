# Usage Guide

How to use the Fake Voice Detection System.

## Command-Line Interface

### Basic Usage

```bash
# Launch GUI (default)
python main.py

# Launch GUI explicitly
python main.py --PyQT

# Use terminal mode
python main.py --terminal
```

### Backend Selection

```bash
# Apple Silicon (MLX/MPS)
python main.py --MSilicon

# NVIDIA CUDA
python main.py --CUDA

# AMD ROCm
python main.py --ROCm

# Auto-detect (default)
python main.py
```

### Analyzing Files

```bash
# Analyze single file in terminal
python main.py --terminal --file audio.mp3

# Output as JSON
python main.py --terminal --file audio.mp3 --json

# Save results to file
python main.py --terminal --file audio.mp3 --output results.json

# Verbose mode
python main.py --terminal --file audio.mp3 --verbose
```

## GUI Mode

### Main Window

The GUI provides:
- **File Upload**: Drag-and-drop or browse for files
- **Backend Selector**: Choose compute backend
- **Analyze Button**: Start analysis
- **Progress Bar**: Shows analysis progress
- **Results Display**: Visual results with scores

### Uploading Files

1. **Drag and Drop**: Drag audio file onto the upload area
2. **Browse**: Click "Browse Files" and select a file

Supported formats: MP3, WAV, AAC, M4A, FLAC, OGG

### Understanding Results

**Overall Score**: 0-100% likelihood of being synthetic

**Risk Levels**:
- 🟢 LOW (0-30%): Likely genuine
- 🟡 MEDIUM (30-60%): Some concerns
- 🔴 HIGH (60-100%): Likely synthetic

**Component Scores**:
- Deep Learning: AASIST model score
- Behavioral: Temporal pattern analysis
- Signal: Acoustic feature analysis
- Linguistic: Speech content analysis
- Anomaly: Outlier detection score

### Exporting Results

- **Export JSON**: Full results in JSON format
- **Export CSV**: Tabular format for spreadsheets

## Terminal Mode

### Interactive Mode

```bash
python main.py --terminal
```

Menu options:
1. Analyze audio file
2. Batch analyze directory
3. Settings
4. Exit

### Non-Interactive Mode

```bash
# Analyze and exit
python main.py --terminal --file audio.mp3

# JSON output for scripting
python main.py --terminal --file audio.mp3 --json
```

### Batch Processing

In interactive mode:
1. Select "Batch analyze directory"
2. Enter directory path
3. All audio files will be analyzed
4. Summary shown at end
5. Option to export results

## Best Practices

### Audio Quality

- Use audio at least 3 seconds long
- Prefer WAV format for best quality
- Ensure clear speech without heavy compression
- Avoid heavily processed audio

### Interpreting Results

- **HIGH risk** doesn't guarantee synthetic speech
- **LOW risk** doesn't guarantee genuine speech
- Consider the context and source
- Use results as one factor in assessment

### Performance Tips

- Use GPU backend when available
- For batch processing, use terminal mode
- Smaller Whisper models are faster
- Close other GPU applications

## Configuration

### Modifying Weights

Edit `config.py` to adjust detection weights:

```python
# Default weights
deep_learning_weight: float = 0.35
behavioral_weight: float = 0.20
signal_weight: float = 0.15
linguistic_weight: float = 0.15
anomaly_weight: float = 0.15
```

### Changing Thresholds

```python
# Risk level thresholds
low_risk_threshold: float = 0.3
medium_risk_threshold: float = 0.6
```

### Model Selection

```python
# Whisper model size
whisper_model_size: str = "base"  # tiny, base, small, medium, large
```

## Scripting

### Python API

```python
from audio.processor import AudioProcessor
from detection.ensemble import EnsembleDetector
from detection.signal_features import SignalFeatureExtractor
from detection.deep_learning import DeepLearningDetector

# Load audio
processor = AudioProcessor()
audio = processor.load("audio.mp3")

# Extract features
extractor = SignalFeatureExtractor()
features = extractor.extract(audio.waveform)

# Run detection
detector = DeepLearningDetector()
result = detector.detect(audio.waveform, audio.sample_rate)

print(f"Spoof probability: {result.spoof_probability:.2%}")
```

### Batch Processing Script

```python
from pathlib import Path
from terminal.cli import TerminalInterface

interface = TerminalInterface()

audio_dir = Path("audio_files")
for audio_file in audio_dir.glob("*.mp3"):
    result = interface.analyze_file(str(audio_file))
    print(f"{audio_file.name}: {result['risk_level']}")
```

