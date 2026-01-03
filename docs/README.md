# Fake Voice Detection System

A comprehensive multi-modal system for detecting AI-generated, cloned, and synthetic speech in audio files.

## Overview

This application uses multiple detection methods to identify fake voices:

- **Deep Learning (AASIST)**: State-of-the-art neural network for spoof detection
- **Signal-Level Analysis**: MFCC, spectral flux, pitch jitter, HNR, phase coherence
- **Behavioral Analysis**: Temporal patterns, pauses, breathing, cadence
- **Linguistic Analysis**: Whisper-based transcription and NLP analysis
- **Anomaly Detection**: Isolation Forest, One-Class SVM, LOF

## Features

- Support for multiple audio formats (MP3, WAV, AAC, FLAC, M4A, OGG)
- PyQt6 graphical user interface with drag-and-drop
- Terminal/CLI interface for scripting
- Multi-backend support:
  - Apple Silicon (MLX/MPS)
  - NVIDIA CUDA
  - AMD ROCm
  - CPU fallback
- Ensemble scoring with calibrated probabilities
- Export results to JSON/CSV

## Quick Start

### Installation

```bash
# Clone or download the repository
cd "Fake Voice Detection"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**GUI Mode (default):**
```bash
python main.py --PyQT
```

**Terminal Mode:**
```bash
python main.py --terminal --file audio.mp3
```

**With Specific Backend:**
```bash
python main.py --MSilicon --PyQT     # Apple Silicon
python main.py --CUDA --terminal      # NVIDIA GPU
python main.py --ROCm --terminal      # AMD GPU
```

## Documentation

- [Architecture](ARCHITECTURE.md) - System design and data flow
- [Installation](INSTALLATION.md) - Detailed setup instructions
- [Usage Guide](USAGE.md) - How to use the application
- [API Reference](API_REFERENCE.md) - Code documentation
- [Detection Methods](DETECTION_METHODS.md) - Algorithm explanations
- [Backend Guide](BACKEND_GUIDE.md) - GPU backend configuration
- [Examples](EXAMPLES.md) - Usage examples and samples

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | .wav | Best quality, recommended |
| MP3 | .mp3 | Common format |
| AAC | .aac, .m4a | Apple format |
| FLAC | .flac | Lossless compression |
| OGG | .ogg | Open format |

## Risk Levels

| Score | Risk Level | Interpretation |
|-------|------------|----------------|
| 0.0 - 0.3 | LOW | Likely genuine human speech |
| 0.3 - 0.6 | MEDIUM | Some synthetic indicators |
| 0.6 - 1.0 | HIGH | Likely AI-generated |

## Requirements

- Python 3.9+
- PyQt6 (for GUI)
- PyTorch 2.0+
- librosa, numpy, scipy
- openai-whisper
- scikit-learn

See [requirements.txt](../requirements.txt) for full list.

## Project Structure

```
fake_voice_detection/
├── main.py              # Entry point
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── backend/             # Device management
├── detection/           # Detection algorithms
├── audio/               # Audio processing
├── ui/                  # PyQt6 GUI
├── terminal/            # CLI interface
├── ml/                  # ML utilities
├── models/              # Model weights
└── docs/                # Documentation
```

## Contributing

Contributions are welcome! Please read the documentation and follow the code style of the project.

## License

This project is for educational and research purposes.

## Acknowledgments

- AASIST model architecture from ASVspoof challenge
- OpenAI Whisper for speech recognition
- scikit-learn for ML algorithms

