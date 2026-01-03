# Installation Guide

Complete installation instructions for the Fake Voice Detection System.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (optional, for cloning repository)

## Quick Installation

```bash
# Navigate to project directory
cd "Fake Voice Detection"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Platform-Specific Installation

### macOS (Apple Silicon)

For M1/M2/M3 Macs with Apple Silicon:

```bash
# Install base dependencies
pip install -r requirements.txt

# Install MLX for native Apple Silicon support (optional)
pip install mlx mlx-whisper

# PyTorch will automatically use MPS backend
```

### macOS (Intel)

```bash
pip install -r requirements.txt
# CPU-only mode will be used
```

### Linux (NVIDIA CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

For CUDA 12.x:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Linux (AMD ROCm)

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Install other dependencies
pip install -r requirements.txt
```

### Windows (NVIDIA CUDA)

```powershell
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Windows (CPU only)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Model Downloads

### AASIST Model Weights

The AASIST model weights need to be downloaded separately:

1. Download weights from the official repository or HuggingFace
2. Place in `models/aasist/weights.pth`

```bash
# Create directory
mkdir -p models/aasist

# Download weights (example - replace with actual URL)
# wget -O models/aasist/weights.pth <URL>
```

### Whisper Model

Whisper models are downloaded automatically on first use:

```python
# Models are cached in ~/.cache/whisper/
# Available sizes: tiny, base, small, medium, large
```

To pre-download:
```bash
python -c "import whisper; whisper.load_model('base')"
```

## Verifying Installation

Run the verification script:

```bash
python -c "
import torch
import librosa
import numpy as np
from PyQt6.QtWidgets import QApplication

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
print('librosa version:', librosa.__version__)
print('All dependencies installed successfully!')
"
```

## Troubleshooting

### PyQt6 Issues

**Error: "qt.qpa.plugin: Could not load the Qt platform plugin"**

On Linux:
```bash
sudo apt-get install libxcb-xinerama0
```

On macOS:
```bash
brew install qt
```

### Audio Loading Issues

**Error: "audioread.NoBackendError"**

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### CUDA Issues

**Error: "CUDA out of memory"**

- Use smaller batch sizes
- Use smaller Whisper model (tiny or base)
- Free GPU memory before running

**Error: "CUDA not available"**

1. Verify NVIDIA drivers are installed
2. Verify CUDA toolkit is installed
3. Reinstall PyTorch with CUDA support

### MLX Issues (Apple Silicon)

**Error: "No module named 'mlx'"**

```bash
pip install mlx mlx-whisper
```

**Error: "MLX requires macOS 13.3+"**

Update macOS to Ventura 13.3 or later.

## Environment Variables

Optional environment variables:

```bash
export FVD_DEBUG=1                    # Enable debug mode
export FVD_LOG_LEVEL=DEBUG           # Set log level
export FVD_MODELS_DIR=/path/to/models # Custom models directory
export FVD_WHISPER_MODEL=small        # Whisper model size
```

## Next Steps

After installation:

1. Run the application: `python main.py`
2. Read the [Usage Guide](USAGE.md)
3. Check [Examples](EXAMPLES.md)

