# Backend Configuration Guide

Configuration and optimization for different GPU backends.

## Supported Backends

| Backend | Hardware | Command |
|---------|----------|---------|
| MLX | Apple Silicon (M1/M2/M3) | `--MSilicon` |
| MPS | Apple Silicon (PyTorch) | `--MSilicon` |
| CUDA | NVIDIA GPUs | `--CUDA` |
| ROCm | AMD GPUs | `--ROCm` |
| CPU | Any | (default fallback) |

## Apple Silicon (MLX / MPS)

### MLX Backend

MLX is Apple's machine learning framework optimized for Apple Silicon.

**Setup:**
```bash
pip install mlx mlx-whisper
```

**Advantages:**
- Native Apple Silicon support
- Unified memory architecture
- Optimized for M-series chips

**Limitations:**
- macOS 13.3+ required
- Limited model support
- May require model conversion

### MPS Backend

Metal Performance Shaders via PyTorch.

**Setup:**
- Automatically available with PyTorch on Apple Silicon
- No additional installation needed

**Usage:**
```bash
python main.py --MSilicon
```

**Performance Tips:**
- Works well with most PyTorch models
- Some operations fall back to CPU
- Close other GPU-intensive apps

## NVIDIA CUDA

### Setup

1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Install PyTorch with CUDA support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Performance Tips

- Use `faster-whisper` for optimized Whisper inference
- Enable TF32 for faster FP32 operations
- Use mixed precision (FP16) when possible

```python
# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## AMD ROCm

### Setup

1. Install ROCm drivers
2. Install PyTorch with ROCm support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### Supported GPUs

- Radeon RX 6000 series
- Radeon RX 7000 series
- Radeon Pro series
- Instinct accelerators

### Verify Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # ROCm uses CUDA API
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Performance Tips

- Similar optimization to CUDA
- Check ROCm-specific documentation
- Some CUDA extensions may need adaptation

## CPU Fallback

The system automatically falls back to CPU when no GPU is available.

### Optimization

```python
# Use multiple threads
import torch
torch.set_num_threads(8)

# Use optimized BLAS
# Install MKL for Intel CPUs
```

### Performance Tips

- Use smaller model sizes (Whisper tiny/base)
- Process shorter audio segments
- Consider batch processing

## Backend Selection Logic

```python
# Priority order
1. CUDA (if --CUDA or available)
2. ROCm (if --ROCm or AMD GPU detected)
3. MLX (if --MSilicon and available)
4. MPS (if --MSilicon and available)
5. CPU (fallback)
```

## Benchmarks

Approximate processing times for 10-second audio:

| Backend | AASIST | Whisper (base) | Total |
|---------|--------|----------------|-------|
| CUDA (RTX 3080) | 0.2s | 1.5s | ~5s |
| MPS (M2 Pro) | 0.3s | 2.0s | ~7s |
| MLX (M2 Pro) | 0.2s | 1.8s | ~6s |
| ROCm (RX 6800) | 0.3s | 2.0s | ~7s |
| CPU (i7-12700) | 1.0s | 8.0s | ~15s |

*Times are approximate and vary by hardware*

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
# Use smaller Whisper model
export FVD_WHISPER_MODEL=tiny

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### MLX Model Conversion

Some PyTorch models need conversion for MLX:

```python
# Model conversion may be needed
# Check mlx documentation for conversion utilities
```

### MPS Fallback to CPU

Some operations aren't supported on MPS and fall back to CPU:
- Complex number operations
- Some convolution variants
- Certain loss functions

This is normal and handled automatically.

### ROCm Compatibility

Some CUDA-specific code may need adaptation:
- Use ROCm-compatible PyTorch
- Check operator support
- May need hip versions of some libraries

