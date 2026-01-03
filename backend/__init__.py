"""
Backend module for device management and model loading.

This module handles hardware detection, backend selection (MLX/MPS/CUDA/ROCm),
and provides utilities for loading models on the appropriate device.
"""

from .device_manager import DeviceManager, get_device, get_backend_type
from .model_loader import ModelLoader

__all__ = [
    'DeviceManager',
    'get_device',
    'get_backend_type',
    'ModelLoader',
]

