"""
Device Manager for handling hardware detection and backend selection.

Supports:
- MLX (Apple Silicon)
- MPS (Metal Performance Shaders for PyTorch on Apple Silicon)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- CPU (fallback)
"""

import os
import sys
import logging
from enum import Enum
from typing import Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Enumeration of supported backend types."""
    MLX = "mlx"
    MPS = "mps"
    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """Information about the selected device."""
    backend: BackendType
    device_name: str
    device_index: int = 0
    memory_total: Optional[int] = None
    memory_available: Optional[int] = None
    compute_capability: Optional[str] = None


class DeviceManager:
    """
    Manages device detection and selection for the fake voice detection system.
    
    Provides unified interface for selecting and using different compute backends
    including MLX, MPS, CUDA, ROCm, and CPU.
    """
    
    def __init__(self, preferred_backend: Optional[str] = None):
        """
        Initialize the device manager.
        
        Args:
            preferred_backend: Preferred backend type ('MSilicon', 'CUDA', 'ROCm', or None for auto)
        """
        self.preferred_backend = preferred_backend
        self._device = None
        self._backend_type = None
        self._device_info = None
        self._torch_device = None
        self._mlx_available = False
        
        # Check available backends
        self._check_available_backends()
        
    def _check_available_backends(self) -> dict:
        """Check which backends are available on the system."""
        self.available_backends = {
            BackendType.MLX: self._check_mlx(),
            BackendType.MPS: self._check_mps(),
            BackendType.CUDA: self._check_cuda(),
            BackendType.ROCM: self._check_rocm(),
            BackendType.CPU: True,  # CPU is always available
        }
        
        logger.info(f"Available backends: {[k.value for k, v in self.available_backends.items() if v]}")
        return self.available_backends
    
    def _check_mlx(self) -> bool:
        """Check if MLX is available (Apple Silicon)."""
        try:
            import platform
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                return False
            import mlx.core as mx
            self._mlx_available = True
            logger.debug("MLX backend available")
            return True
        except ImportError:
            logger.debug("MLX not available (not installed)")
            return False
        except Exception as e:
            logger.debug(f"MLX not available: {e}")
            return False
    
    def _check_mps(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.debug("MPS backend available")
                return True
            return False
        except ImportError:
            logger.debug("PyTorch not installed, MPS unavailable")
            return False
        except Exception as e:
            logger.debug(f"MPS not available: {e}")
            return False
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                logger.debug(f"CUDA backend available: {device_count} device(s), {device_name}")
                return True
            return False
        except ImportError:
            logger.debug("PyTorch not installed, CUDA unavailable")
            return False
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")
            return False
    
    def _check_rocm(self) -> bool:
        """Check if ROCm is available (AMD GPUs)."""
        try:
            import torch
            # ROCm uses the same CUDA API in PyTorch
            if torch.cuda.is_available():
                # Check if this is actually ROCm by looking at device name
                device_name = torch.cuda.get_device_name(0)
                if "AMD" in device_name or "Radeon" in device_name:
                    logger.debug(f"ROCm backend available: {device_name}")
                    return True
            return False
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"ROCm not available: {e}")
            return False
    
    def select_backend(self, backend_arg: Optional[str] = None) -> BackendType:
        """
        Select the best available backend based on preference and availability.
        
        Args:
            backend_arg: Backend argument from CLI ('MSilicon', 'CUDA', 'ROCm')
            
        Returns:
            Selected BackendType
        """
        backend_arg = backend_arg or self.preferred_backend
        
        # Map CLI arguments to backend types
        if backend_arg == "MSilicon":
            # Prefer MLX, fallback to MPS
            if self.available_backends[BackendType.MLX]:
                self._backend_type = BackendType.MLX
            elif self.available_backends[BackendType.MPS]:
                self._backend_type = BackendType.MPS
            else:
                logger.warning("Apple Silicon backends not available, falling back to CPU")
                self._backend_type = BackendType.CPU
                
        elif backend_arg == "CUDA":
            if self.available_backends[BackendType.CUDA]:
                self._backend_type = BackendType.CUDA
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self._backend_type = BackendType.CPU
                
        elif backend_arg == "ROCm":
            if self.available_backends[BackendType.ROCM]:
                self._backend_type = BackendType.ROCM
            else:
                logger.warning("ROCm not available, falling back to CPU")
                self._backend_type = BackendType.CPU
                
        else:
            # Auto-select best available backend
            self._backend_type = self._auto_select_backend()
        
        logger.info(f"Selected backend: {self._backend_type.value}")
        self._initialize_device()
        return self._backend_type
    
    def _auto_select_backend(self) -> BackendType:
        """Auto-select the best available backend."""
        # Priority: CUDA > ROCm > MLX > MPS > CPU
        priority_order = [
            BackendType.CUDA,
            BackendType.ROCM,
            BackendType.MLX,
            BackendType.MPS,
            BackendType.CPU,
        ]
        
        for backend in priority_order:
            if self.available_backends[backend]:
                return backend
        
        return BackendType.CPU
    
    def _initialize_device(self) -> None:
        """Initialize the selected device."""
        if self._backend_type == BackendType.MLX:
            self._initialize_mlx()
        elif self._backend_type == BackendType.MPS:
            self._initialize_mps()
        elif self._backend_type in (BackendType.CUDA, BackendType.ROCM):
            self._initialize_cuda()
        else:
            self._initialize_cpu()
    
    def _initialize_mlx(self) -> None:
        """Initialize MLX backend."""
        import mlx.core as mx
        self._device = mx.default_device()
        self._device_info = DeviceInfo(
            backend=BackendType.MLX,
            device_name="Apple Silicon (MLX)",
        )
        logger.info("MLX backend initialized")
    
    def _initialize_mps(self) -> None:
        """Initialize MPS backend."""
        import torch
        self._torch_device = torch.device("mps")
        self._device = self._torch_device
        self._device_info = DeviceInfo(
            backend=BackendType.MPS,
            device_name="Apple Silicon (MPS)",
        )
        logger.info("MPS backend initialized")
    
    def _initialize_cuda(self) -> None:
        """Initialize CUDA/ROCm backend."""
        import torch
        self._torch_device = torch.device("cuda:0")
        self._device = self._torch_device
        
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_available = memory_total - torch.cuda.memory_allocated(0)
        
        self._device_info = DeviceInfo(
            backend=self._backend_type,
            device_name=device_name,
            memory_total=memory_total,
            memory_available=memory_available,
        )
        logger.info(f"CUDA/ROCm backend initialized: {device_name}")
    
    def _initialize_cpu(self) -> None:
        """Initialize CPU backend."""
        import torch
        self._torch_device = torch.device("cpu")
        self._device = self._torch_device
        self._device_info = DeviceInfo(
            backend=BackendType.CPU,
            device_name="CPU",
        )
        logger.info("CPU backend initialized")
    
    @property
    def device(self) -> Any:
        """Get the current device object."""
        if self._device is None:
            self.select_backend()
        return self._device
    
    @property
    def torch_device(self) -> Any:
        """Get the PyTorch device object."""
        if self._torch_device is None:
            if self._backend_type == BackendType.MLX:
                # For MLX, we still need PyTorch for some operations
                import torch
                self._torch_device = torch.device("cpu")
            else:
                self.select_backend()
        return self._torch_device
    
    @property
    def backend_type(self) -> BackendType:
        """Get the current backend type."""
        if self._backend_type is None:
            self.select_backend()
        return self._backend_type
    
    @property
    def device_info(self) -> DeviceInfo:
        """Get information about the current device."""
        if self._device_info is None:
            self.select_backend()
        return self._device_info
    
    def is_mlx(self) -> bool:
        """Check if current backend is MLX."""
        return self.backend_type == BackendType.MLX
    
    def is_mps(self) -> bool:
        """Check if current backend is MPS."""
        return self.backend_type == BackendType.MPS
    
    def is_cuda(self) -> bool:
        """Check if current backend is CUDA."""
        return self.backend_type == BackendType.CUDA
    
    def is_rocm(self) -> bool:
        """Check if current backend is ROCm."""
        return self.backend_type == BackendType.ROCM
    
    def is_cpu(self) -> bool:
        """Check if current backend is CPU."""
        return self.backend_type == BackendType.CPU
    
    def is_gpu(self) -> bool:
        """Check if current backend is a GPU (not CPU)."""
        return self.backend_type != BackendType.CPU
    
    def to_device(self, tensor: Any) -> Any:
        """
        Move a tensor to the current device.
        
        Args:
            tensor: PyTorch tensor or MLX array
            
        Returns:
            Tensor on the current device
        """
        if self.is_mlx():
            import mlx.core as mx
            if hasattr(tensor, 'numpy'):
                # Convert PyTorch tensor to MLX array
                return mx.array(tensor.numpy())
            return tensor
        else:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.to(self.torch_device)
            return tensor
    
    def get_memory_info(self) -> Tuple[int, int]:
        """
        Get memory information for the current device.
        
        Returns:
            Tuple of (total_memory, available_memory) in bytes
        """
        if self.is_cuda() or self.is_rocm():
            import torch
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            return total, total - allocated
        elif self.is_mlx():
            # MLX doesn't provide direct memory info
            return -1, -1
        else:
            # CPU - use system memory
            import psutil
            mem = psutil.virtual_memory()
            return mem.total, mem.available
    
    def synchronize(self) -> None:
        """Synchronize the current device (wait for all operations to complete)."""
        if self.is_cuda() or self.is_rocm():
            import torch
            torch.cuda.synchronize()
        elif self.is_mps():
            import torch
            torch.mps.synchronize()
        elif self.is_mlx():
            import mlx.core as mx
            mx.eval()


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager(preferred_backend: Optional[str] = None) -> DeviceManager:
    """
    Get the global device manager instance.
    
    Args:
        preferred_backend: Preferred backend type
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(preferred_backend)
    return _device_manager


def get_device() -> Any:
    """Get the current device object."""
    return get_device_manager().device


def get_backend_type() -> BackendType:
    """Get the current backend type."""
    return get_device_manager().backend_type


def initialize_backend(backend_arg: Optional[str] = None) -> BackendType:
    """
    Initialize the backend with the specified argument.
    
    Args:
        backend_arg: Backend argument ('MSilicon', 'CUDA', 'ROCm', or None for auto)
        
    Returns:
        Selected BackendType
    """
    global _device_manager
    _device_manager = DeviceManager()
    return _device_manager.select_backend(backend_arg)

