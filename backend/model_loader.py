"""
Model Loader for loading and managing ML models across different backends.

Handles loading models for:
- AASIST (PyTorch / MLX)
- Whisper (openai-whisper / faster-whisper / mlx-whisper)
- Anomaly detection models (scikit-learn)
- Ensemble meta-models (XGBoost / LightGBM)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict, Union
from dataclasses import dataclass
import pickle

from .device_manager import DeviceManager, BackendType, get_device_manager

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    path: str
    backend: BackendType
    loaded: bool = False
    model_type: str = "unknown"
    version: Optional[str] = None


class ModelLoader:
    """
    Unified model loader for all ML models in the fake voice detection system.
    
    Handles backend-specific loading and caching of models.
    """
    
    def __init__(self, models_dir: Optional[str] = None, device_manager: Optional[DeviceManager] = None):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory containing model weights
            device_manager: DeviceManager instance for backend selection
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / "models"
        self.device_manager = device_manager or get_device_manager()
        self._model_cache: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        
    def load_resnet18(self, weights_path: Optional[str] = None) -> Any:
        """
        Load the ResNet18 model for spoof detection.
        
        Args:
            weights_path: Path to model weights (optional, uses default if not provided)
            
        Returns:
            Loaded ResNet18 model (DeepLearningDetector instance)
        """
        cache_key = "resnet18"
        if cache_key in self._model_cache:
            logger.debug("Using cached ResNet18 model")
            return self._model_cache[cache_key]
        
        # Try full model path first, then weights path
        full_path = weights_path or str(self.models_dir / "resnet18" / "resnet18_spectrogram_full.pth")
        weights_path = weights_path or str(self.models_dir / "resnet18" / "resnet18_spectrogram_weights.pth")
        
        # Prefer full model if it exists
        if os.path.exists(full_path):
            weights_path = full_path
        
        if self.device_manager.is_mlx():
            model = self._load_resnet18_mlx(weights_path)
        else:
            model = self._load_resnet18_torch(weights_path)
        
        if model is not None:
            self._model_cache[cache_key] = model
            self._model_info[cache_key] = ModelInfo(
                name="ResNet18",
                path=weights_path,
                backend=self.device_manager.backend_type,
                loaded=True,
                model_type="spoof_detection"
            )
        
        return model
    
    def _load_resnet18_torch(self, weights_path: str) -> Any:
        """Load ResNet18 model using PyTorch."""
        try:
            from detection.deep_learning import DeepLearningDetector
            
            # Get the device from device_manager (will be MPS if available on Apple Silicon)
            device = self.device_manager.torch_device
            
            # Log device selection
            if self.device_manager.is_mps():
                logger.info("🚀 Loading ResNet18 with MPS acceleration")
            elif self.device_manager.is_cuda():
                logger.info("🚀 Loading ResNet18 with CUDA acceleration")
            else:
                logger.info("Loading ResNet18 on CPU")
            
            # Create detector with model path and device
            model = DeepLearningDetector(
                model_path=weights_path,
                device=device,  # Use device_manager's device (MPS/CUDA/CPU)
                use_mlx=False
            )
            
            if os.path.exists(weights_path):
                logger.info(f"✅ ResNet18 model loaded from {weights_path}")
            else:
                logger.warning(f"⚠️  ResNet18 weights not found at {weights_path}, using ImageNet pre-trained weights")
            
            return model
            
        except ImportError as e:
            logger.error(f"Failed to import ResNet18 model: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load ResNet18 model: {e}")
            return None
    
    def _load_resnet18_mlx(self, weights_path: str) -> Any:
        """Load ResNet18 model using MLX."""
        try:
            # For now, fall back to PyTorch
            logger.warning("MLX ResNet18 not yet implemented, using PyTorch")
            return self._load_resnet18_torch(weights_path)
            
        except ImportError as e:
            logger.error(f"Failed to import MLX: {e}")
            return self._load_resnet18_torch(weights_path)
    
    # Keep load_aasist as alias for backward compatibility
    def load_aasist(self, weights_path: Optional[str] = None) -> Any:
        """Alias for load_resnet18 for backward compatibility."""
        return self.load_resnet18(weights_path)
    
    def load_whisper(self, model_size: str = "large-v2") -> Any:
        """
        Load the Whisper model for speech recognition.
        
        Args:
            model_size: Model size (default: 'large-v2' for best accuracy)
            
        Returns:
            Loaded Whisper model
        """
        cache_key = f"whisper_{model_size}"
        if cache_key in self._model_cache:
            logger.debug(f"Using cached Whisper model ({model_size})")
            return self._model_cache[cache_key]
        
        if self.device_manager.is_mlx():
            model = self._load_whisper_mlx(model_size)
        elif self.device_manager.is_cuda() or self.device_manager.is_rocm():
            model = self._load_whisper_faster(model_size)
        else:
            model = self._load_whisper_openai(model_size)
        
        if model is not None:
            self._model_cache[cache_key] = model
            self._model_info[cache_key] = ModelInfo(
                name=f"Whisper-{model_size}",
                path=f"~/.cache/whisper/{model_size}",
                backend=self.device_manager.backend_type,
                loaded=True,
                model_type="speech_recognition"
            )
        
        return model
    
    def _load_whisper_openai(self, model_size: str) -> Any:
        """Load Whisper using OpenAI's implementation."""
        try:
            import whisper
            import torch
            
            # Auto-detect device: CUDA > CPU
            # Note: Whisper has limited MPS support (sparse tensor issues), so use CPU on MPS
            device = "cpu"
            if self.device_manager.is_cuda() or self.device_manager.is_rocm():
                device = "cuda"
            # Skip MPS for Whisper due to compatibility issues
            
            model = whisper.load_model(model_size, device=device)
            logger.info(f"Whisper model loaded ({model_size}) on {device}")
            return model
            
        except ImportError:
            logger.error("openai-whisper not installed. Install with: pip install openai-whisper")
            return None
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return None
    
    def _load_whisper_faster(self, model_size: str) -> Any:
        """Load Whisper using faster-whisper for CUDA optimization."""
        try:
            from faster_whisper import WhisperModel
            
            device = "cuda" if self.device_manager.is_cuda() or self.device_manager.is_rocm() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info(f"Faster-Whisper model loaded ({model_size}) on {device}")
            return model
            
        except ImportError:
            logger.info("faster-whisper not installed, falling back to openai-whisper")
            return self._load_whisper_openai(model_size)
        except Exception as e:
            logger.warning(f"Failed to load faster-whisper, falling back: {e}")
            return self._load_whisper_openai(model_size)
    
    def _load_whisper_mlx(self, model_size: str) -> Any:
        """Load Whisper using MLX implementation."""
        try:
            import mlx_whisper
            
            model = mlx_whisper.load_model(model_size)
            logger.info(f"MLX Whisper model loaded ({model_size})")
            return model
            
        except ImportError:
            logger.info("mlx-whisper not installed, falling back to openai-whisper")
            return self._load_whisper_openai(model_size)
        except Exception as e:
            logger.warning(f"Failed to load MLX Whisper, falling back: {e}")
            return self._load_whisper_openai(model_size)
    
    def load_anomaly_model(self, model_name: str) -> Any:
        """
        Load a trained anomaly detection model.
        
        Args:
            model_name: Name of the model ('isolation_forest', 'one_class_svm', 'lof')
            
        Returns:
            Loaded scikit-learn model
        """
        cache_key = f"anomaly_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_path = self.models_dir / "anomaly" / f"{model_name}.pkl"
        
        if not model_path.exists():
            logger.warning(f"Anomaly model not found at {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self._model_cache[cache_key] = model
            self._model_info[cache_key] = ModelInfo(
                name=model_name,
                path=str(model_path),
                backend=BackendType.CPU,
                loaded=True,
                model_type="anomaly_detection"
            )
            logger.info(f"Anomaly model loaded: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
            return None
    
    def load_ensemble_model(self, model_name: str = "stacking") -> Any:
        """
        Load a trained ensemble meta-model.
        
        Args:
            model_name: Name of the ensemble model ('stacking', 'boosting')
            
        Returns:
            Loaded ensemble model
        """
        cache_key = f"ensemble_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_path = self.models_dir / "ensemble" / f"{model_name}.pkl"
        
        if not model_path.exists():
            logger.warning(f"Ensemble model not found at {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self._model_cache[cache_key] = model
            self._model_info[cache_key] = ModelInfo(
                name=model_name,
                path=str(model_path),
                backend=BackendType.CPU,
                loaded=True,
                model_type="ensemble"
            )
            logger.info(f"Ensemble model loaded: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ensemble model: {e}")
            return None
    
    def save_model(self, model: Any, model_name: str, model_type: str = "custom") -> bool:
        """
        Save a trained model to disk.
        
        Args:
            model: Model to save
            model_name: Name for the model file
            model_type: Type of model ('anomaly', 'ensemble', 'custom')
            
        Returns:
            True if save was successful
        """
        try:
            if model_type == "anomaly":
                save_dir = self.models_dir / "anomaly"
            elif model_type == "ensemble":
                save_dir = self.models_dir / "ensemble"
            else:
                save_dir = self.models_dir
            
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{model_name}.pkl"
            
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        return self._model_info.get(model_name)
    
    def list_loaded_models(self) -> Dict[str, ModelInfo]:
        """List all currently loaded models."""
        return self._model_info.copy()
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            self._model_cache.pop(model_name, None)
            self._model_info.pop(model_name, None)
        else:
            self._model_cache.clear()
            self._model_info.clear()
        logger.debug(f"Model cache cleared: {model_name or 'all'}")

