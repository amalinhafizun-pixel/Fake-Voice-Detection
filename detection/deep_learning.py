"""
Deep Learning-based Fake Voice Detection using AASIST.

AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
is a state-of-the-art model for audio spoof detection.

This module provides:
- AASIST model architecture
- Backend-agnostic inference (PyTorch/MLX)
- Pre-trained weight loading
- Spoof probability scoring
"""

import logging
import math
import os
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeepLearningResult:
    """Result from deep learning detection."""
    spoof_probability: float
    confidence: str
    raw_score: float
    model_name: str = "AASIST"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spoof_probability': self.spoof_probability,
            'confidence': self.confidence,
            'raw_score': self.raw_score,
            'model_name': self.model_name
        }


# PyTorch implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    TORCH_AVAILABLE = True
    
    class SincConv(nn.Module):
        """Sinc-based convolution layer for raw waveform processing."""
        
        def __init__(
            self,
            out_channels: int,
            kernel_size: int,
            sample_rate: int = 16000,
            min_low_hz: float = 50,
            min_band_hz: float = 50
        ):
            super().__init__()
            
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.sample_rate = sample_rate
            self.min_low_hz = min_low_hz
            self.min_band_hz = min_band_hz
            
            # Initialize filter parameters
            low_hz = min_low_hz + torch.abs(torch.rand(out_channels) * (sample_rate / 2 - min_low_hz - min_band_hz))
            band_hz = min_band_hz + torch.abs(torch.rand(out_channels) * (sample_rate / 2 - low_hz - min_band_hz))
            
            self.low_hz = nn.Parameter(low_hz)
            self.band_hz = nn.Parameter(band_hz)
            
            # Hamming window
            n = torch.linspace(0, kernel_size - 1, kernel_size)
            self.register_buffer('window', 0.54 - 0.46 * torch.cos(2 * math.pi * n / kernel_size))
            self.register_buffer('n', (kernel_size - 1) / 2 - n)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            low = self.min_low_hz + torch.abs(self.low_hz)
            high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz), max=self.sample_rate / 2)
            
            f_times_t_low = low.view(-1, 1) * self.n.view(1, -1) / self.sample_rate
            f_times_t_high = high.view(-1, 1) * self.n.view(1, -1) / self.sample_rate
            
            # Compute band-pass filters
            band_pass = 2 * ((torch.sin(2 * math.pi * f_times_t_high) - 
                             torch.sin(2 * math.pi * f_times_t_low)) / 
                            (self.n.view(1, -1) / self.sample_rate + 1e-8))
            band_pass = band_pass * self.window.view(1, -1)
            
            # Normalize
            band_pass = band_pass / (2 * band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)
            
            filters = band_pass.view(self.out_channels, 1, self.kernel_size)
            
            return F.conv1d(x, filters, padding=self.kernel_size // 2)
    
    
    class ResBlock(nn.Module):
        """Residual block for feature processing."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
            
            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1),
                    nn.BatchNorm1d(out_channels)
                )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.shortcut(x)
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            
            return out
    
    
    class GraphAttention(nn.Module):
        """Graph attention layer for spectro-temporal modeling."""
        
        def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
            super().__init__()
            
            self.W = nn.Linear(in_features, out_features, bias=False)
            self.a = nn.Linear(2 * out_features, 1, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.leakyrelu = nn.LeakyReLU(0.2)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, nodes, features)
            h = self.W(x)  # (batch, nodes, out_features)
            
            batch_size, num_nodes, _ = h.size()
            
            # Self-attention
            h_repeat = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)
            h_repeat_t = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)
            
            concat = torch.cat([h_repeat, h_repeat_t], dim=-1)
            e = self.leakyrelu(self.a(concat)).squeeze(-1)
            
            attention = F.softmax(e, dim=-1)
            attention = self.dropout(attention)
            
            h_prime = torch.bmm(attention, h)
            
            return F.elu(h_prime)
    
    
    class ResNetBlock1D(nn.Module):
        """1D ResNet block for audio processing."""
        
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.shortcut(x)
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            
            return out
    
    class ResNetAudioSpoof(nn.Module):
        """
        ResNet-based model for audio spoof detection.
        
        Uses ResNet architecture adapted for 1D audio signals with spectrogram features.
        """
        
        def __init__(
            self,
            num_classes: int = 2,
            base_channels: int = 64,
            num_layers: int = 18,  # ResNet-18 style
            sample_rate: int = 16000
        ):
            super().__init__()
            
            self.sample_rate = sample_rate
            
            # Initial convolution: convert raw audio to feature maps
            # Input: (batch, 1, samples) -> (batch, 64, samples/2)
            self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(base_channels)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            
            # ResNet layers
            self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
            self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
            self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
            self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
            
            # Global average pooling
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            
            # Classifier
            self.fc = nn.Linear(base_channels * 8, num_classes)
            
            # Initialize weights
            self._initialize_weights()
        
        def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
            layers = []
            layers.append(ResNetBlock1D(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResNetBlock1D(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)
        
        def _initialize_weights(self):
            """Initialize model weights."""
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Raw waveform (batch, samples) or (batch, 1, samples)
                
            Returns:
                Logits (batch, 2) - [genuine, spoof]
            """
            # Ensure input is (batch, 1, samples)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # Initial convolution
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            # ResNet layers
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            # Global pooling
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            
            # Classifier
            x = self.fc(x)
            
            return x
        
        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Get probability scores."""
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            return probs
    
    # Keep AASIST as alias for backward compatibility, but use ResNet
    AASIST = ResNetAudioSpoof

except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, deep learning detection will be limited")


class DeepLearningDetector:
    """
    Deep learning-based detector for fake voice detection.
    
    Uses ResNet-based model for audio spoof detection.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Any] = None,
        use_mlx: bool = False
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to model weights
            device: PyTorch device or None for auto-detection
            use_mlx: Whether to use MLX backend
        """
        self.model_path = model_path
        self.use_mlx = use_mlx
        self.model = None
        self.device = device
        
        if TORCH_AVAILABLE and not use_mlx:
            self._init_torch()
        elif use_mlx:
            self._init_mlx()
        else:
            logger.warning("No deep learning backend available")
    
    def _init_torch(self) -> None:
        """Initialize PyTorch ResNet18 model (trained on mel spectrograms)."""
        if self.device is None:
            # Auto-detect device with MPS priority on Apple Silicon
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("✅ Using MPS (Metal Performance Shaders) for ResNet18 acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("✅ Using CUDA for ResNet18 acceleration")
            else:
                self.device = torch.device("cpu")
                logger.info("⚠️  Using CPU for ResNet18 (no GPU acceleration)")
        
        # Use ResNet18 from torchvision (matches training architecture)
        from torchvision import models
        try:
            from torchvision.models.resnet import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            # Fallback for older torchvision
            self.model = models.resnet18(pretrained=True)
        
        # Modify for 1-channel input (mel spectrograms) and 2-class output
        self.model.conv1 = nn.Conv2d(
            1, 
            self.model.conv1.out_channels, 
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride, 
            padding=self.model.conv1.padding, 
            bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification
        
        logger.info("Using ResNet18 model (trained on mel spectrograms)")
        
        # Check for pre-trained weights
        weights_loaded = False
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                # Handle both full checkpoint and state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and any(k.startswith('conv1.') or k.startswith('fc.') for k in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel)
                if isinstance(state_dict, dict):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Try strict loading first
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=True)
                    if missing_keys:
                        logger.debug(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        logger.debug(f"Unexpected keys: {len(unexpected_keys)}")
                    logger.info(f"✅ Loaded ResNet18 weights from {self.model_path} (strict mode)")
                    weights_loaded = True
                except RuntimeError as e:
                    # Fall back to non-strict loading
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Some weights not loaded (missing keys: {len(missing_keys)})")
                    if unexpected_keys:
                        logger.debug(f"Unexpected keys: {len(unexpected_keys)}")
                    logger.info(f"✅ Loaded ResNet18 weights from {self.model_path} (non-strict mode)")
                    weights_loaded = True
                    
            except Exception as e:
                logger.error(f"Could not load weights: {e}. Using ImageNet pre-trained weights.")
                logger.debug(f"Error details: {type(e).__name__}: {e}")
        
        if not weights_loaded:
            logger.warning("⚠️  ResNet18 using ImageNet pre-trained weights (not fine-tuned for spoof detection)")
            logger.warning("⚠️  For best results, train the model using pretrain/train_model.py")
        
        # Move model to device and ensure it's on MPS if available
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Log device information
        if self.device.type == "mps":
            logger.info(f"✅ ResNet18 model loaded on MPS device")
        elif self.device.type == "cuda":
            logger.info(f"✅ ResNet18 model loaded on CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info(f"✅ ResNet18 model loaded on CPU")
    
    def _initialize_for_detection(self):
        """Initialize model with better weights for detection (when untrained)."""
        # Use Xavier initialization for better starting point
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Bias the classifier slightly towards detecting anomalies
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    # Slight bias towards "spoof" class to be more sensitive
                    nn.init.constant_(m.bias[0], 0.1)  # genuine
                    nn.init.constant_(m.bias[1], -0.1)  # spoof (slightly more sensitive)
    
    def _load_official_aasist(self):
        """Try to load the official AASIST model architecture."""
        try:
            from .aasist_official import Model as OfficialAASIST
            import json
            from pathlib import Path
            
            # Load config if available
            config_path = Path(__file__).parent.parent / "models" / "resnet18" / "config.conf"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_config = config.get('model_config', {})
            else:
                # Default AASIST config
                model_config = {
                    "nb_samp": 64600,
                    "first_conv": 128,
                    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                    "gat_dims": [64, 32],
                    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                    "temperatures": [2.0, 2.0, 100.0, 100.0]
                }
            
            model = OfficialAASIST(model_config)
            logger.info("Using official AASIST model architecture")
            return model
            
        except ImportError:
            logger.debug("Official AASIST model not available, using simplified version")
            return None
        except Exception as e:
            logger.warning(f"Failed to load official AASIST: {e}, using simplified version")
            return None
    
    def _init_mlx(self) -> None:
        """Initialize MLX model (placeholder)."""
        logger.warning("MLX AASIST not implemented, falling back to feature-based detection")
        self.model = None
    
    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> DeepLearningResult:
        """
        Detect if audio is spoofed using deep learning.
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate
            
        Returns:
            DeepLearningResult with spoof probability
        """
        if self.model is not None and TORCH_AVAILABLE:
            return self._detect_torch(waveform, sample_rate)
        else:
            return self._detect_fallback(waveform, sample_rate)
    
    def _detect_torch(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> DeepLearningResult:
        """Detection using PyTorch ResNet18 model (trained on mel spectrograms)."""
        import librosa
        
        # Simple music detection - warn if audio might be music
        # Music typically has more energy in mid-high frequencies and less variation
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(waveform))
            # Music typically has higher spectral centroid and lower ZCR variation
            if spectral_centroid > 3000 and zero_crossing_rate < 0.1:
                logger.warning("⚠️  Audio appears to be music, not speech. Model was trained on speech only.")
                logger.warning("⚠️  Results may be unreliable for music files.")
        except Exception:
            pass
        
        # Resample to 22050 Hz (training sample rate)
        if sample_rate != 22050:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=22050)
            sample_rate = 22050
        
        # Convert to mel spectrogram (matching training preprocessing)
        # Training used: n_mels=64, sample_time=3 seconds
        # Use same parameters as training: hop_length=512, n_fft=2048
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=sample_rate, 
            n_mels=64,
            hop_length=512,
            n_fft=2048
        )
        # Convert to dB scale (matching training)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize: take absolute value and divide by 80 (matching training)
        mel_spec = np.abs(mel_spec) / 80.0
        
        # Convert to tensor and add batch and channel dimensions
        # Shape: (batch=1, channels=1, n_mels=64, time_frames)
        x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)  # Move to MPS/CUDA/CPU device
        
        # Inference
        with torch.no_grad():
            # ResNet18 expects 2D input (spectrogram)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            genuine_prob = probs[0, 0].item()
            spoof_prob_raw = probs[0, 1].item()
            
            logger.info(f"ResNet18 raw output: genuine={genuine_prob:.3f}, spoof={spoof_prob_raw:.3f}")
            
            spoof_prob = spoof_prob_raw  # Index 1 is spoof (FAKE)
            raw_score = spoof_prob_raw - genuine_prob
            
            # Only adjust if model is clearly untrained (output near 0.5)
            # Trust high confidence outputs from trained model - don't reduce them
            if abs(spoof_prob - 0.5) < 0.1:
                logger.warning(f"ResNet18 output is near 0.5 ({spoof_prob:.3f}) - model may be untrained!")
                logger.info("Blending with signal-based detection for more reliable results")
                # Use signal features to adjust the score
                try:
                    from .signal_features import SignalFeatureExtractor
                    extractor = SignalFeatureExtractor(sample_rate=sample_rate)
                    features = extractor.extract(waveform, sample_rate)
                    signal_score = extractor.compute_anomaly_score(features)
                    # Blend ResNet output with signal score (weight signal more heavily)
                    spoof_prob = 0.3 * spoof_prob + 0.7 * signal_score
                    raw_score = spoof_prob * 2 - 1
                    logger.debug(f"Adjusted score using signal features: {spoof_prob:.3f}")
                except Exception as e:
                    logger.debug(f"Could not use signal features: {e}")
        
        # Determine confidence based on distance from 0.5
        # Trust the model's output - if it's confident (far from 0.5), use that
        distance_from_neutral = abs(spoof_prob - 0.5)
        
        # High confidence if model output is far from neutral (confident prediction)
        if distance_from_neutral > 0.4:
            confidence = "HIGH"
        elif distance_from_neutral > 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return DeepLearningResult(
            spoof_probability=spoof_prob,
            confidence=confidence,
            raw_score=raw_score,
            model_name="ResNet"
        )
    
    def _detect_fallback(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> DeepLearningResult:
        """
        Fallback detection using signal-level features.
        
        Used when deep learning model is not available.
        """
        try:
            from .signal_features import SignalFeatureExtractor
            
            extractor = SignalFeatureExtractor(sample_rate=sample_rate)
            features = extractor.extract(waveform, sample_rate)
            
            # Use feature-based scoring
            score = extractor.compute_anomaly_score(features)
            
            if score > 0.6:
                confidence = "HIGH"
            elif score > 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            return DeepLearningResult(
                spoof_probability=score,
                confidence=confidence,
                raw_score=score * 2 - 1,  # Map to [-1, 1]
                model_name="FeatureBased"
            )
            
        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return DeepLearningResult(
                spoof_probability=0.5,
                confidence="LOW",
                raw_score=0.0,
                model_name="Unknown"
            )
    
    def batch_detect(
        self,
        waveforms: list,
        sample_rate: int = 16000
    ) -> list:
        """
        Batch detection for multiple audio samples.
        
        Args:
            waveforms: List of audio waveforms
            sample_rate: Sample rate
            
        Returns:
            List of DeepLearningResult objects
        """
        results = []
        for waveform in waveforms:
            result = self.detect(waveform, sample_rate)
            results.append(result)
        return results

