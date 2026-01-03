"""
Configuration and Constants for Fake Voice Detection System.

This module contains all configuration settings, constants, and
default parameters used throughout the application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================================
# Paths
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Model storage paths
MODELS_DIR = PROJECT_ROOT / "models"
RESNET18_WEIGHTS_PATH = MODELS_DIR / "resnet18" / "resnet18_spectrogram_weights.pth"
RESNET18_FULL_PATH = MODELS_DIR / "resnet18" / "resnet18_spectrogram_full.pth"
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
ANOMALY_MODELS_DIR = MODELS_DIR / "anomaly"
ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"


# ============================================================================
# Audio Processing
# ============================================================================

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    max_duration: float = 30.0  # seconds
    min_duration: float = 1.0   # seconds
    normalize: bool = True
    
    # FFT parameters
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80
    n_mfcc: int = 13
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        '.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'
    ])


AUDIO_CONFIG = AudioConfig()


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # AASIST
    aasist_hidden_dim: int = 64
    aasist_sinc_channels: int = 70
    aasist_sinc_kernel: int = 128
    
    # Whisper
    whisper_model_size: str = "large-v2"  # Only large-v2 for best accuracy
    whisper_language: str = "en"
    
    # Anomaly detection
    isolation_forest_estimators: int = 100
    isolation_forest_contamination: float = 0.1
    one_class_svm_nu: float = 0.1
    lof_neighbors: int = 20
    
    # Ensemble
    ensemble_method: str = "weighted"  # weighted, stacking, voting
    use_dynamic_weights: bool = True


MODEL_CONFIG = ModelConfig()


# ============================================================================
# Detection Thresholds
# ============================================================================

@dataclass
class ThresholdConfig:
    """Configuration for detection thresholds."""
    # Risk levels
    low_risk_threshold: float = 0.3
    medium_risk_threshold: float = 0.6
    high_risk_threshold: float = 0.6  # Above this = HIGH risk
    
    # Component weights
    deep_learning_weight: float = 0.35
    behavioral_weight: float = 0.20
    signal_weight: float = 0.15
    linguistic_weight: float = 0.15
    anomaly_weight: float = 0.15
    
    @property
    def weights(self) -> Dict[str, float]:
        """Get component weights as dictionary."""
        return {
            'deep_learning': self.deep_learning_weight,
            'behavioral': self.behavioral_weight,
            'signal': self.signal_weight,
            'linguistic': self.linguistic_weight,
            'anomaly': self.anomaly_weight,
        }


THRESHOLD_CONFIG = ThresholdConfig()


# ============================================================================
# Signal Features
# ============================================================================

@dataclass
class SignalFeatureConfig:
    """Configuration for signal-level feature extraction."""
    # MFCC
    n_mfcc: int = 13
    mfcc_delta_width: int = 9
    
    # Pitch
    pitch_fmin: float = 65.0   # Hz (C2)
    pitch_fmax: float = 2093.0 # Hz (C7)
    
    # Voice quality thresholds
    jitter_threshold: float = 0.05  # Above this = natural
    shimmer_threshold: float = 0.1  # Above this = natural
    hnr_threshold: float = 15.0     # Below this = natural
    
    # Energy
    energy_threshold: float = 0.01  # Silence threshold


SIGNAL_CONFIG = SignalFeatureConfig()


# ============================================================================
# Behavioral Analysis
# ============================================================================

@dataclass
class BehavioralConfig:
    """Configuration for behavioral analysis."""
    # Pause detection
    min_pause_duration: float = 0.1   # seconds
    max_pause_duration: float = 2.0   # seconds
    micro_pause_min: float = 0.02     # seconds
    micro_pause_max: float = 0.1      # seconds
    
    # Breathing detection
    breathing_fmin: float = 100.0     # Hz
    breathing_fmax: float = 500.0     # Hz
    expected_breathing_interval: float = 4.0  # seconds
    
    # Hesitation patterns
    hesitation_patterns: List[str] = field(default_factory=lambda: [
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean',
        'well', 'so', 'basically', 'actually', 'literally'
    ])


BEHAVIORAL_CONFIG = BehavioralConfig()


# ============================================================================
# Linguistic Analysis
# ============================================================================

@dataclass
class LinguisticConfig:
    """Configuration for linguistic analysis."""
    # Formality indicators
    formal_patterns: List[str] = field(default_factory=lambda: [
        'furthermore', 'moreover', 'nevertheless',
        'consequently', 'subsequently', 'therefore',
        'hence', 'thus', 'whereby'
    ])
    
    # Natural speech disfluencies
    expected_disfluency_rate: float = 0.02  # 2% of words
    
    # Sentence analysis
    min_sentence_length: int = 3   # words
    max_sentence_length: int = 50  # words


LINGUISTIC_CONFIG = LinguisticConfig()


# ============================================================================
# GUI Configuration
# ============================================================================

@dataclass
class GUIConfig:
    """Configuration for PyQt6 GUI."""
    window_title: str = "Fake Voice Detection System"
    min_width: int = 900
    min_height: int = 700
    
    # Colors (dark theme)
    background_color: str = "#1a1a2e"
    accent_color: str = "#3498db"
    success_color: str = "#2ecc71"
    warning_color: str = "#f39c12"
    error_color: str = "#e74c3c"
    
    # Risk level colors
    low_risk_color: str = "#2ecc71"
    medium_risk_color: str = "#f39c12"
    high_risk_color: str = "#e74c3c"


GUI_CONFIG = GUIConfig()


# ============================================================================
# Logging
# ============================================================================

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = None  # None = console only


LOGGING_CONFIG = LoggingConfig()


# ============================================================================
# Version Information
# ============================================================================

VERSION = "1.0.0"
AUTHOR = "Fake Voice Detection Team"
DESCRIPTION = "Multi-modal fake voice detection system"


# ============================================================================
# Environment Variables
# ============================================================================

def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    return {
        'debug': os.getenv('FVD_DEBUG', '0') == '1',
        'log_level': os.getenv('FVD_LOG_LEVEL', LOGGING_CONFIG.level),
        'models_dir': os.getenv('FVD_MODELS_DIR', str(MODELS_DIR)),
        'whisper_model': os.getenv('FVD_WHISPER_MODEL', MODEL_CONFIG.whisper_model_size),
    }


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        MODELS_DIR,
        MODELS_DIR / "resnet18",
        MODELS_DIR / "whisper",
        MODELS_DIR / "anomaly",
        MODELS_DIR / "ensemble",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config() -> Dict[str, Any]:
    """Get complete configuration as dictionary."""
    return {
        'version': VERSION,
        'audio': AUDIO_CONFIG.__dict__,
        'model': MODEL_CONFIG.__dict__,
        'threshold': THRESHOLD_CONFIG.__dict__,
        'signal': SIGNAL_CONFIG.__dict__,
        'behavioral': BEHAVIORAL_CONFIG.__dict__,
        'linguistic': LINGUISTIC_CONFIG.__dict__,
        'gui': GUI_CONFIG.__dict__,
        'logging': LOGGING_CONFIG.__dict__,
        'env': get_env_config(),
    }

