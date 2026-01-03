"""
Detection module for fake voice analysis.

This module contains all detection methods including:
- Signal-level feature extraction (MFCC, spectral flux, pitch, HNR, phase)
- Deep learning-based detection (AASIST)
- Behavioral analysis (temporal patterns)
- Linguistic analysis (Whisper-based)
- Anomaly detection (Isolation Forest, One-Class SVM, LOF)
- Feature engineering (normalization, PCA, selection)
- Ensemble methods (stacking, boosting, voting)
- Model calibration (Platt scaling, isotonic regression)
"""

from .signal_features import SignalFeatureExtractor
from .deep_learning import DeepLearningDetector
from .behavioral import BehavioralAnalyzer
from .linguistic import LinguisticAnalyzer
from .anomaly_detection import AnomalyDetector
from .feature_engineering import FeatureEngineer
from .ensemble import EnsembleDetector
from .calibration import ModelCalibrator

__all__ = [
    'SignalFeatureExtractor',
    'DeepLearningDetector',
    'BehavioralAnalyzer',
    'LinguisticAnalyzer',
    'AnomalyDetector',
    'FeatureEngineer',
    'EnsembleDetector',
    'ModelCalibrator',
]

