"""
Anomaly Detection for Fake Voice Detection.

Implements multiple anomaly detection algorithms:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)

These algorithms detect outliers in feature space that may indicate
AI-generated or spoofed audio.
"""

import logging
from typing import Dict, Optional, Any, Tuple, List, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    anomaly_score: float
    is_anomaly: bool
    confidence: str
    method_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_score': self.anomaly_score,
            'is_anomaly': self.is_anomaly,
            'confidence': self.confidence,
            'method_scores': self.method_scores
        }


class IsolationForestDetector:
    """Isolation Forest-based anomaly detection."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of trees
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        self._fitted = False
    
    def fit(self, features: np.ndarray) -> 'IsolationForestDetector':
        """Fit the model on normal data."""
        from sklearn.ensemble import IsolationForest
        
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self._model.fit(features)
        self._fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores.
        
        Returns:
            Tuple of (anomaly_scores, predictions)
            anomaly_scores: Higher = more anomalous
            predictions: 1 = normal, -1 = anomaly
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = self._model.predict(features)
        # score_samples returns negative scores where lower is more anomalous
        raw_scores = self._model.score_samples(features)
        
        # Convert to 0-1 anomaly score (higher = more anomalous)
        anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        
        return anomaly_scores, predictions


class OneClassSVMDetector:
    """One-Class SVM-based anomaly detection."""
    
    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: str = "scale"
    ):
        """
        Initialize One-Class SVM detector.
        
        Args:
            kernel: Kernel type ('rbf', 'poly', 'sigmoid', 'linear')
            nu: Upper bound on fraction of outliers
            gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self._model = None
        self._fitted = False
    
    def fit(self, features: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the model on normal data."""
        from sklearn.svm import OneClassSVM
        
        self._model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )
        self._model.fit(features)
        self._fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = self._model.predict(features)
        # decision_function: positive = normal, negative = anomaly
        raw_scores = self._model.decision_function(features)
        
        # Convert to 0-1 anomaly score
        anomaly_scores = 1 / (1 + np.exp(raw_scores))  # Sigmoid transform
        
        return anomaly_scores, predictions


class LocalOutlierFactorDetector:
    """Local Outlier Factor-based anomaly detection."""
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        novelty: bool = True
    ):
        """
        Initialize LOF detector.
        
        Args:
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of anomalies
            novelty: If True, use novelty detection mode
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        self._model = None
        self._fitted = False
    
    def fit(self, features: np.ndarray) -> 'LocalOutlierFactorDetector':
        """Fit the model on normal data."""
        from sklearn.neighbors import LocalOutlierFactor
        
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=-1
        )
        self._model.fit(features)
        self._fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = self._model.predict(features)
        raw_scores = self._model.decision_function(features)
        
        # Convert to 0-1 anomaly score
        anomaly_scores = 1 / (1 + np.exp(raw_scores))
        
        return anomaly_scores, predictions


class AnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods.
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        weights: Dict[str, float] = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            methods: List of methods to use ('isolation_forest', 'one_class_svm', 'lof')
            weights: Weights for each method (default: equal weights)
        """
        self.methods = methods or ['isolation_forest', 'one_class_svm', 'lof']
        self.weights = weights or {m: 1.0 / len(self.methods) for m in self.methods}
        
        self._detectors: Dict[str, Any] = {}
        self._fitted = False
    
    def fit(self, features: np.ndarray) -> 'AnomalyDetector':
        """
        Fit all anomaly detectors on normal data.
        
        Args:
            features: Training features (assumed to be mostly normal)
            
        Returns:
            Self for chaining
        """
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        for method in self.methods:
            if method == 'isolation_forest':
                self._detectors[method] = IsolationForestDetector()
            elif method == 'one_class_svm':
                self._detectors[method] = OneClassSVMDetector()
            elif method == 'lof':
                self._detectors[method] = LocalOutlierFactorDetector()
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            try:
                self._detectors[method].fit(features)
                logger.debug(f"Fitted {method} detector")
            except Exception as e:
                logger.error(f"Failed to fit {method}: {e}")
        
        self._fitted = True
        return self
    
    def predict(self, features: np.ndarray, signal_features: Optional[Any] = None) -> AnomalyResult:
        """
        Predict anomaly score using ensemble of methods.
        
        Args:
            features: Feature vector to analyze
            signal_features: Optional SignalFeatures object for better heuristic
            
        Returns:
            AnomalyResult with combined score
        """
        if not self._fitted:
            # If signal_features object is available, use it for better scoring
            if signal_features is not None:
                try:
                    # Use signal features directly for more accurate heuristic
                    # Robotic voices have specific characteristics
                    scores = []
                    
                    # Low jitter = synthetic (too perfect)
                    if signal_features.jitter < 0.003:
                        scores.append(0.8)
                    elif signal_features.jitter < 0.005:
                        scores.append(0.5)
                    else:
                        scores.append(0.2)
                    
                    # Low shimmer = synthetic
                    if signal_features.shimmer < 0.01:
                        scores.append(0.8)
                    elif signal_features.shimmer < 0.02:
                        scores.append(0.5)
                    else:
                        scores.append(0.2)
                    
                    # High HNR = synthetic (too clean)
                    if signal_features.hnr > 25:
                        scores.append(0.9)
                    elif signal_features.hnr > 20:
                        scores.append(0.6)
                    else:
                        scores.append(0.3)
                    
                    # High phase coherence = synthetic (too regular)
                    if signal_features.phase_coherence > 0.9:
                        scores.append(0.7)
                    elif signal_features.phase_coherence > 0.7:
                        scores.append(0.4)
                    else:
                        scores.append(0.2)
                    
                    # Low spectral flux variation = synthetic (too uniform)
                    flux_std = float(np.std(signal_features.spectral_flux))
                    if flux_std < 0.1:
                        scores.append(0.7)
                    elif flux_std < 0.2:
                        scores.append(0.4)
                    else:
                        scores.append(0.2)
                    
                    # Weighted average
                    heuristic_score = np.mean(scores)
                    heuristic_score = max(0.0, min(1.0, heuristic_score))
                    
                    logger.debug(f"Anomaly heuristic from signal features: {heuristic_score:.3f}")
                    
                    return AnomalyResult(
                        anomaly_score=heuristic_score,
                        is_anomaly=heuristic_score > 0.5,
                        confidence="MEDIUM" if heuristic_score > 0.65 or heuristic_score < 0.35 else "LOW",
                        method_scores={'signal_heuristic': heuristic_score}
                    )
                except Exception as e:
                    logger.debug(f"Could not use signal_features object: {e}, falling back to vector heuristic")
            
            # Fallback to vector-based heuristic
            # If not fitted, use improved signal-based heuristic
            # Use multiple indicators of synthetic/robotic voices
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            features_flat = features.flatten()
            
            # 1. Feature variance - low variance indicates robotic/synthetic (too uniform)
            feature_variance = float(np.var(features_flat))
            feature_std = float(np.std(features_flat))
            feature_mean = float(np.mean(np.abs(features_flat)))
            
            # Normalize variance score (more lenient thresholds)
            # Typical variance range: 0.001-1.0 for normalized features
            if feature_variance < 0.001:
                variance_score = 0.9  # Very low variance = very synthetic
            elif feature_variance < 0.01:
                variance_score = 0.7  # Low variance = likely synthetic
            elif feature_variance < 0.1:
                variance_score = 0.4  # Medium variance = uncertain
            else:
                variance_score = 0.2  # High variance = likely natural
            
            # 2. Coefficient of variation (regularity)
            if feature_mean > 1e-6:
                cv = feature_std / feature_mean
                if cv < 0.1:
                    regularity_score = 0.8  # Very regular = synthetic
                elif cv < 0.3:
                    regularity_score = 0.5  # Somewhat regular
                else:
                    regularity_score = 0.2  # Irregular = natural
            else:
                regularity_score = 0.5
            
            # 3. Feature range - synthetic voices often have compressed dynamic range
            feature_range = float(np.max(features_flat) - np.min(features_flat))
            if feature_range < 0.1:
                range_score = 0.7  # Very compressed = synthetic
            elif feature_range < 0.5:
                range_score = 0.4
            else:
                range_score = 0.2  # Wide range = natural
            
            # 4. Feature distribution - check for unusual patterns
            # Synthetic voices may have more uniform distribution
            percentiles = np.percentile(features_flat, [25, 50, 75])
            iqr = percentiles[2] - percentiles[1]
            if iqr < 0.05:
                distribution_score = 0.6  # Very tight distribution = synthetic
            else:
                distribution_score = 0.3
            
            # Combine all heuristics with weights
            heuristic_score = (
                variance_score * 0.35 +
                regularity_score * 0.25 +
                range_score * 0.20 +
                distribution_score * 0.20
            )
            
            # Ensure score is in valid range
            heuristic_score = max(0.0, min(1.0, heuristic_score))
            
            logger.debug(f"Anomaly heuristic: variance={variance_score:.3f}, "
                        f"regularity={regularity_score:.3f}, range={range_score:.3f}, "
                        f"distribution={distribution_score:.3f}, final={heuristic_score:.3f}")
            
            return AnomalyResult(
                anomaly_score=heuristic_score,
                is_anomaly=heuristic_score > 0.5,
                confidence="MEDIUM" if heuristic_score > 0.65 or heuristic_score < 0.35 else "LOW",
                method_scores={'heuristic': heuristic_score}
            )
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        method_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, detector in self._detectors.items():
            try:
                scores, predictions = detector.predict(features)
                score = float(scores[0])
                method_scores[method] = score
                weighted_sum += score * self.weights.get(method, 1.0)
                total_weight += self.weights.get(method, 1.0)
            except Exception as e:
                logger.warning(f"{method} prediction failed: {e}")
        
        if total_weight == 0:
            combined_score = 0.5
        else:
            combined_score = weighted_sum / total_weight
        
        # Determine if anomaly
        is_anomaly = combined_score > 0.5
        
        # Determine confidence
        if combined_score > 0.8 or combined_score < 0.2:
            confidence = "HIGH"
        elif combined_score > 0.6 or combined_score < 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return AnomalyResult(
            anomaly_score=combined_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            method_scores=method_scores
        )
    
    def save(self, path: str) -> None:
        """Save fitted models to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'methods': self.methods,
                'weights': self.weights,
                'detectors': self._detectors,
                'fitted': self._fitted
            }, f)
        logger.info(f"Anomaly detector saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load fitted models from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(methods=data['methods'], weights=data['weights'])
        detector._detectors = data['detectors']
        detector._fitted = data['fitted']
        logger.info(f"Anomaly detector loaded from {path}")
        return detector


def detect_anomalies(
    features: np.ndarray,
    training_features: Optional[np.ndarray] = None,
    methods: List[str] = None
) -> AnomalyResult:
    """
    Convenience function for anomaly detection.
    
    Args:
        features: Features to analyze
        training_features: Training data for fitting (optional)
        methods: Detection methods to use
        
    Returns:
        AnomalyResult with detection results
    """
    detector = AnomalyDetector(methods=methods)
    
    if training_features is not None:
        detector.fit(training_features)
    else:
        # Use input features as pseudo-training (not ideal but works for single-sample)
        detector.fit(features.reshape(1, -1) if features.ndim == 1 else features)
    
    return detector.predict(features)

