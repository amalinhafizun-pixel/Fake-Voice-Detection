"""
Model Calibration for Fake Voice Detection.

Implements probability calibration methods:
- Platt Scaling (Sigmoid calibration)
- Isotonic Regression
- Temperature Scaling

Calibration ensures that predicted probabilities accurately reflect
true frequencies of outcomes.
"""

import logging
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of probability calibration."""
    original_prob: float
    calibrated_prob: float
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None


class PlattScaler:
    """
    Platt scaling (sigmoid calibration) for probability calibration.
    
    Fits a sigmoid function to map raw scores to calibrated probabilities.
    """
    
    def __init__(self):
        self._a = 0.0
        self._b = 0.0
        self._fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'PlattScaler':
        """
        Fit Platt scaler on validation data.
        
        Args:
            scores: Raw model scores/probabilities
            labels: True binary labels (0 or 1)
            
        Returns:
            Self for chaining
        """
        from sklearn.linear_model import LogisticRegression
        
        # Reshape for sklearn
        scores = scores.reshape(-1, 1)
        
        # Fit logistic regression
        lr = LogisticRegression(random_state=42)
        lr.fit(scores, labels)
        
        self._a = lr.coef_[0][0]
        self._b = lr.intercept_[0]
        self._fitted = True
        
        logger.debug(f"Platt scaler fitted: a={self._a:.4f}, b={self._b:.4f}")
        return self
    
    def calibrate(self, score: float) -> float:
        """
        Calibrate a single probability score.
        
        Args:
            score: Raw probability (0-1)
            
        Returns:
            Calibrated probability
        """
        if not self._fitted:
            return score
        
        # Apply sigmoid transformation
        calibrated = 1.0 / (1.0 + np.exp(-(self._a * score + self._b)))
        return float(calibrated)
    
    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate multiple scores."""
        if not self._fitted:
            return scores
        return 1.0 / (1.0 + np.exp(-(self._a * scores + self._b)))


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    
    Non-parametric calibration that preserves ordering while
    mapping to calibrated probabilities.
    """
    
    def __init__(self):
        self._model = None
        self._fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression on validation data.
        
        Args:
            scores: Raw model scores/probabilities
            labels: True binary labels
            
        Returns:
            Self for chaining
        """
        from sklearn.isotonic import IsotonicRegression
        
        self._model = IsotonicRegression(out_of_bounds='clip')
        self._model.fit(scores, labels)
        self._fitted = True
        
        logger.debug("Isotonic calibrator fitted")
        return self
    
    def calibrate(self, score: float) -> float:
        """Calibrate a single probability score."""
        if not self._fitted:
            return score
        
        calibrated = self._model.predict([score])[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate multiple scores."""
        if not self._fitted:
            return scores
        return np.clip(self._model.predict(scores), 0.0, 1.0)


class TemperatureScaler:
    """
    Temperature scaling for neural network calibration.
    
    Learns a single temperature parameter to scale logits.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        self._temperature = initial_temperature
        self._fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> 'TemperatureScaler':
        """
        Fit temperature parameter using gradient descent on NLL.
        
        Args:
            logits: Raw model logits (before softmax)
            labels: True binary labels
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Self for chaining
        """
        temperature = self._temperature
        
        for _ in range(max_iter):
            # Forward pass
            scaled_logits = logits / temperature
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            
            # Compute NLL loss
            eps = 1e-10
            nll = -np.mean(
                labels * np.log(probs + eps) + 
                (1 - labels) * np.log(1 - probs + eps)
            )
            
            # Compute gradient
            gradient = np.mean(
                (probs - labels) * logits * (-1 / temperature**2)
            )
            
            # Update temperature
            temperature -= lr * gradient
            temperature = max(0.1, min(temperature, 10.0))  # Clamp
        
        self._temperature = temperature
        self._fitted = True
        
        logger.debug(f"Temperature scaler fitted: T={self._temperature:.4f}")
        return self
    
    def calibrate(self, logit: float) -> float:
        """
        Calibrate a single logit.
        
        Args:
            logit: Raw logit value
            
        Returns:
            Calibrated probability
        """
        if not self._fitted:
            return 1.0 / (1.0 + np.exp(-logit))
        
        scaled = logit / self._temperature
        return float(1.0 / (1.0 + np.exp(-scaled)))
    
    def calibrate_batch(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate multiple logits."""
        scaled = logits / self._temperature
        return 1.0 / (1.0 + np.exp(-scaled))
    
    @property
    def temperature(self) -> float:
        """Get the learned temperature."""
        return self._temperature


class ModelCalibrator:
    """
    Unified calibration interface supporting multiple methods.
    """
    
    def __init__(self, method: str = "platt"):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', 'temperature')
        """
        self.method = method
        
        if method == "platt":
            self._calibrator = PlattScaler()
        elif method == "isotonic":
            self._calibrator = IsotonicCalibrator()
        elif method == "temperature":
            self._calibrator = TemperatureScaler()
        else:
            logger.warning(f"Unknown method: {method}, using Platt scaling")
            self._calibrator = PlattScaler()
        
        self._fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'ModelCalibrator':
        """
        Fit the calibrator on validation data.
        
        Args:
            scores: Raw model scores/probabilities
            labels: True binary labels
            
        Returns:
            Self for chaining
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        
        self._calibrator.fit(scores, labels)
        self._fitted = True
        
        logger.info(f"Calibrator fitted using {self.method} method")
        return self
    
    def calibrate(self, score: float) -> CalibrationResult:
        """
        Calibrate a single probability score.
        
        Args:
            score: Raw probability (0-1)
            
        Returns:
            CalibrationResult with calibrated probability
        """
        if not self._fitted:
            return CalibrationResult(
                original_prob=score,
                calibrated_prob=score,
                method=self.method
            )
        
        calibrated = self._calibrator.calibrate(score)
        
        return CalibrationResult(
            original_prob=score,
            calibrated_prob=calibrated,
            method=self.method
        )
    
    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate multiple scores."""
        if not self._fitted:
            return scores
        return self._calibrator.calibrate_batch(scores)
    
    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self._calibrator,
                'fitted': self._fitted
            }, f)
        logger.info(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelCalibrator':
        """Load calibrator from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(method=data['method'])
        calibrator._calibrator = data['calibrator']
        calibrator._fitted = data['fitted']
        
        logger.info(f"Calibrator loaded from {path}")
        return calibrator


def compute_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration error metrics.
    
    Args:
        probabilities: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary with calibration metrics
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(labels[mask])
            bin_confidence = np.mean(probabilities[mask])
            bin_weight = np.sum(mask) / len(probabilities)
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
    
    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(labels[mask])
            bin_confidence = np.mean(probabilities[mask])
            mce = max(mce, np.abs(bin_accuracy - bin_confidence))
    
    # Brier Score
    brier = np.mean((probabilities - labels) ** 2)
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': float(brier)
    }


def create_calibrator(method: str = "platt") -> ModelCalibrator:
    """Factory function to create calibrator."""
    return ModelCalibrator(method=method)

