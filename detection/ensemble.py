"""
Ensemble Methods for Fake Voice Detection.

Combines multiple detection methods using:
- Weighted scoring (baseline)
- Stacking ensemble with meta-learner
- Boosting ensemble
- Voting ensemble
- Dynamic weight adjustment
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Unified detection result from ensemble."""
    overall_score: float
    risk_level: str
    confidence: str
    component_scores: Dict[str, float]
    interpretation: str
    raw_predictions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'risk_level': self.risk_level,
            'confidence': self.confidence,
            'component_scores': self.component_scores,
            'interpretation': self.interpretation,
        }
    
    def to_json(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""
        return {
            'spoof_probability': self.overall_score,
            'risk_level': self.risk_level,
            'confidence': self.confidence,
            'scores': self.component_scores,
            'interpretation': self.interpretation,
        }


class WeightedEnsemble:
    """
    Simple weighted average ensemble.
    """
    
    DEFAULT_WEIGHTS = {
        'deep_learning': 0.25,  # Reduced - can be unreliable if model not properly trained
        'behavioral': 0.20,
        'signal': 0.30,  # Increased - more reliable for robotic voices
        'linguistic': 0.15,
        'anomaly': 0.10,
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize weighted ensemble.
        
        Args:
            weights: Component weights (must sum to 1)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def combine(self, scores: Dict[str, float]) -> float:
        """
        Combine scores using weighted average.
        
        Args:
            scores: Dictionary of component scores
            
        Returns:
            Combined score (0-1)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            weight = self.weights.get(component, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    """
    
    def __init__(self, meta_learner: str = "logistic"):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_learner: Type of meta-learner ('logistic', 'xgboost', 'lightgbm')
        """
        self.meta_learner_type = meta_learner
        self._meta_model = None
        self._fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> 'StackingEnsemble':
        """
        Fit the meta-learner on base model predictions.
        
        Args:
            predictions: Base model predictions (n_samples, n_models)
            labels: True labels
            
        Returns:
            Self for chaining
        """
        if self.meta_learner_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self._meta_model = LogisticRegression(random_state=42)
        elif self.meta_learner_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                self._meta_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                logger.warning("XGBoost not available, using LogisticRegression")
                from sklearn.linear_model import LogisticRegression
                self._meta_model = LogisticRegression(random_state=42)
        elif self.meta_learner_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                self._meta_model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=42,
                    verbose=-1
                )
            except ImportError:
                logger.warning("LightGBM not available, using LogisticRegression")
                from sklearn.linear_model import LogisticRegression
                self._meta_model = LogisticRegression(random_state=42)
        
        self._meta_model.fit(predictions, labels)
        self._fitted = True
        return self
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Make predictions using the meta-learner.
        
        Args:
            predictions: Base model predictions
            
        Returns:
            Probability of being spoofed
        """
        if not self._fitted:
            raise ValueError("Meta-learner not fitted")
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        
        if hasattr(self._meta_model, 'predict_proba'):
            probs = self._meta_model.predict_proba(predictions)
            return probs[:, 1]  # Probability of positive class (spoof)
        else:
            return self._meta_model.predict(predictions).astype(float)


class VotingEnsemble:
    """
    Voting ensemble (hard, soft, or weighted voting).
    """
    
    def __init__(self, voting: str = "soft", weights: Dict[str, float] = None):
        """
        Initialize voting ensemble.
        
        Args:
            voting: Voting type ('hard', 'soft', 'weighted')
            weights: Weights for weighted voting
        """
        self.voting = voting
        self.weights = weights
    
    def combine(
        self,
        predictions: Dict[str, float],
        threshold: float = 0.5
    ) -> Tuple[float, bool]:
        """
        Combine predictions using voting.
        
        Args:
            predictions: Dictionary of model predictions (probabilities)
            threshold: Decision threshold for hard voting
            
        Returns:
            Tuple of (combined_probability, is_spoof)
        """
        if not predictions:
            return 0.5, False
        
        probs = list(predictions.values())
        names = list(predictions.keys())
        
        if self.voting == "hard":
            # Majority vote
            votes = [1 if p > threshold else 0 for p in probs]
            is_spoof = sum(votes) > len(votes) / 2
            combined = float(is_spoof)
            
        elif self.voting == "soft":
            # Average probability
            combined = np.mean(probs)
            is_spoof = combined > threshold
            
        elif self.voting == "weighted":
            # Weighted average
            if self.weights:
                weighted_sum = sum(p * self.weights.get(n, 1.0) for n, p in zip(names, probs))
                total_weight = sum(self.weights.get(n, 1.0) for n in names)
                combined = weighted_sum / total_weight if total_weight > 0 else 0.5
            else:
                combined = np.mean(probs)
            is_spoof = combined > threshold
        
        else:
            combined = np.mean(probs)
            is_spoof = combined > threshold
        
        return float(combined), is_spoof


class DynamicWeightAdjuster:
    """
    Dynamically adjusts ensemble weights based on confidence and context.
    """
    
    def __init__(self, base_weights: Dict[str, float] = None):
        """
        Initialize dynamic weight adjuster.
        
        Args:
            base_weights: Base weights for components
        """
        self.base_weights = base_weights or WeightedEnsemble.DEFAULT_WEIGHTS.copy()
    
    def adjust_weights(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Adjust weights based on confidence levels.
        
        Args:
            scores: Component scores
            confidences: Confidence levels ('HIGH', 'MEDIUM', 'LOW')
            
        Returns:
            Adjusted weights
        """
        adjusted = {}
        
        confidence_multipliers = {
            'HIGH': 1.5,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        for component in self.base_weights:
            base = self.base_weights[component]
            confidence = confidences.get(component, 'MEDIUM')
            multiplier = confidence_multipliers.get(confidence, 1.0)
            adjusted[component] = base * multiplier
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted


class EnsembleDetector:
    """
    Main ensemble detector combining all detection methods.
    """
    
    def __init__(
        self,
        ensemble_method: str = "weighted",
        weights: Dict[str, float] = None,
        use_dynamic_weights: bool = True
    ):
        """
        Initialize the ensemble detector.
        
        Args:
            ensemble_method: Method for combining ('weighted', 'stacking', 'voting')
            weights: Component weights
            use_dynamic_weights: Whether to use dynamic weight adjustment
        """
        self.ensemble_method = ensemble_method
        self.use_dynamic_weights = use_dynamic_weights
        
        self.weighted_ensemble = WeightedEnsemble(weights)
        self.voting_ensemble = VotingEnsemble(voting="soft", weights=weights)
        self.dynamic_adjuster = DynamicWeightAdjuster(weights)
        self.stacking_ensemble = None  # Initialized when trained
    
    def detect(
        self,
        deep_learning_score: float = 0.0,
        behavioral_score: float = 0.0,
        signal_score: float = 0.0,
        linguistic_score: float = 0.0,
        anomaly_score: float = 0.0,
        confidences: Dict[str, str] = None
    ) -> DetectionResult:
        """
        Perform ensemble detection.
        
        Args:
            deep_learning_score: Score from AASIST model (0-1)
            behavioral_score: Score from behavioral analysis (0-1)
            signal_score: Score from signal features (0-1)
            linguistic_score: Score from linguistic analysis (0-1)
            anomaly_score: Score from anomaly detection (0-1)
            confidences: Confidence levels for each component
            
        Returns:
            DetectionResult with combined score and interpretation
        """
        scores = {
            'deep_learning': deep_learning_score,
            'behavioral': behavioral_score,
            'signal': signal_score,
            'linguistic': linguistic_score,
            'anomaly': anomaly_score,
        }
        
        confidences = confidences or {}
        
        # Get weights
        if self.use_dynamic_weights and confidences:
            weights = self.dynamic_adjuster.adjust_weights(scores, confidences)
            self.weighted_ensemble.weights = weights
            self.voting_ensemble.weights = weights
        
        # Combine scores
        if self.ensemble_method == "weighted":
            overall_score = self.weighted_ensemble.combine(scores)
        elif self.ensemble_method == "voting":
            overall_score, _ = self.voting_ensemble.combine(scores)
        elif self.ensemble_method == "stacking" and self.stacking_ensemble is not None:
            pred_array = np.array(list(scores.values())).reshape(1, -1)
            overall_score = float(self.stacking_ensemble.predict(pred_array)[0])
        else:
            overall_score = self.weighted_ensemble.combine(scores)
        
        # Determine risk level (adjusted thresholds for better sensitivity)
        # Lower thresholds to catch more synthetic voices
        if overall_score < 0.25:
            risk_level = "LOW"
        elif overall_score < 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Determine confidence
        score_std = np.std(list(scores.values()))
        if score_std < 0.1:
            confidence = "HIGH"
        elif score_std < 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Generate interpretation
        interpretation = self._generate_interpretation(scores, overall_score, risk_level)
        
        return DetectionResult(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            component_scores=scores,
            interpretation=interpretation,
            raw_predictions={}
        )
    
    def _generate_interpretation(
        self,
        scores: Dict[str, float],
        overall_score: float,
        risk_level: str
    ) -> str:
        """Generate human-readable interpretation."""
        high_scores = [k for k, v in scores.items() if v > 0.6]
        
        if risk_level == "LOW":
            return "Audio appears to be genuine human speech."
        elif risk_level == "MEDIUM":
            if high_scores:
                concerns = ", ".join(high_scores).replace("_", " ")
                return f"Some indicators of synthetic speech detected in: {concerns}."
            return "Mixed indicators - manual review recommended."
        else:  # HIGH
            if len(high_scores) >= 3:
                return "Strong evidence of AI-generated or synthetic speech."
            elif high_scores:
                concerns = ", ".join(high_scores).replace("_", " ")
                return f"High likelihood of synthetic speech based on: {concerns}."
            return "Audio likely contains AI-generated content."
    
    def train_stacking(
        self,
        training_predictions: np.ndarray,
        labels: np.ndarray,
        meta_learner: str = "logistic"
    ) -> None:
        """
        Train stacking ensemble meta-learner.
        
        Args:
            training_predictions: Base model predictions (n_samples, n_models)
            labels: True labels
            meta_learner: Type of meta-learner
        """
        self.stacking_ensemble = StackingEnsemble(meta_learner=meta_learner)
        self.stacking_ensemble.fit(training_predictions, labels)
        logger.info(f"Stacking ensemble trained with {meta_learner}")


def create_ensemble_detector(
    method: str = "weighted",
    weights: Dict[str, float] = None,
    dynamic_weights: bool = True
) -> EnsembleDetector:
    """
    Factory function to create ensemble detector.
    
    Args:
        method: Ensemble method
        weights: Component weights
        dynamic_weights: Use dynamic weight adjustment
        
    Returns:
        Configured EnsembleDetector
    """
    return EnsembleDetector(
        ensemble_method=method,
        weights=weights,
        use_dynamic_weights=dynamic_weights
    )

