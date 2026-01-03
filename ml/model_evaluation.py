"""
Model Evaluation Utilities for Fake Voice Detection.

Implements:
- Standard metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix
- ROC and PR curves
- Calibration curves
- Feature importance visualization
"""

import logging
from typing import Dict, Optional, Any, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation utilities.
    """
    
    def __init__(self):
        self._results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            except ValueError:
                # AUC undefined for single class
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        self._results = metrics
        return metrics
    
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            2x2 confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
    
    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: List[str] = None
    ) -> str:
        """
        Generate classification report.
        
        Returns:
            Formatted classification report string
        """
        from sklearn.metrics import classification_report
        target_names = target_names or ['Genuine', 'Spoof']
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def roc_curve_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        from sklearn.metrics import roc_curve
        return roc_curve(y_true, y_prob)
    
    def precision_recall_curve_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve data.
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        from sklearn.metrics import precision_recall_curve
        return precision_recall_curve(y_true, y_prob)
    
    def calibration_curve_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve data.
        
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value)
        """
        from sklearn.calibration import calibration_curve
        return calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    def feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract feature importance from model.
        
        Args:
            model: Fitted model with feature_importances_ or coef_
            feature_names: Names for features
            
        Returns:
            Dictionary of feature name to importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't have feature importances")
            return {}
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
    
    def get_results(self) -> Dict[str, float]:
        """Get last evaluation results."""
        return self._results


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Convenience function for model evaluation.
    
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate(y_true, y_pred, y_prob)


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> None:
    """
    Print comprehensive evaluation report.
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_prob)
    
    print("\n" + "=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"\nROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR AUC:    {metrics['pr_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = evaluator.confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(evaluator.classification_report(y_true, y_pred))

