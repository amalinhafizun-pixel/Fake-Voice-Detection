"""
Feature Selection Utilities for Fake Voice Detection.

Implements various feature selection methods:
- Wrapper methods (RFE, forward/backward selection)
- Filter methods (mutual information, chi-square)
- Embedded methods (L1 regularization, tree-based importance)
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection utilities for improving model performance.
    """
    
    def __init__(self, method: str = "mutual_info", n_features: int = 10):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'chi2', 'rfe', 'l1', 'tree')
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self._selector = None
        self._selected_indices = None
        self._feature_scores = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            
        Returns:
            Self for chaining
        """
        n_features = min(self.n_features, X.shape[1])
        
        if self.method == "mutual_info":
            self._fit_mutual_info(X, y, n_features)
        elif self.method == "chi2":
            self._fit_chi2(X, y, n_features)
        elif self.method == "rfe":
            self._fit_rfe(X, y, n_features)
        elif self.method == "l1":
            self._fit_l1(X, y, n_features)
        elif self.method == "tree":
            self._fit_tree(X, y, n_features)
        else:
            logger.warning(f"Unknown method: {self.method}, using mutual_info")
            self._fit_mutual_info(X, y, n_features)
        
        self._fitted = True
        return self
    
    def _fit_mutual_info(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        """Fit using mutual information."""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        self._selector = SelectKBest(score_func=mutual_info_classif, k=k)
        self._selector.fit(X, y)
        self._selected_indices = self._selector.get_support(indices=True)
        self._feature_scores = self._selector.scores_
    
    def _fit_chi2(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        """Fit using chi-squared test."""
        from sklearn.feature_selection import SelectKBest, chi2
        
        # Chi2 requires non-negative features
        X_pos = X - X.min(axis=0)
        
        self._selector = SelectKBest(score_func=chi2, k=k)
        self._selector.fit(X_pos, y)
        self._selected_indices = self._selector.get_support(indices=True)
        self._feature_scores = self._selector.scores_
    
    def _fit_rfe(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        """Fit using Recursive Feature Elimination."""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        self._selector = RFE(estimator, n_features_to_select=k)
        self._selector.fit(X, y)
        self._selected_indices = self._selector.get_support(indices=True)
        self._feature_scores = self._selector.ranking_
    
    def _fit_l1(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        """Fit using L1 regularization."""
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LogisticRegression
        
        estimator = LogisticRegression(penalty='l1', solver='saga', random_state=42, max_iter=1000)
        self._selector = SelectFromModel(estimator, max_features=k)
        self._selector.fit(X, y)
        self._selected_indices = self._selector.get_support(indices=True)
        self._feature_scores = np.abs(estimator.fit(X, y).coef_[0])
    
    def _fit_tree(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        """Fit using tree-based feature importance."""
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier
        
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        estimator.fit(X, y)
        
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        
        self._selected_indices = indices
        self._feature_scores = importances
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        if not self._fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        if self._selector is not None:
            return self._selector.transform(X)
        else:
            return X[:, self._selected_indices]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        return self._selected_indices
    
    def get_feature_scores(self) -> np.ndarray:
        """Get feature scores/importances."""
        return self._feature_scores
    
    def get_feature_ranking(self) -> List[Tuple[int, float]]:
        """Get ranked list of (feature_index, score) tuples."""
        if self._feature_scores is None:
            return []
        
        ranked = sorted(
            enumerate(self._feature_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "mutual_info",
    n_features: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for feature selection.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: Selection method
        n_features: Number of features to select
        
    Returns:
        Tuple of (selected_features, selected_indices)
    """
    selector = FeatureSelector(method=method, n_features=n_features)
    selected = selector.fit_transform(X, y)
    indices = selector.get_selected_indices()
    return selected, indices

