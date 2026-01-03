"""
Cross-Validation Utilities for Fake Voice Detection.

Implements:
- K-fold cross-validation
- Stratified K-fold
- Time-series aware CV
- Nested CV for unbiased evaluation
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Callable, Iterator
import numpy as np

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross-validation utilities for model evaluation.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV folds
            stratified: Use stratified splitting
            shuffle: Shuffle data before splitting
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if self.stratified:
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        
        for train_idx, test_idx in splitter.split(X, y):
            yield train_idx, test_idx
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "accuracy"
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model with fit/predict methods
            X: Feature matrix
            y: Target labels
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import cross_validate as sklearn_cv
        
        if self.stratified:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        
        results = sklearn_cv(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        return results
    
    def nested_cross_validate(
        self,
        model_factory: Callable,
        param_grid: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        inner_cv: int = 3,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased evaluation.
        
        Args:
            model_factory: Function that returns a new model instance
            param_grid: Parameter grid for inner CV
            X: Feature matrix
            y: Target labels
            inner_cv: Number of inner CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with nested CV results
        """
        from sklearn.model_selection import GridSearchCV
        
        outer_scores = []
        best_params_list = []
        
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            model = model_factory()
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Evaluate on outer test set
            score = grid_search.score(X_test, y_test)
            outer_scores.append(score)
            best_params_list.append(grid_search.best_params_)
        
        return {
            'outer_scores': np.array(outer_scores),
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params_per_fold': best_params_list
        }


class TimeSeriesCV:
    """
    Time-series aware cross-validation.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None
    ):
        """
        Initialize time-series CV.
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and test sets
            test_size: Fixed test set size (None for expanding window)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices respecting temporal order.
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        ts_cv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
            test_size=self.test_size
        )
        
        for train_idx, test_idx in ts_cv.split(X):
            yield train_idx, test_idx


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    scoring: str = "accuracy"
) -> Dict[str, float]:
    """
    Convenience function for cross-validation.
    
    Returns:
        Dictionary with mean and std of scores
    """
    cv = CrossValidator(n_splits=n_splits, stratified=stratified)
    results = cv.cross_validate(model, X, y, scoring=scoring)
    
    return {
        'train_mean': float(np.mean(results['train_score'])),
        'train_std': float(np.std(results['train_score'])),
        'test_mean': float(np.mean(results['test_score'])),
        'test_std': float(np.std(results['test_score']))
    }

