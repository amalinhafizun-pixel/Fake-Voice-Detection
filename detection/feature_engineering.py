"""
Feature Engineering for Fake Voice Detection.

Provides utilities for:
- Feature selection (RFE, Mutual Information)
- Feature normalization (StandardScaler, MinMaxScaler, RobustScaler)
- Dimensionality reduction (PCA, t-SNE)
- Feature aggregation (statistical, temporal)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering."""
    original_features: np.ndarray
    processed_features: np.ndarray
    feature_names: List[str]
    selected_indices: Optional[np.ndarray]
    pca_explained_variance: Optional[float]
    scaler_params: Optional[Dict[str, Any]]


class FeatureEngineer:
    """
    Feature engineering utilities for fake voice detection.
    
    Handles feature selection, normalization, and dimensionality reduction.
    """
    
    def __init__(
        self,
        normalize_method: str = "standard",
        use_pca: bool = False,
        pca_variance: float = 0.95,
        select_features: bool = False,
        n_features: Optional[int] = None
    ):
        """
        Initialize the feature engineer.
        
        Args:
            normalize_method: Normalization method ('standard', 'minmax', 'robust', 'none')
            use_pca: Whether to apply PCA
            pca_variance: Variance to retain with PCA (0-1)
            select_features: Whether to perform feature selection
            n_features: Number of features to select (None for auto)
        """
        self.normalize_method = normalize_method
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.select_features = select_features
        self.n_features = n_features
        
        # Fitted transformers
        self._scaler = None
        self._pca = None
        self._selector = None
        self._fitted = False
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels (required for feature selection)
            
        Returns:
            Self for chaining
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # Fit scaler
        if self.normalize_method == "standard":
            self._scaler = StandardScaler()
        elif self.normalize_method == "minmax":
            self._scaler = MinMaxScaler()
        elif self.normalize_method == "robust":
            self._scaler = RobustScaler()
        else:
            self._scaler = None
        
        if self._scaler is not None:
            scaled_features = self._scaler.fit_transform(features)
        else:
            scaled_features = features.copy()
        
        # Fit feature selector
        if self.select_features and labels is not None:
            self._fit_selector(scaled_features, labels)
            selected_features = self._selector.transform(scaled_features)
        else:
            selected_features = scaled_features
        
        # Fit PCA
        if self.use_pca:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.pca_variance)
            self._pca.fit(selected_features)
        
        self._fitted = True
        return self
    
    def _fit_selector(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit feature selector using mutual information."""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        n_features = self.n_features or max(features.shape[1] // 2, 5)
        n_features = min(n_features, features.shape[1])
        
        self._selector = SelectKBest(
            score_func=mutual_info_classif,
            k=n_features
        )
        self._selector.fit(features, labels)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted transformers.
        
        Args:
            features: Features to transform
            
        Returns:
            Transformed features
        """
        if not self._fitted:
            logger.warning("FeatureEngineer not fitted, returning original features")
            return features
        
        result = features.copy()
        
        # Normalize
        if self._scaler is not None:
            result = self._scaler.transform(result)
        
        # Feature selection
        if self._selector is not None:
            result = self._selector.transform(result)
        
        # PCA
        if self._pca is not None:
            result = self._pca.transform(result)
        
        return result
    
    def fit_transform(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features, labels)
        return self.transform(features)
    
    def get_selected_feature_indices(self) -> Optional[np.ndarray]:
        """Get indices of selected features."""
        if self._selector is not None:
            return self._selector.get_support(indices=True)
        return None
    
    def get_pca_explained_variance(self) -> Optional[float]:
        """Get total explained variance ratio from PCA."""
        if self._pca is not None:
            return float(np.sum(self._pca.explained_variance_ratio_))
        return None


class FeatureAggregator:
    """
    Aggregates multiple feature sets into a single feature vector,
    enhanced with temporal delta and delta-delta variations.
    """
    
    def __init__(self, aggregation_stats: List[str] = None, compute_dynamics: bool = True):
        """
        Initialize the aggregator.
        
        Args:
            aggregation_stats: Statistics to compute ('mean', 'std', 'min', 'max', 'percentile_25', 'percentile_75')
            compute_dynamics: Whether to automatically inject velocity (delta) and acceleration (delta-delta) features
        """
        self.aggregation_stats = aggregation_stats or ['mean', 'std', 'min', 'max']
        self.compute_dynamics = compute_dynamics
    
    def compute_delta_features(self, feat: np.ndarray, width: int = 3) -> np.ndarray:
        """
        Compute delta (velocity) and delta-delta (acceleration) features 
        to capture unnatural, rigid transitions typical in synthetic voices.
        """
        if len(feat) < width:
            return feat  # Not enough data points to compute derivatives
        
        # Calculate Delta (First Derivative / Velocity)
        delta = np.zeros_like(feat)
        n = (width - 1) // 2
        for t in range(n, len(feat) - n):
            numerator = sum(k * (feat[t + k] - feat[t - k]) for k in range(1, n + 1))
            denominator = 2 * sum(k**2 for k in range(1, n + 1))
            delta[t] = numerator / denominator
            
        # Calculate Delta-Delta (Second Derivative / Acceleration)
        delta_delta = np.zeros_like(delta)
        for t in range(n, len(delta) - n):
            numerator = sum(k * (delta[t + k] - delta[t - k]) for k in range(1, n + 1))
            denominator = 2 * sum(k**2 for k in range(1, n + 1))
            delta_delta[t] = numerator / denominator
            
        # Stack original features, velocity, and acceleration together
        return np.hstack([feat, delta, delta_delta])
    
    def aggregate(
        self,
        features: Dict[str, np.ndarray],
        include_raw: bool = False
    ) -> np.ndarray:
        """
        Aggregate multiple feature arrays into a single vector.
        
        Args:
            features: Dictionary of named feature arrays
            include_raw: Whether to include raw features (first N values)
            
        Returns:
            Aggregated feature vector
        """
        aggregated = []
        
        for name, feat in features.items():
            if feat is None or len(feat) == 0:
                continue
            
            feat = np.asarray(feat).flatten()
            
            # UPGRADE: Inject transitional speech dynamics before processing statistics
            if self.compute_dynamics:
                feat = self.compute_delta_features(feat)
            
            for stat in self.aggregation_stats:
                if stat == 'mean':
                    aggregated.append(np.mean(feat))
                elif stat == 'std':
                    aggregated.append(np.std(feat))
                elif stat == 'min':
                    aggregated.append(np.min(feat))
                elif stat == 'max':
                    aggregated.append(np.max(feat))
                elif stat == 'percentile_25':
                    aggregated.append(np.percentile(feat, 25))
                elif stat == 'percentile_75':
                    aggregated.append(np.percentile(feat, 75))
                elif stat == 'median':
                    aggregated.append(np.median(feat))
            
            # Include first few raw values if requested
            if include_raw:
                n_raw = min(5, len(feat))
                aggregated.extend(feat[:n_raw])
        
        return np.array(aggregated, dtype=np.float32)
    
    def compute_temporal_features(
        self,
        time_series: np.ndarray,
        window_size: int = 10
    ) -> np.ndarray:
        """
        Compute temporal features from time series data.
        
        Args:
            time_series: 1D time series data
            window_size: Window size for rolling statistics
            
        Returns:
            Temporal feature vector
        """
        if len(time_series) < window_size:
            window_size = max(len(time_series) // 2, 1)
        
        features = []
        
        # Rolling statistics
        for i in range(0, len(time_series) - window_size, window_size // 2):
            window = time_series[i:i + window_size]
            features.append(np.mean(window))
            features.append(np.std(window))
        
        if len(features) == 0:
            features = [np.mean(time_series), np.std(time_series)]
        
        # First and second derivatives
        if len(time_series) > 1:
            first_diff = np.diff(time_series)
            features.append(np.mean(np.abs(first_diff)))
            features.append(np.std(first_diff))
            
            if len(first_diff) > 1:
                second_diff = np.diff(first_diff)
                features.append(np.mean(np.abs(second_diff)))
        
        return np.array(features, dtype=np.float32)


class PolynomialFeatureGenerator:
    """
    Generates polynomial and interaction features.
    """
    
    def __init__(self, degree: int = 2, interaction_only: bool = False):
        """
        Initialize the generator.
        
        Args:
            degree: Maximum polynomial degree
            interaction_only: Only include interaction terms
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self._transformer = None
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Generate polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        
        self._transformer = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=False
        )
        
        return self._transformer.fit_transform(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform using fitted generator."""
        if self._transformer is None:
            return self.fit_transform(features)
        return self._transformer.transform(features)


def normalize_features(
    features: np.ndarray,
    method: str = "standard"
) -> Tuple[np.ndarray, Any]:
    """
    Normalize features using specified method.
    
    Args:
        features: Feature array
        method: 'standard', 'minmax', or 'robust'
        
    Returns:
        Tuple of (normalized_features, fitted_scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        return features, None
    
    normalized = scaler.fit_transform(features)
    return normalized, scaler


def apply_pca(
    features: np.ndarray,
    n_components: Union[int, float] = 0.95
) -> Tuple[np.ndarray, Any, float]:
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        features: Feature array
        n_components: Number of components or variance to retain
        
    Returns:
        Tuple of (reduced_features, pca_model, explained_variance)
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    explained = np.sum(pca.explained_variance_ratio_)
    
    return reduced, pca, float(explained)


def select_features_mutual_info(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select top-k features using mutual information.
    
    Args:
        features: Feature array
        labels: Class labels
        k: Number of features to select
        
    Returns:
        Tuple of (selected_features, selected_indices, scores)
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    k = min(k, features.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selected = selector.fit_transform(features, labels)
    indices = selector.get_support(indices=True)
    scores = selector.scores_
    
    return selected, indices, scores


def combine_feature_sets(*feature_arrays: np.ndarray) -> np.ndarray:
    """
    Combine multiple feature arrays horizontally.
    
    Args:
        feature_arrays: Variable number of feature arrays
        
    Returns:
        Combined feature array
    """
    valid_arrays = [f for f in feature_arrays if f is not None and len(f) > 0]
    
    if len(valid_arrays) == 0:
        return np.array([])
    
    if len(valid_arrays) == 1:
        return valid_arrays[0]
    
    # Ensure 2D
    arrays_2d = []
    for arr in valid_arrays:
        if arr.ndim == 1:
            arr = arr.reshape(1, -1) if len(arr.shape) == 1 else arr
        arrays_2d.append(arr)
    
    return np.hstack(arrays_2d)

