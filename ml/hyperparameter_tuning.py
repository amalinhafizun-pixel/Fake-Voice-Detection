"""
Hyperparameter Tuning for Fake Voice Detection.

Implements:
- Optuna-based Bayesian optimization
- Grid search
- Random search
- Multi-objective optimization
"""

import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna or sklearn.
    """
    
    def __init__(
        self,
        method: str = "optuna",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize the tuner.
        
        Args:
            method: Tuning method ('optuna', 'grid', 'random')
            n_trials: Number of trials for optimization
            timeout: Timeout in seconds (None for no limit)
            random_state: Random seed
        """
        self.method = method
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self._study = None
        self._best_params = None
        self._best_score = None
    
    def tune(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        direction: str = "maximize"
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            objective: Objective function to optimize
            param_space: Parameter search space
            direction: 'maximize' or 'minimize'
            
        Returns:
            Best parameters found
        """
        if self.method == "optuna":
            return self._tune_optuna(objective, param_space, direction)
        elif self.method == "grid":
            return self._tune_grid(objective, param_space, direction)
        elif self.method == "random":
            return self._tune_random(objective, param_space, direction)
        else:
            logger.warning(f"Unknown method: {self.method}, using optuna")
            return self._tune_optuna(objective, param_space, direction)
    
    def _tune_optuna(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        direction: str
    ) -> Dict[str, Any]:
        """Tune using Optuna."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def optuna_objective(trial):
                params = {}
                for name, spec in param_space.items():
                    if spec['type'] == 'float':
                        params[name] = trial.suggest_float(
                            name, spec['low'], spec['high'],
                            log=spec.get('log', False)
                        )
                    elif spec['type'] == 'int':
                        params[name] = trial.suggest_int(
                            name, spec['low'], spec['high'],
                            log=spec.get('log', False)
                        )
                    elif spec['type'] == 'categorical':
                        params[name] = trial.suggest_categorical(name, spec['choices'])
                
                return objective(params)
            
            self._study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            
            self._study.optimize(
                optuna_objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            self._best_params = self._study.best_params
            self._best_score = self._study.best_value
            
            logger.info(f"Best score: {self._best_score:.4f}")
            return self._best_params
            
        except ImportError:
            logger.warning("Optuna not installed, falling back to random search")
            return self._tune_random(objective, param_space, direction)
    
    def _tune_grid(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        direction: str
    ) -> Dict[str, Any]:
        """Tune using grid search."""
        from itertools import product
        
        # Generate grid
        param_names = list(param_space.keys())
        param_values = []
        
        for name in param_names:
            spec = param_space[name]
            if spec['type'] in ('float', 'int'):
                # Generate evenly spaced values
                n_points = spec.get('n_points', 5)
                values = np.linspace(spec['low'], spec['high'], n_points)
                if spec['type'] == 'int':
                    values = np.unique(values.astype(int))
                param_values.append(values)
            elif spec['type'] == 'categorical':
                param_values.append(spec['choices'])
        
        best_score = float('-inf') if direction == "maximize" else float('inf')
        best_params = None
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            score = objective(params)
            
            if direction == "maximize" and score > best_score:
                best_score = score
                best_params = params
            elif direction == "minimize" and score < best_score:
                best_score = score
                best_params = params
        
        self._best_params = best_params
        self._best_score = best_score
        return best_params
    
    def _tune_random(
        self,
        objective: Callable,
        param_space: Dict[str, Any],
        direction: str
    ) -> Dict[str, Any]:
        """Tune using random search."""
        np.random.seed(self.random_state)
        
        best_score = float('-inf') if direction == "maximize" else float('inf')
        best_params = None
        
        for _ in range(self.n_trials):
            params = {}
            for name, spec in param_space.items():
                if spec['type'] == 'float':
                    if spec.get('log', False):
                        log_val = np.random.uniform(np.log(spec['low']), np.log(spec['high']))
                        params[name] = np.exp(log_val)
                    else:
                        params[name] = np.random.uniform(spec['low'], spec['high'])
                elif spec['type'] == 'int':
                    params[name] = np.random.randint(spec['low'], spec['high'] + 1)
                elif spec['type'] == 'categorical':
                    params[name] = np.random.choice(spec['choices'])
            
            score = objective(params)
            
            if direction == "maximize" and score > best_score:
                best_score = score
                best_params = params
            elif direction == "minimize" and score < best_score:
                best_score = score
                best_params = params
        
        self._best_params = best_params
        self._best_score = best_score
        return best_params
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found."""
        return self._best_params
    
    def get_best_score(self) -> Optional[float]:
        """Get best score achieved."""
        return self._best_score
    
    def get_optimization_history(self) -> Optional[List[Tuple[int, float]]]:
        """Get optimization history (trial number, score)."""
        if self._study is not None:
            return [(t.number, t.value) for t in self._study.trials if t.value is not None]
        return None


def tune_ensemble_weights(
    X: np.ndarray,
    y: np.ndarray,
    base_predictions: Dict[str, np.ndarray],
    n_trials: int = 50
) -> Dict[str, float]:
    """
    Tune ensemble weights using Optuna.
    
    Args:
        X: Feature matrix (unused, for compatibility)
        y: True labels
        base_predictions: Dictionary of base model predictions
        n_trials: Number of optimization trials
        
    Returns:
        Optimized weights for each base model
    """
    model_names = list(base_predictions.keys())
    
    def objective(params):
        # Normalize weights
        weights = np.array([params[name] for name in model_names])
        weights = weights / weights.sum()
        
        # Compute weighted prediction
        weighted_pred = np.zeros_like(list(base_predictions.values())[0])
        for i, name in enumerate(model_names):
            weighted_pred += weights[i] * base_predictions[name]
        
        # Compute accuracy
        predictions = (weighted_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    param_space = {
        name: {'type': 'float', 'low': 0.01, 'high': 1.0}
        for name in model_names
    }
    
    tuner = HyperparameterTuner(method="optuna", n_trials=n_trials)
    best_params = tuner.tune(objective, param_space, direction="maximize")
    
    # Normalize final weights
    total = sum(best_params.values())
    return {name: val / total for name, val in best_params.items()}

