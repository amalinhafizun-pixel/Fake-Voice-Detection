# API Reference

Complete API documentation for the Fake Voice Detection System.

## Audio Processing

### `audio.processor.AudioProcessor`

```python
class AudioProcessor:
    """Audio processor for loading and preprocessing audio files."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: Optional[float] = None,
        normalize: bool = True
    ):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for resampling
            duration: Target duration in seconds (None for no truncation)
            normalize: Whether to normalize audio amplitude
        """
    
    def load(self, file_path: Union[str, Path]) -> AudioData:
        """
        Load an audio file and preprocess it.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioData object containing the processed audio
        """
    
    def compute_spectrogram(
        self,
        audio_data: AudioData,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: Optional[int] = None
    ) -> np.ndarray:
        """Compute spectrogram from audio data."""
    
    def compute_mfcc(
        self,
        audio_data: AudioData,
        n_mfcc: int = 13
    ) -> np.ndarray:
        """Compute MFCCs from audio data."""
```

### `audio.processor.AudioData`

```python
@dataclass
class AudioData:
    """Container for processed audio data."""
    waveform: np.ndarray
    sample_rate: int
    duration: float
    channels: int
    file_path: Optional[str] = None
    format: Optional[str] = None
```

## Detection Modules

### `detection.signal_features.SignalFeatureExtractor`

```python
class SignalFeatureExtractor:
    """Extracts signal-level audio features."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160
    ):
        """Initialize the feature extractor."""
    
    def extract(
        self,
        waveform: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> SignalFeatures:
        """Extract all signal features from audio waveform."""
    
    def compute_anomaly_score(self, features: SignalFeatures) -> float:
        """Compute anomaly score based on signal features (0-1)."""
```

### `detection.deep_learning.DeepLearningDetector`

```python
class DeepLearningDetector:
    """Deep learning-based detector using AASIST model."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Any] = None,
        use_mlx: bool = False
    ):
        """Initialize the detector."""
    
    def detect(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> DeepLearningResult:
        """
        Detect if audio is spoofed.
        
        Returns:
            DeepLearningResult with spoof_probability (0-1)
        """
```

### `detection.behavioral.BehavioralAnalyzer`

```python
class BehavioralAnalyzer:
    """Analyzes behavioral patterns in speech."""
    
    def analyze(
        self,
        waveform: np.ndarray,
        sample_rate: Optional[int] = None,
        transcription: Optional[str] = None
    ) -> BehavioralResult:
        """
        Perform behavioral analysis.
        
        Returns:
            BehavioralResult with anomaly_score (0-1)
        """
```

### `detection.linguistic.LinguisticAnalyzer`

```python
class LinguisticAnalyzer:
    """Analyzes linguistic patterns using Whisper."""
    
    def __init__(
        self,
        whisper_model: str = "base",
        device: Optional[str] = None
    ):
        """Initialize the analyzer."""
    
    def transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """Transcribe audio using Whisper."""
    
    def analyze(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        transcription: Optional[str] = None
    ) -> LinguisticResult:
        """Perform linguistic analysis."""
```

### `detection.anomaly_detection.AnomalyDetector`

```python
class AnomalyDetector:
    """Ensemble anomaly detector."""
    
    def __init__(
        self,
        methods: List[str] = None,
        weights: Dict[str, float] = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            methods: List of methods ('isolation_forest', 'one_class_svm', 'lof')
            weights: Weights for each method
        """
    
    def fit(self, features: np.ndarray) -> 'AnomalyDetector':
        """Fit all anomaly detectors on normal data."""
    
    def predict(self, features: np.ndarray) -> AnomalyResult:
        """Predict anomaly score using ensemble."""
```

### `detection.ensemble.EnsembleDetector`

```python
class EnsembleDetector:
    """Main ensemble detector combining all methods."""
    
    def __init__(
        self,
        ensemble_method: str = "weighted",
        weights: Dict[str, float] = None,
        use_dynamic_weights: bool = True
    ):
        """Initialize the ensemble detector."""
    
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
        
        Returns:
            DetectionResult with overall_score and risk_level
        """
```

### `detection.calibration.ModelCalibrator`

```python
class ModelCalibrator:
    """Unified calibration interface."""
    
    def __init__(self, method: str = "platt"):
        """
        Initialize the calibrator.
        
        Args:
            method: 'platt', 'isotonic', or 'temperature'
        """
    
    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> 'ModelCalibrator':
        """Fit the calibrator on validation data."""
    
    def calibrate(self, score: float) -> CalibrationResult:
        """Calibrate a single probability score."""
```

## Backend

### `backend.device_manager.DeviceManager`

```python
class DeviceManager:
    """Manages device detection and selection."""
    
    def __init__(self, preferred_backend: Optional[str] = None):
        """Initialize the device manager."""
    
    def select_backend(
        self,
        backend_arg: Optional[str] = None
    ) -> BackendType:
        """
        Select the best available backend.
        
        Args:
            backend_arg: 'MSilicon', 'CUDA', 'ROCm', or None
            
        Returns:
            Selected BackendType
        """
    
    def to_device(self, tensor: Any) -> Any:
        """Move a tensor to the current device."""
```

## ML Utilities

### `ml.feature_selector.FeatureSelector`

```python
class FeatureSelector:
    """Feature selection utilities."""
    
    def __init__(
        self,
        method: str = "mutual_info",
        n_features: int = 10
    ):
        """
        Initialize feature selector.
        
        Args:
            method: 'mutual_info', 'chi2', 'rfe', 'l1', 'tree'
            n_features: Number of features to select
        """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector."""
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
```

### `ml.model_evaluation.ModelEvaluator`

```python
class ModelEvaluator:
    """Model evaluation utilities."""
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, roc_auc, pr_auc
        """
    
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix."""
```

## UI Components

### `ui.main_window.MainWindow`

```python
class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, backend: Optional[str] = None):
        """Initialize the main window."""

def run_gui(backend: Optional[str] = None) -> None:
    """Run the PyQt6 GUI application."""
```

### `terminal.cli.TerminalInterface`

```python
class TerminalInterface:
    """Terminal interface for fake voice detection."""
    
    def analyze_file(
        self,
        file_path: str,
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a single audio file."""
    
    def run_interactive(self) -> None:
        """Run interactive mode."""

def run_cli(
    file_path: Optional[str] = None,
    backend: Optional[str] = None,
    json_output: bool = False,
    interactive: bool = False
) -> Optional[Dict[str, Any]]:
    """Run the CLI application."""
```

