"""
Main Window for Fake Voice Detection PyQt6 GUI.

Provides the main application window with:
- File upload
- Analysis controls
- Results display
- Settings
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QStatusBar, QMessageBox,
    QTabWidget, QFrame, QComboBox, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from typing import Optional, Dict, Any
import logging

from .file_upload import FileUploadWidget
from .results_display import ResultsDisplay

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """
    Worker thread for running analysis in background.
    """
    
    progress = pyqtSignal(str, int)  # (message, percentage)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path: str, backend: Optional[str] = None):
        super().__init__()
        self.file_path = file_path
        self.backend = backend
    
    def run(self) -> None:
        """Run the analysis."""
        try:
            self.progress.emit("Loading audio file...", 10)
            
            from audio.processor import AudioProcessor
            processor = AudioProcessor()
            audio_data = processor.load(self.file_path)
            
            self.progress.emit("Extracting signal features...", 25)
            from detection.signal_features import SignalFeatureExtractor
            signal_extractor = SignalFeatureExtractor(sample_rate=audio_data.sample_rate)
            signal_features = signal_extractor.extract(audio_data.waveform)
            signal_score = signal_extractor.compute_anomaly_score(signal_features)
            
            self.progress.emit("Running deep learning model...", 40)
            from detection.deep_learning import DeepLearningDetector
            from config import RESNET18_WEIGHTS_PATH, RESNET18_FULL_PATH
            import os
            # Prefer full model if available, otherwise use weights
            model_path = str(RESNET18_FULL_PATH) if os.path.exists(RESNET18_FULL_PATH) else str(RESNET18_WEIGHTS_PATH)
            dl_detector = DeepLearningDetector(model_path=model_path if os.path.exists(model_path) else None)
            dl_result = dl_detector.detect(audio_data.waveform, audio_data.sample_rate)
            
            self.progress.emit("Analyzing behavioral patterns...", 55)
            from detection.behavioral import BehavioralAnalyzer
            behavioral = BehavioralAnalyzer(sample_rate=audio_data.sample_rate)
            behavioral_result = behavioral.analyze(audio_data.waveform)
            
            self.progress.emit("Performing linguistic analysis...", 70)
            from detection.linguistic import LinguisticAnalyzer
            linguistic = LinguisticAnalyzer()
            linguistic_result = linguistic.analyze(audio_data.waveform, audio_data.sample_rate)
            
            self.progress.emit("Running anomaly detection...", 85)
            feature_vector = signal_features.to_vector()
            from detection.anomaly_detection import AnomalyDetector
            anomaly_detector = AnomalyDetector()
            # Pass signal_features object for better heuristic when not fitted
            anomaly_result = anomaly_detector.predict(feature_vector, signal_features=signal_features)
            
            self.progress.emit("Computing ensemble score...", 95)
            from detection.ensemble import EnsembleDetector
            ensemble = EnsembleDetector()
            
            result = ensemble.detect(
                deep_learning_score=dl_result.spoof_probability,
                behavioral_score=behavioral_result.anomaly_score,
                signal_score=signal_score,
                linguistic_score=linguistic_result.anomaly_score,
                anomaly_score=anomaly_result.anomaly_score,
                confidences={
                    'deep_learning': dl_result.confidence,
                    'behavioral': behavioral_result.confidence,
                    'linguistic': linguistic_result.confidence,
                    'anomaly': anomaly_result.confidence,
                }
            )
            
            self.progress.emit("Analysis complete!", 100)
            self.finished.emit(result.to_dict())
            
        except Exception as e:
            logger.exception("Analysis failed")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window for Fake Voice Detection.
    """
    
    def __init__(self, backend: Optional[str] = None):
        super().__init__()
        self.backend = backend
        self._worker: Optional[AnalysisWorker] = None
        self._setup_ui()
        self._setup_statusbar()
    
    def _setup_ui(self) -> None:
        """Set up the main UI."""
        self.setWindowTitle("🎤 Fake Voice Detection")
        self.setMinimumSize(900, 700)
        
        # Apply dark theme
        self._apply_dark_theme()
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Upload and controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 500])
        layout.addWidget(splitter, 1)
    
    def _apply_dark_theme(self) -> None:
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #eee;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #aaa;
            }
            QComboBox {
                background-color: #2d2d44;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d44;
                selection-background-color: #3498db;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                background-color: #2d2d44;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #16213e;
                color: #888;
            }
        """)
    
    def _create_header(self) -> QWidget:
        """Create header widget."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-bottom: 1px solid #444;
            }
        """)
        header.setMinimumHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Logo/Title
        title = QLabel("🎤 Fake Voice Detection System")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #3498db;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Backend selector
        backend_label = QLabel("Backend:")
        backend_label.setStyleSheet("color: #888;")
        layout.addWidget(backend_label)
        
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Auto", "Apple Silicon", "CUDA", "ROCm", "CPU"])
        self.backend_combo.setCurrentText("Auto")
        layout.addWidget(self.backend_combo)
        
        return header
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with upload and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 10, 20)
        
        # File upload section
        upload_group = QGroupBox("Upload Audio")
        upload_layout = QVBoxLayout(upload_group)
        
        self.file_upload = FileUploadWidget()
        self.file_upload.file_selected.connect(self._on_file_selected)
        upload_layout.addWidget(self.file_upload)
        
        layout.addWidget(upload_group)
        
        # Analysis controls
        controls_group = QGroupBox("Analysis")
        controls_layout = QVBoxLayout(controls_group)
        
        self.analyze_btn = QPushButton("🔍 Analyze Audio")
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        self.analyze_btn.clicked.connect(self._start_analysis)
        controls_layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        controls_layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888; font-style: italic;")
        self.progress_label.setVisible(False)
        controls_layout.addWidget(self.progress_label)
        
        layout.addWidget(controls_group)
        
        # Quick tips
        tips_group = QGroupBox("Quick Tips")
        tips_layout = QVBoxLayout(tips_group)
        
        tips = [
            "• Supported formats: MP3, WAV, AAC, FLAC",
            "• For best results, use audio >= 3 seconds",
            "• High risk = likely AI-generated",
            "• Low risk = likely genuine speech"
        ]
        
        for tip in tips:
            tip_label = QLabel(tip)
            tip_label.setStyleSheet("color: #888; font-size: 12px;")
            tips_layout.addWidget(tip_label)
        
        layout.addWidget(tips_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with results."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 20, 20, 20)
        
        # Results display
        self.results_display = ResultsDisplay()
        self.results_display.export_requested.connect(self._on_export)
        layout.addWidget(self.results_display)
        
        return panel
    
    def _setup_statusbar(self) -> None:
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select an audio file to analyze")
    
    def _on_file_selected(self, file_path: str) -> None:
        """Handle file selection."""
        self.analyze_btn.setEnabled(True)
        self.status_bar.showMessage(f"Selected: {file_path}")
    
    def _start_analysis(self) -> None:
        """Start the analysis."""
        file_path = self.file_upload.get_selected_file()
        if not file_path:
            return
        
        # Get backend
        backend_map = {
            "Auto": None,
            "Apple Silicon": "MSilicon",
            "CUDA": "CUDA",
            "ROCm": "ROCm",
            "CPU": None
        }
        backend = backend_map.get(self.backend_combo.currentText())
        
        # Disable controls
        self.analyze_btn.setEnabled(False)
        self.file_upload.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        
        # Start worker
        self._worker = AnalysisWorker(file_path, backend)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        
        self.status_bar.showMessage("Analyzing...")
    
    def _on_progress(self, message: str, percentage: int) -> None:
        """Handle progress updates."""
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(message)
    
    def _on_finished(self, result: Dict[str, Any]) -> None:
        """Handle analysis completion."""
        self._reset_ui()
        self.results_display.display_result(result)
        
        risk = result.get('risk_level', 'UNKNOWN')
        self.status_bar.showMessage(f"Analysis complete - Risk Level: {risk}")
    
    def _on_error(self, error: str) -> None:
        """Handle analysis error."""
        self._reset_ui()
        
        QMessageBox.critical(self, "Analysis Error", f"Failed to analyze audio:\n{error}")
        self.status_bar.showMessage(f"Error: {error}")
    
    def _reset_ui(self) -> None:
        """Reset UI after analysis."""
        self.analyze_btn.setEnabled(True)
        self.file_upload.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
    
    def _on_export(self, file_path: str) -> None:
        """Handle export completion."""
        self.status_bar.showMessage(f"Exported to: {file_path}")


def run_gui(backend: Optional[str] = None) -> None:
    """
    Run the PyQt6 GUI application.
    
    Args:
        backend: Backend to use
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow(backend)
    window.show()
    
    sys.exit(app.exec())

