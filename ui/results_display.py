"""
Results Display Widget for PyQt6 GUI.

Displays detection results with:
- Overall risk score with color coding
- Breakdown by detection method
- Export functionality
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QScrollArea, QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import Dict, Any, Optional
import json


class ResultsDisplay(QWidget):
    """
    Widget for displaying fake voice detection results.
    """
    
    export_requested = pyqtSignal(str)  # Emits file path
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._result: Optional[Dict[str, Any]] = None
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Detection Results")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Overall score display
        self.score_widget = ScoreWidget()
        layout.addWidget(self.score_widget)
        
        # Component scores
        self.components_group = QGroupBox("Component Scores")
        self.components_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        self.components_layout = QVBoxLayout(self.components_group)
        layout.addWidget(self.components_group)
        
        # Interpretation
        self.interpretation_label = QLabel("")
        self.interpretation_label.setWordWrap(True)
        self.interpretation_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #2d2d2d;
                border-radius: 5px;
                color: #ddd;
            }
        """)
        layout.addWidget(self.interpretation_label)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(self._export_json)
        self.export_json_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        export_layout.addWidget(self.export_json_btn)
        
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #7d3c98;
            }
        """)
        export_layout.addWidget(self.export_csv_btn)
        
        layout.addLayout(export_layout)
        
        # Initially hide results
        self._set_visible(False)
    
    def _set_visible(self, visible: bool) -> None:
        """Set visibility of result components."""
        self.score_widget.setVisible(visible)
        self.components_group.setVisible(visible)
        self.interpretation_label.setVisible(visible)
        self.export_json_btn.setVisible(visible)
        self.export_csv_btn.setVisible(visible)
    
    def display_result(self, result: Dict[str, Any]) -> None:
        """
        Display detection result.
        
        Args:
            result: Detection result dictionary
        """
        self._result = result
        
        if 'error' in result:
            self._show_error(result['error'])
            return
        
        self._set_visible(True)
        
        # Update overall score
        score = result.get('overall_score', 0)
        risk = result.get('risk_level', 'UNKNOWN')
        confidence = result.get('confidence', 'UNKNOWN')
        
        self.score_widget.set_score(score, risk, confidence)
        
        # Update component scores
        self._clear_components()
        if 'component_scores' in result:
            for name, score in result['component_scores'].items():
                bar = ComponentBar(name.replace('_', ' ').title(), score)
                self.components_layout.addWidget(bar)
        
        # Update interpretation
        interpretation = result.get('interpretation', '')
        self.interpretation_label.setText(f"📋 {interpretation}")
    
    def _clear_components(self) -> None:
        """Clear component score widgets."""
        while self.components_layout.count():
            child = self.components_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _show_error(self, error: str) -> None:
        """Show error message."""
        self._set_visible(True)
        self.score_widget.set_error(error)
        self.components_group.setVisible(False)
        self.interpretation_label.setVisible(False)
    
    def _export_json(self) -> None:
        """Export results as JSON."""
        if not self._result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", "results.json", "JSON Files (*.json)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self._result, f, indent=2)
            self.export_requested.emit(file_path)
    
    def _export_csv(self) -> None:
        """Export results as CSV."""
        if not self._result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "results.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Overall Score', self._result.get('overall_score', '')])
                writer.writerow(['Risk Level', self._result.get('risk_level', '')])
                writer.writerow(['Confidence', self._result.get('confidence', '')])
                
                if 'component_scores' in self._result:
                    for name, score in self._result['component_scores'].items():
                        writer.writerow([name, score])
            
            self.export_requested.emit(file_path)
    
    def clear(self) -> None:
        """Clear the display."""
        self._result = None
        self._set_visible(False)
        self._clear_components()


class ScoreWidget(QFrame):
    """
    Widget displaying the overall score with visual indicator.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI."""
        self.setMinimumHeight(120)
        self.setStyleSheet("""
            ScoreWidget {
                background-color: #2d2d2d;
                border-radius: 10px;
                border: 1px solid #444;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Score label
        self.score_label = QLabel("0%")
        self.score_label.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)
        
        # Risk label
        self.risk_label = QLabel("Risk Level")
        self.risk_label.setFont(QFont("Arial", 14))
        self.risk_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.risk_label)
        
        # Confidence label
        self.confidence_label = QLabel("Confidence: ---")
        self.confidence_label.setStyleSheet("color: #888;")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)
    
    def set_score(self, score: float, risk: str, confidence: str) -> None:
        """Set the score display."""
        self.score_label.setText(f"{score:.0%}")
        
        # Color based on risk
        if risk == "LOW":
            color = "#2ecc71"
        elif risk == "MEDIUM":
            color = "#f39c12"
        else:
            color = "#e74c3c"
        
        self.score_label.setStyleSheet(f"color: {color};")
        self.risk_label.setText(f"Risk: {risk}")
        self.risk_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.confidence_label.setText(f"Confidence: {confidence}")
    
    def set_error(self, error: str) -> None:
        """Show error state."""
        self.score_label.setText("❌")
        self.score_label.setStyleSheet("color: #e74c3c;")
        self.risk_label.setText(f"Error: {error[:50]}...")
        self.risk_label.setStyleSheet("color: #e74c3c;")
        self.confidence_label.setText("")


class ComponentBar(QFrame):
    """
    Progress bar for component score display.
    """
    
    def __init__(self, name: str, score: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui(name, score)
    
    def _setup_ui(self, name: str, score: float) -> None:
        """Set up the component bar."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Name label
        name_label = QLabel(name)
        name_label.setMinimumWidth(120)
        name_label.setStyleSheet("color: #ddd;")
        layout.addWidget(name_label)
        
        # Progress bar
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setValue(int(score * 100))
        progress.setTextVisible(False)
        progress.setMinimumHeight(20)
        
        # Color based on score
        if score < 0.3:
            color = "#2ecc71"
        elif score < 0.6:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 5px;
                background-color: #1a1a1a;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(progress, 1)
        
        # Score label
        score_label = QLabel(f"{score:.0%}")
        score_label.setMinimumWidth(50)
        score_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        score_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(score_label)

