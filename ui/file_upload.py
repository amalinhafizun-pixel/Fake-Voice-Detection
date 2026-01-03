"""
File Upload Widget for PyQt6 GUI.

Provides drag-and-drop and button-based file upload functionality.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path
from typing import Optional, List


class FileUploadWidget(QWidget):
    """
    Widget for uploading audio files with drag-and-drop support.
    """
    
    # Signal emitted when a file is selected
    file_selected = pyqtSignal(str)
    
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg']
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_file: Optional[str] = None
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Drop zone
        self.drop_zone = DropZone(self)
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self.drop_zone)
        
        # Or divider
        divider_layout = QHBoxLayout()
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setStyleSheet("color: #555;")
        
        or_label = QLabel("or")
        or_label.setStyleSheet("color: #888; padding: 0 10px;")
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #555;")
        
        divider_layout.addWidget(line1)
        divider_layout.addWidget(or_label)
        divider_layout.addWidget(line2)
        layout.addLayout(divider_layout)
        
        # Browse button
        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setMinimumHeight(40)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.browse_btn.clicked.connect(self._browse_file)
        layout.addWidget(self.browse_btn)
        
        # Selected file label
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("color: #888; font-style: italic; margin-top: 10px;")
        layout.addWidget(self.file_label)
    
    def _browse_file(self) -> None:
        """Open file browser dialog."""
        formats = " ".join([f"*{ext}" for ext in self.SUPPORTED_FORMATS])
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            f"Audio Files ({formats})"
        )
        
        if file_path:
            self._set_file(file_path)
    
    def _on_file_dropped(self, file_path: str) -> None:
        """Handle file dropped on drop zone."""
        self._set_file(file_path)
    
    def _set_file(self, file_path: str) -> None:
        """Set the selected file."""
        path = Path(file_path)
        
        if not path.exists():
            self.file_label.setText("File not found")
            self.file_label.setStyleSheet("color: #e74c3c;")
            return
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            self.file_label.setText(f"Unsupported format: {path.suffix}")
            self.file_label.setStyleSheet("color: #e74c3c;")
            return
        
        self._selected_file = str(path)
        self.file_label.setText(f"Selected: {path.name}")
        self.file_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        
        self.file_selected.emit(self._selected_file)
    
    def get_selected_file(self) -> Optional[str]:
        """Get the currently selected file path."""
        return self._selected_file
    
    def clear(self) -> None:
        """Clear the selection."""
        self._selected_file = None
        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet("color: #888; font-style: italic;")


class DropZone(QFrame):
    """
    Drag-and-drop zone for file upload.
    """
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the drop zone UI."""
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #555;
                border-radius: 10px;
                background-color: #2d2d2d;
            }
            DropZone:hover {
                border-color: #3498db;
                background-color: #333;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon
        icon_label = QLabel("🎵")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)
        
        # Text
        text_label = QLabel("Drag and drop audio file here")
        text_label.setStyleSheet("color: #888; font-size: 14px;")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text_label)
        
        # Supported formats
        formats_label = QLabel("MP3, WAV, AAC, M4A, FLAC, OGG")
        formats_label.setStyleSheet("color: #666; font-size: 11px;")
        formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(formats_label)
    
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropZone {
                    border: 2px dashed #2ecc71;
                    border-radius: 10px;
                    background-color: #2a3a2a;
                }
            """)
    
    def dragLeaveEvent(self, event) -> None:
        """Handle drag leave event."""
        self._setup_ui()
    
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop event."""
        self._setup_ui()
        
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)

