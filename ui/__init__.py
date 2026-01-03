"""
UI module for PyQt6 graphical interface.

This module contains the main window, file upload widget, and results display
components for the fake voice detection application.
"""

from .main_window import MainWindow
from .file_upload import FileUploadWidget
from .results_display import ResultsDisplay

__all__ = [
    'MainWindow',
    'FileUploadWidget',
    'ResultsDisplay',
]

