"""
Terminal interface module.

This module provides a command-line interface for fake voice detection,
including file input, progress indicators, and formatted output.
"""

from .cli import TerminalInterface, run_cli

__all__ = [
    'TerminalInterface',
    'run_cli',
]

