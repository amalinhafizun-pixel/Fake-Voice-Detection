"""
Audio processing module.

This module handles audio file loading, preprocessing, and feature extraction
for mp3, wav, and aac audio formats.
"""

from .processor import AudioProcessor, load_audio, preprocess_audio

__all__ = [
    'AudioProcessor',
    'load_audio',
    'preprocess_audio',
]

