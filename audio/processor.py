"""
Audio Processor for loading and preprocessing audio files.

Supports:
- MP3 format
- WAV format
- AAC format

Features:
- Mono conversion
- Resampling to 16kHz
- Waveform extraction
- Spectrogram generation
- Variable-length handling with padding/truncation
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Default audio parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION = 10.0  # seconds
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 160
DEFAULT_N_MELS = 80


@dataclass
class AudioData:
    """Container for processed audio data."""
    waveform: np.ndarray
    sample_rate: int
    duration: float
    channels: int
    file_path: Optional[str] = None
    format: Optional[str] = None
    
    @property
    def num_samples(self) -> int:
        """Number of samples in the waveform."""
        return len(self.waveform)
    
    @property
    def is_mono(self) -> bool:
        """Check if audio is mono."""
        return self.channels == 1


class AudioProcessor:
    """
    Audio processor for loading and preprocessing audio files.
    
    Handles multiple audio formats and provides preprocessing utilities
    for the fake voice detection system.
    """
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
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
        self.sample_rate = sample_rate
        self.duration = duration
        self.normalize = normalize
        
    def load(self, file_path: Union[str, Path]) -> AudioData:
        """
        Load an audio file and preprocess it.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioData object containing the processed audio
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")
        
        # Load audio using librosa
        waveform, sr = self._load_audio(file_path)
        
        # Convert to mono if stereo
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)
            channels = 1
        else:
            channels = 1
        
        # Resample if necessary
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)
        
        # Normalize if requested
        if self.normalize:
            waveform = self._normalize(waveform)
        
        # Pad or truncate to target duration
        if self.duration is not None:
            waveform = self._adjust_duration(waveform, self.sample_rate, self.duration)
        
        duration = len(waveform) / self.sample_rate
        
        return AudioData(
            waveform=waveform,
            sample_rate=self.sample_rate,
            duration=duration,
            channels=channels,
            file_path=str(file_path),
            format=suffix[1:]  # Remove the dot
        )
    
    def _load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using librosa."""
        try:
            import librosa
            waveform, sr = librosa.load(str(file_path), sr=None, mono=False)
            return waveform, sr
        except Exception as e:
            logger.warning(f"librosa failed, trying soundfile: {e}")
            return self._load_with_soundfile(file_path)
    
    def _load_with_soundfile(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile as fallback."""
        try:
            import soundfile as sf
            waveform, sr = sf.read(str(file_path))
            # soundfile returns (samples, channels), we need (channels, samples)
            if waveform.ndim > 1:
                waveform = waveform.T
            return waveform, sr
        except Exception as e:
            logger.warning(f"soundfile failed, trying pydub: {e}")
            return self._load_with_pydub(file_path)
    
    def _load_with_pydub(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using pydub as fallback (supports more formats)."""
        try:
            from pydub import AudioSegment
            
            suffix = file_path.suffix.lower()[1:]  # Remove dot
            audio = AudioSegment.from_file(str(file_path), format=suffix)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Handle stereo
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).T
            
            # Normalize to float [-1, 1]
            max_val = float(2 ** (audio.sample_width * 8 - 1))
            samples = samples.astype(np.float32) / max_val
            
            return samples, audio.frame_rate
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
    
    def _resample(self, waveform: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(waveform, orig_sr=sr_orig, target_sr=sr_target)
        except ImportError:
            # Simple resampling using scipy
            from scipy import signal
            duration = len(waveform) / sr_orig
            num_samples = int(duration * sr_target)
            return signal.resample(waveform, num_samples)
    
    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            return waveform / max_val
        return waveform
    
    def _adjust_duration(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        target_duration: float
    ) -> np.ndarray:
        """Pad or truncate audio to target duration."""
        target_samples = int(target_duration * sample_rate)
        current_samples = len(waveform)
        
        if current_samples > target_samples:
            # Truncate
            return waveform[:target_samples]
        elif current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            return np.pad(waveform, (0, padding), mode='constant')
        else:
            return waveform
    
    def compute_spectrogram(
        self,
        audio_data: AudioData,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute spectrogram from audio data.
        
        Args:
            audio_data: AudioData object
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands (None for linear spectrogram)
            
        Returns:
            Spectrogram as numpy array
        """
        import librosa
        
        if n_mels is not None:
            # Mel spectrogram
            spec = librosa.feature.melspectrogram(
                y=audio_data.waveform,
                sr=audio_data.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            # Convert to dB scale
            spec = librosa.power_to_db(spec, ref=np.max)
        else:
            # Linear spectrogram
            spec = np.abs(librosa.stft(
                audio_data.waveform,
                n_fft=n_fft,
                hop_length=hop_length
            ))
            spec = librosa.amplitude_to_db(spec, ref=np.max)
        
        return spec
    
    def compute_mfcc(
        self,
        audio_data: AudioData,
        n_mfcc: int = 13,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH
    ) -> np.ndarray:
        """
        Compute MFCCs from audio data.
        
        Args:
            audio_data: AudioData object
            n_mfcc: Number of MFCCs to compute
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            MFCC features as numpy array
        """
        import librosa
        
        mfcc = librosa.feature.mfcc(
            y=audio_data.waveform,
            sr=audio_data.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        return mfcc
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an audio file without fully loading it.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio file information
        """
        file_path = Path(file_path)
        
        try:
            import librosa
            duration = librosa.get_duration(path=str(file_path))
            info = {
                'path': str(file_path),
                'format': file_path.suffix[1:],
                'duration': duration,
                'file_size': file_path.stat().st_size,
            }
            return info
        except Exception as e:
            return {
                'path': str(file_path),
                'format': file_path.suffix[1:],
                'error': str(e)
            }


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: Optional[float] = None,
    normalize: bool = True
) -> AudioData:
    """
    Convenience function to load an audio file.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate
        duration: Target duration in seconds
        normalize: Whether to normalize audio
        
    Returns:
        AudioData object
    """
    processor = AudioProcessor(
        sample_rate=sample_rate,
        duration=duration,
        normalize=normalize
    )
    return processor.load(file_path)


def preprocess_audio(
    audio_data: AudioData,
    target_duration: Optional[float] = None,
    normalize: bool = True
) -> AudioData:
    """
    Preprocess already loaded audio data.
    
    Args:
        audio_data: AudioData object to preprocess
        target_duration: Target duration in seconds
        normalize: Whether to normalize
        
    Returns:
        Preprocessed AudioData object
    """
    waveform = audio_data.waveform.copy()
    
    if normalize:
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
    
    if target_duration is not None:
        target_samples = int(target_duration * audio_data.sample_rate)
        if len(waveform) > target_samples:
            waveform = waveform[:target_samples]
        elif len(waveform) < target_samples:
            padding = target_samples - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
    
    return AudioData(
        waveform=waveform,
        sample_rate=audio_data.sample_rate,
        duration=len(waveform) / audio_data.sample_rate,
        channels=audio_data.channels,
        file_path=audio_data.file_path,
        format=audio_data.format
    )

