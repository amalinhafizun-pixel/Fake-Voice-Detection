"""
Signal-Level Feature Extraction for Fake Voice Detection.

Extracts acoustic features that are commonly abnormal in AI-generated speech:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral Flux
- Pitch Jitter & Shimmer
- Harmonic-to-Noise Ratio (HNR)
- Phase Coherence
"""

import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalFeatures:
    """Container for extracted signal features."""
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    spectral_flux: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    pitch: np.ndarray
    jitter: float
    shimmer: float
    hnr: float
    phase_coherence: float
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return {
            'mfcc_mean': np.mean(self.mfcc, axis=1).tolist(),
            'mfcc_std': np.std(self.mfcc, axis=1).tolist(),
            'mfcc_delta_mean': np.mean(self.mfcc_delta, axis=1).tolist(),
            'spectral_flux_mean': float(np.mean(self.spectral_flux)),
            'spectral_flux_std': float(np.std(self.spectral_flux)),
            'spectral_centroid_mean': float(np.mean(self.spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(self.spectral_rolloff)),
            'pitch_mean': float(np.mean(self.pitch[self.pitch > 0])) if np.any(self.pitch > 0) else 0.0,
            'pitch_std': float(np.std(self.pitch[self.pitch > 0])) if np.any(self.pitch > 0) else 0.0,
            'jitter': self.jitter,
            'shimmer': self.shimmer,
            'hnr': self.hnr,
            'phase_coherence': self.phase_coherence,
            'zcr_mean': float(np.mean(self.zero_crossing_rate)),
            'rms_mean': float(np.mean(self.rms_energy)),
            'rms_std': float(np.std(self.rms_energy)),
        }
    
    def to_vector(self) -> np.ndarray:
        """Convert features to a single feature vector."""
        features = []
        
        # MFCC statistics
        features.extend(np.mean(self.mfcc, axis=1))
        features.extend(np.std(self.mfcc, axis=1))
        features.extend(np.mean(self.mfcc_delta, axis=1))
        features.extend(np.mean(self.mfcc_delta2, axis=1))
        
        # Spectral features
        features.append(np.mean(self.spectral_flux))
        features.append(np.std(self.spectral_flux))
        features.append(np.mean(self.spectral_centroid))
        features.append(np.std(self.spectral_centroid))
        features.append(np.mean(self.spectral_rolloff))
        
        # Pitch features
        valid_pitch = self.pitch[self.pitch > 0]
        if len(valid_pitch) > 0:
            features.append(np.mean(valid_pitch))
            features.append(np.std(valid_pitch))
        else:
            features.extend([0.0, 0.0])
        
        # Voice quality
        features.append(self.jitter)
        features.append(self.shimmer)
        features.append(self.hnr)
        features.append(self.phase_coherence)
        
        # Energy features
        features.append(np.mean(self.zero_crossing_rate))
        features.append(np.mean(self.rms_energy))
        features.append(np.std(self.rms_energy))
        
        return np.array(features, dtype=np.float32)


class SignalFeatureExtractor:
    """
    Extracts signal-level audio features for fake voice detection.
    
    These features capture acoustic artifacts commonly present in
    AI-generated or cloned voices.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract(self, waveform: np.ndarray, sample_rate: Optional[int] = None) -> SignalFeatures:
        """
        Extract all signal features from audio waveform.
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate (uses default if None)
            
        Returns:
            SignalFeatures object containing all extracted features
        """
        sr = sample_rate or self.sample_rate
        
        # Extract MFCCs and deltas
        mfcc = self._extract_mfcc(waveform, sr)
        mfcc_delta = self._compute_delta(mfcc)
        mfcc_delta2 = self._compute_delta(mfcc_delta)
        
        # Extract spectral features
        spectral_flux = self._compute_spectral_flux(waveform, sr)
        spectral_centroid = self._compute_spectral_centroid(waveform, sr)
        spectral_rolloff = self._compute_spectral_rolloff(waveform, sr)
        
        # Extract pitch and voice quality
        pitch = self._extract_pitch(waveform, sr)
        jitter = self._compute_jitter(pitch)
        shimmer = self._compute_shimmer(waveform, pitch, sr)
        
        # Extract HNR and phase coherence
        hnr = self._compute_hnr(waveform, sr)
        phase_coherence = self._compute_phase_coherence(waveform, sr)
        
        # Extract energy features
        zcr = self._compute_zero_crossing_rate(waveform)
        rms = self._compute_rms_energy(waveform)
        
        return SignalFeatures(
            mfcc=mfcc,
            mfcc_delta=mfcc_delta,
            mfcc_delta2=mfcc_delta2,
            spectral_flux=spectral_flux,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            pitch=pitch,
            jitter=jitter,
            shimmer=shimmer,
            hnr=hnr,
            phase_coherence=phase_coherence,
            zero_crossing_rate=zcr,
            rms_energy=rms
        )
    
    def _extract_mfcc(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features."""
        try:
            import librosa
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return mfcc
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            return np.zeros((self.n_mfcc, 1))
    
    def _compute_delta(self, features: np.ndarray, width: int = 9) -> np.ndarray:
        """Compute delta (derivative) of features."""
        try:
            import librosa
            return librosa.feature.delta(features, width=width)
        except Exception as e:
            logger.warning(f"Delta computation failed: {e}")
            return np.zeros_like(features)
    
    def _compute_spectral_flux(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute spectral flux - measures rate of change in the power spectrum.
        AI-generated speech often has abnormally low spectral flux.
        """
        try:
            import librosa
            spec = np.abs(librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length))
            
            # Compute frame-to-frame difference
            diff = np.diff(spec, axis=1)
            
            # Sum of squared differences (half-wave rectified)
            flux = np.sum(np.maximum(0, diff) ** 2, axis=0)
            
            # Normalize
            flux = flux / (np.max(flux) + 1e-10)
            
            return flux
        except Exception as e:
            logger.error(f"Spectral flux computation failed: {e}")
            return np.zeros(1)
    
    def _compute_spectral_centroid(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectral centroid."""
        try:
            import librosa
            centroid = librosa.feature.spectral_centroid(
                y=waveform,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return centroid.flatten()
        except Exception as e:
            logger.error(f"Spectral centroid computation failed: {e}")
            return np.zeros(1)
    
    def _compute_spectral_rolloff(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectral rolloff."""
        try:
            import librosa
            rolloff = librosa.feature.spectral_rolloff(
                y=waveform,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return rolloff.flatten()
        except Exception as e:
            logger.error(f"Spectral rolloff computation failed: {e}")
            return np.zeros(1)
    
    def _extract_pitch(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch (F0) contour."""
        try:
            import librosa
            
            # Use librosa's pyin for pitch extraction
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                hop_length=self.hop_length
            )
            
            # Replace NaN with 0
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
            
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return np.zeros(1)
    
    def _compute_jitter(self, pitch: np.ndarray) -> float:
        """
        Compute jitter - cycle-to-cycle variation in pitch.
        AI-generated voices often have unnaturally low jitter.
        """
        try:
            # Get voiced frames only
            voiced_pitch = pitch[pitch > 0]
            
            if len(voiced_pitch) < 2:
                return 0.0
            
            # Compute periods from pitch
            periods = 1.0 / voiced_pitch
            
            # Compute absolute jitter (average absolute difference)
            diffs = np.abs(np.diff(periods))
            jitter = np.mean(diffs) / np.mean(periods)
            
            return float(jitter)
            
        except Exception as e:
            logger.warning(f"Jitter computation failed: {e}")
            return 0.0
    
    def _compute_shimmer(self, waveform: np.ndarray, pitch: np.ndarray, sr: int) -> float:
        """
        Compute shimmer - cycle-to-cycle variation in amplitude.
        AI-generated voices often have unnaturally low shimmer.
        """
        try:
            # Simple shimmer approximation using RMS energy variation
            import librosa
            
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            ).flatten()
            
            if len(rms) < 2:
                return 0.0
            
            # Compute amplitude variation
            diffs = np.abs(np.diff(rms))
            shimmer = np.mean(diffs) / (np.mean(rms) + 1e-10)
            
            return float(shimmer)
            
        except Exception as e:
            logger.warning(f"Shimmer computation failed: {e}")
            return 0.0
    
    def _compute_hnr(self, waveform: np.ndarray, sr: int) -> float:
        """
        Compute Harmonic-to-Noise Ratio.
        AI-generated audio is often unnaturally "clean" with high HNR.
        """
        try:
            import librosa
            
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(waveform)
            
            # Compute power of harmonic and noise (percussive + residual)
            harmonic_power = np.mean(harmonic ** 2)
            noise_power = np.mean((waveform - harmonic) ** 2)
            
            if noise_power < 1e-10:
                return 100.0  # Very high HNR
            
            hnr = 10 * np.log10(harmonic_power / noise_power)
            return float(hnr)
            
        except Exception as e:
            logger.warning(f"HNR computation failed: {e}")
            return 0.0
    
    def _compute_phase_coherence(self, waveform: np.ndarray, sr: int) -> float:
        """
        Compute phase coherence metric.
        GAN/TTS models often produce audio with inconsistent phase relationships.
        """
        try:
            import librosa
            
            # Compute STFT
            stft = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Get phase
            phase = np.angle(stft)
            
            # Compute phase derivative (instantaneous frequency)
            phase_diff = np.diff(phase, axis=1)
            
            # Unwrap phase differences
            phase_diff = np.unwrap(phase_diff, axis=1)
            
            # Compute variance of phase differences across frequency bins
            phase_variance = np.var(phase_diff, axis=0)
            
            # Average coherence (inverse of variance)
            coherence = 1.0 / (1.0 + np.mean(phase_variance))
            
            return float(coherence)
            
        except Exception as e:
            logger.warning(f"Phase coherence computation failed: {e}")
            return 0.5
    
    def _compute_zero_crossing_rate(self, waveform: np.ndarray) -> np.ndarray:
        """Compute zero crossing rate."""
        try:
            import librosa
            zcr = librosa.feature.zero_crossing_rate(
                waveform,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            return zcr.flatten()
        except Exception as e:
            logger.warning(f"ZCR computation failed: {e}")
            return np.zeros(1)
    
    def _compute_rms_energy(self, waveform: np.ndarray) -> np.ndarray:
        """Compute RMS energy."""
        try:
            import librosa
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            return rms.flatten()
        except Exception as e:
            logger.warning(f"RMS computation failed: {e}")
            return np.zeros(1)
    
    def compute_anomaly_score(self, features: SignalFeatures) -> float:
        """
        Compute an anomaly score based on signal features.
        
        Higher scores indicate more likely to be AI-generated.
        
        Args:
            features: Extracted SignalFeatures
            
        Returns:
            Anomaly score between 0 and 1
        """
        scores = []
        
        # Low jitter indicates synthetic voice (robotic voices are too perfect)
        # Normal jitter: 0.5-2%, robotic: <0.3%
        jitter_threshold = 0.003  # 0.3%
        if features.jitter < jitter_threshold:
            jitter_score = 1.0 - (features.jitter / jitter_threshold)  # Perfect = 1.0
        else:
            jitter_score = max(0.0, 1.0 - (features.jitter - jitter_threshold) * 10)
        scores.append(jitter_score * 0.25)
        
        # Low shimmer indicates synthetic voice
        # Normal shimmer: 2-5%, robotic: <1%
        shimmer_threshold = 0.01  # 1%
        if features.shimmer < shimmer_threshold:
            shimmer_score = 1.0 - (features.shimmer / shimmer_threshold)
        else:
            shimmer_score = max(0.0, 1.0 - (features.shimmer - shimmer_threshold) * 5)
        scores.append(shimmer_score * 0.25)
        
        # Very high HNR indicates synthetic voice (too clean)
        # Normal HNR: 10-20dB, robotic: >25dB
        if features.hnr > 25:
            hnr_score = 1.0  # Very high HNR = definitely synthetic
        elif features.hnr > 20:
            hnr_score = 0.5 + (features.hnr - 20) / 10  # 0.5-1.0
        else:
            hnr_score = max(0.0, (features.hnr - 10) / 20)  # 0-0.5
        scores.append(hnr_score * 0.2)
        
        # Low spectral flux variation indicates synthetic (too uniform)
        flux_std = float(np.std(features.spectral_flux))
        # Normal variation: 0.1-0.5, robotic: <0.05
        if flux_std < 0.05:
            flux_score = 1.0
        elif flux_std < 0.1:
            flux_score = 0.5 + (0.1 - flux_std) / 0.1  # 0.5-1.0
        else:
            flux_score = max(0.0, 0.5 - (flux_std - 0.1) * 1.25)  # 0-0.5
        scores.append(flux_score * 0.15)
        
        # Phase coherence (too coherent is suspicious)
        # Robotic voices have very high phase coherence (>0.9)
        if features.phase_coherence > 0.9:
            phase_score = 1.0
        elif features.phase_coherence > 0.7:
            phase_score = 0.5 + (features.phase_coherence - 0.7) / 0.4  # 0.5-1.0
        else:
            phase_score = features.phase_coherence / 0.7  # 0-0.5
        scores.append(phase_score * 0.15)
        
        final_score = float(np.sum(scores))
        # Ensure score is in [0, 1] range
        return min(1.0, max(0.0, final_score))

