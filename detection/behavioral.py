"""
Behavioral Analysis for Fake Voice Detection.

Analyzes temporal and behavioral patterns in speech that differ between
human and AI-generated voices:
- Energy level variation over time
- Pause detection and micro-pause analysis
- Cadence uniformity (variance in speech rate)
- Absence of hesitation words
- Breathing patterns
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BehavioralFeatures:
    """Container for behavioral analysis features."""
    energy_variation: float
    energy_dynamics: np.ndarray
    pause_count: int
    pause_durations: List[float]
    average_pause_duration: float
    micro_pause_count: int
    speech_rate_variance: float
    cadence_uniformity: float
    breathing_detected: bool
    breathing_count: int
    hesitation_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'energy_variation': self.energy_variation,
            'pause_count': self.pause_count,
            'average_pause_duration': self.average_pause_duration,
            'micro_pause_count': self.micro_pause_count,
            'speech_rate_variance': self.speech_rate_variance,
            'cadence_uniformity': self.cadence_uniformity,
            'breathing_detected': self.breathing_detected,
            'breathing_count': self.breathing_count,
            'hesitation_score': self.hesitation_score,
        }
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.energy_variation,
            self.pause_count,
            self.average_pause_duration,
            self.micro_pause_count,
            self.speech_rate_variance,
            self.cadence_uniformity,
            float(self.breathing_detected),
            self.breathing_count,
            self.hesitation_score,
        ], dtype=np.float32)


@dataclass
class BehavioralResult:
    """Result of behavioral analysis."""
    anomaly_score: float
    confidence: str
    features: BehavioralFeatures
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_score': self.anomaly_score,
            'confidence': self.confidence,
            'features': self.features.to_dict(),
            'interpretation': self.interpretation,
        }


class BehavioralAnalyzer:
    """
    Analyzes behavioral patterns in speech for fake voice detection.
    
    Human speech exhibits natural variations in timing, energy, and patterns
    that AI-generated speech often lacks.
    """
    
    # Common hesitation words/sounds
    HESITATION_PATTERNS = [
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean',
        'well', 'so', 'basically', 'actually', 'literally'
    ]
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 160,
        energy_threshold: float = 0.01
    ):
        """
        Initialize the behavioral analyzer.
        
        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length for analysis
            hop_length: Hop length for frame analysis
            energy_threshold: Threshold for silence detection
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
    
    def analyze(
        self,
        waveform: np.ndarray,
        sample_rate: Optional[int] = None,
        transcription: Optional[str] = None
    ) -> BehavioralResult:
        """
        Perform behavioral analysis on audio.
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate (uses default if None)
            transcription: Optional text transcription for linguistic analysis
            
        Returns:
            BehavioralResult with anomaly score and features
        """
        sr = sample_rate or self.sample_rate
        
        # Extract behavioral features
        features = self._extract_features(waveform, sr, transcription)
        
        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(features)
        
        # Determine confidence
        if anomaly_score > 0.7 or anomaly_score < 0.3:
            confidence = "HIGH"
        elif anomaly_score > 0.5 or anomaly_score < 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Generate interpretation
        interpretation = self._generate_interpretation(features, anomaly_score)
        
        return BehavioralResult(
            anomaly_score=anomaly_score,
            confidence=confidence,
            features=features,
            interpretation=interpretation
        )
    
    def _extract_features(
        self,
        waveform: np.ndarray,
        sr: int,
        transcription: Optional[str]
    ) -> BehavioralFeatures:
        """Extract all behavioral features."""
        
        # Compute energy dynamics
        energy_dynamics = self._compute_energy_dynamics(waveform, sr)
        energy_variation = float(np.std(energy_dynamics))
        
        # Detect pauses
        pause_info = self._detect_pauses(energy_dynamics, sr)
        
        # Detect micro-pauses
        micro_pause_count = self._detect_micro_pauses(energy_dynamics, sr)
        
        # Compute speech rate variance
        speech_rate_variance = self._compute_speech_rate_variance(energy_dynamics, sr)
        
        # Compute cadence uniformity
        cadence_uniformity = self._compute_cadence_uniformity(energy_dynamics)
        
        # Detect breathing patterns
        breathing_detected, breathing_count = self._detect_breathing(waveform, sr)
        
        # Analyze hesitation (requires transcription)
        hesitation_score = self._analyze_hesitation(transcription) if transcription else 0.5
        
        return BehavioralFeatures(
            energy_variation=energy_variation,
            energy_dynamics=energy_dynamics,
            pause_count=pause_info['count'],
            pause_durations=pause_info['durations'],
            average_pause_duration=pause_info['average_duration'],
            micro_pause_count=micro_pause_count,
            speech_rate_variance=speech_rate_variance,
            cadence_uniformity=cadence_uniformity,
            breathing_detected=breathing_detected,
            breathing_count=breathing_count,
            hesitation_score=hesitation_score
        )
    
    def _compute_energy_dynamics(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute frame-level energy dynamics."""
        try:
            import librosa
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            return rms.flatten()
        except ImportError:
            # Fallback implementation
            num_frames = len(waveform) // self.hop_length
            energy = np.zeros(num_frames)
            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.frame_length
                if end <= len(waveform):
                    frame = waveform[start:end]
                    energy[i] = np.sqrt(np.mean(frame ** 2))
            return energy
    
    def _detect_pauses(self, energy: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect pauses in speech based on energy."""
        threshold = self.energy_threshold * np.max(energy)
        
        # Find silent regions
        is_silent = energy < threshold
        
        # Find pause boundaries
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * self.hop_length / sr
                if pause_duration > 0.1:  # Minimum 100ms pause
                    pauses.append(pause_duration)
        
        # Handle pause at end
        if in_pause:
            pause_duration = (len(energy) - pause_start) * self.hop_length / sr
            if pause_duration > 0.1:
                pauses.append(pause_duration)
        
        return {
            'count': len(pauses),
            'durations': pauses,
            'average_duration': float(np.mean(pauses)) if pauses else 0.0
        }
    
    def _detect_micro_pauses(self, energy: np.ndarray, sr: int) -> int:
        """Detect micro-pauses (very short pauses typical of natural speech)."""
        threshold = self.energy_threshold * np.max(energy)
        
        micro_pause_count = 0
        in_pause = False
        pause_start = 0
        
        for i, e in enumerate(energy):
            if e < threshold and not in_pause:
                in_pause = True
                pause_start = i
            elif e >= threshold and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * self.hop_length / sr
                # Micro-pauses are 20-100ms
                if 0.02 < pause_duration < 0.1:
                    micro_pause_count += 1
        
        return micro_pause_count
    
    def _compute_speech_rate_variance(self, energy: np.ndarray, sr: int) -> float:
        """
        Estimate variance in speech rate.
        AI-generated speech often has more uniform speech rate.
        """
        threshold = self.energy_threshold * np.max(energy)
        
        # Find speech segments
        is_speech = energy > threshold
        
        # Find segment lengths
        segment_lengths = []
        in_segment = False
        segment_start = 0
        
        for i, speaking in enumerate(is_speech):
            if speaking and not in_segment:
                in_segment = True
                segment_start = i
            elif not speaking and in_segment:
                in_segment = False
                length = (i - segment_start) * self.hop_length / sr
                if length > 0.05:  # Minimum 50ms segment
                    segment_lengths.append(length)
        
        if in_segment:
            length = (len(energy) - segment_start) * self.hop_length / sr
            if length > 0.05:
                segment_lengths.append(length)
        
        if len(segment_lengths) < 2:
            return 0.0
        
        return float(np.std(segment_lengths) / (np.mean(segment_lengths) + 1e-10))
    
    def _compute_cadence_uniformity(self, energy: np.ndarray) -> float:
        """
        Compute cadence uniformity.
        Higher uniformity (closer to 1) suggests AI-generated speech.
        """
        if len(energy) < 10:
            return 0.5
        
        # Compute local energy variations
        window_size = min(50, len(energy) // 4)
        if window_size < 5:
            return 0.5
        
        local_stds = []
        for i in range(0, len(energy) - window_size, window_size // 2):
            local_std = np.std(energy[i:i + window_size])
            local_stds.append(local_std)
        
        if len(local_stds) < 2:
            return 0.5
        
        # Uniformity = 1 - coefficient of variation of local stds
        cv = np.std(local_stds) / (np.mean(local_stds) + 1e-10)
        uniformity = 1.0 / (1.0 + cv)
        
        return float(uniformity)
    
    def _detect_breathing(self, waveform: np.ndarray, sr: int) -> Tuple[bool, int]:
        """
        Detect breathing patterns in audio.
        Natural speech typically has audible breathing patterns.
        """
        try:
            import librosa
            
            # Breathing typically occurs in 100-500 Hz range with specific patterns
            # Use bandpass filter to isolate potential breathing frequencies
            
            # Compute spectrogram
            spec = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            # Focus on breathing frequency range (100-500 Hz)
            breathing_range = (freqs >= 100) & (freqs <= 500)
            breathing_energy = np.mean(spec[breathing_range, :], axis=0)
            
            # Look for breathing-like patterns (periodic low-energy bursts)
            # Normalize
            breathing_energy = breathing_energy / (np.max(breathing_energy) + 1e-10)
            
            # Detect peaks that might be breaths
            threshold = np.percentile(breathing_energy, 90)
            peaks = breathing_energy > threshold
            
            # Count breathing events
            breathing_count = 0
            in_breath = False
            for p in peaks:
                if p and not in_breath:
                    in_breath = True
                    breathing_count += 1
                elif not p:
                    in_breath = False
            
            # Breathing is "detected" if we found reasonable number of breaths
            duration = len(waveform) / sr
            expected_breaths = duration / 4  # Roughly one breath every 4 seconds
            breathing_detected = breathing_count >= expected_breaths * 0.3
            
            return breathing_detected, breathing_count
            
        except Exception as e:
            logger.warning(f"Breathing detection failed: {e}")
            return False, 0
    
    def _analyze_hesitation(self, transcription: str) -> float:
        """
        Analyze transcription for hesitation words/sounds.
        Returns score 0-1 where 0 = no hesitations (suspicious), 1 = natural hesitations.
        """
        if not transcription:
            return 0.5  # Neutral score
        
        text = transcription.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.5
        
        hesitation_count = 0
        for pattern in self.HESITATION_PATTERNS:
            hesitation_count += text.count(pattern)
        
        # Expected hesitation rate: roughly 1-3% of words
        hesitation_rate = hesitation_count / word_count
        
        # Score: 0 if no hesitations, 1 if natural rate
        if hesitation_rate == 0:
            return 0.0  # Suspicious - no hesitations
        elif hesitation_rate < 0.01:
            return 0.3  # Few hesitations
        elif hesitation_rate < 0.03:
            return 1.0  # Natural rate
        else:
            return 0.7  # Slightly high but could be natural
    
    def _compute_anomaly_score(self, features: BehavioralFeatures) -> float:
        """
        Compute behavioral anomaly score.
        Higher score = more likely to be AI-generated.
        """
        scores = []
        
        # Low energy variation is suspicious
        # Natural speech has CV > 0.3, AI often < 0.2
        energy_score = 1.0 - min(features.energy_variation * 3, 1.0)
        scores.append(energy_score * 0.2)
        
        # Few pauses is suspicious for long audio
        pause_score = 1.0 - min(features.pause_count / 5, 1.0)
        scores.append(pause_score * 0.15)
        
        # Few micro-pauses is suspicious
        micro_pause_score = 1.0 - min(features.micro_pause_count / 10, 1.0)
        scores.append(micro_pause_score * 0.15)
        
        # High cadence uniformity is suspicious
        scores.append(features.cadence_uniformity * 0.2)
        
        # Low speech rate variance is suspicious
        rate_var_score = 1.0 - min(features.speech_rate_variance * 2, 1.0)
        scores.append(rate_var_score * 0.15)
        
        # No breathing detected is suspicious
        breathing_score = 0.0 if features.breathing_detected else 1.0
        scores.append(breathing_score * 0.1)
        
        # Low hesitation is suspicious
        hesitation_score = 1.0 - features.hesitation_score
        scores.append(hesitation_score * 0.05)
        
        return float(np.sum(scores))
    
    def _generate_interpretation(
        self,
        features: BehavioralFeatures,
        anomaly_score: float
    ) -> str:
        """Generate human-readable interpretation of results."""
        issues = []
        
        if features.energy_variation < 0.1:
            issues.append("unnaturally constant energy levels")
        
        if features.micro_pause_count < 3:
            issues.append("lack of natural micro-pauses")
        
        if features.cadence_uniformity > 0.8:
            issues.append("overly uniform speech rhythm")
        
        if not features.breathing_detected:
            issues.append("no breathing patterns detected")
        
        if features.hesitation_score < 0.3:
            issues.append("absence of natural hesitation words")
        
        if len(issues) == 0:
            return "Speech patterns appear natural"
        elif len(issues) <= 2:
            return f"Minor concerns: {', '.join(issues)}"
        else:
            return f"Multiple anomalies detected: {', '.join(issues)}"

