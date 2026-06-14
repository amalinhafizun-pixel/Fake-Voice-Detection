"""
Linguistic Analysis for Fake Voice Detection using Whisper.

Analyzes transcribed speech for linguistic patterns that differ between
human and AI-generated speech:
- Perplexity score
- Token repetition rate
- Semantic entropy
- Over-formal grammar detection
- Disfluency detection
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LinguisticFeatures:
    """Container for linguistic analysis features."""
    transcription: str
    word_count: int
    unique_words: int
    vocabulary_richness: float
    perplexity_score: float
    repetition_rate: float
    semantic_entropy: float
    formality_score: float
    disfluency_count: int
    disfluency_rate: float
    sentence_length_variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transcription': self.transcription,
            'word_count': self.word_count,
            'unique_words': self.unique_words,
            'vocabulary_richness': self.vocabulary_richness,
            'perplexity_score': self.perplexity_score,
            'repetition_rate': self.repetition_rate,
            'semantic_entropy': self.semantic_entropy,
            'formality_score': self.formality_score,
            'disfluency_count': self.disfluency_count,
            'disfluency_rate': self.disfluency_rate,
            'sentence_length_variance': self.sentence_length_variance,
        }
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.vocabulary_richness,
            self.perplexity_score,
            self.repetition_rate,
            self.semantic_entropy,
            self.formality_score,
            self.disfluency_rate,
            self.sentence_length_variance,
        ], dtype=np.float32)


@dataclass
class LinguisticResult:
    """Result of linguistic analysis."""
    anomaly_score: float
    confidence: str
    features: LinguisticFeatures
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_score': self.anomaly_score,
            'confidence': self.confidence,
            'features': self.features.to_dict(),
            'interpretation': self.interpretation,
        }


class LinguisticAnalyzer:
    """
    Analyzes linguistic patterns in transcribed speech for fake voice detection.
    
    Uses OpenAI Whisper for transcription and NLP techniques for analysis.
    """
    
    # Disfluency patterns (hesitation sounds/words)
    DISFLUENCIES = [
        r'\bum+\b', r'\buh+\b', r'\ber+\b', r'\bah+\b',
        r'\blike\b', r'\byou know\b', r'\bi mean\b',
        r'\bwell\b', r'\bso\b', r'\bbasically\b',
        r'\bactually\b', r'\bliterally\b', r'\bkinda\b',
        r'\bsorta\b', r'\bgonna\b', r'\bwanna\b'
    ]
    
    # Formal word patterns (indicate potentially AI-generated text)
    FORMAL_PATTERNS = [
        r'\bfurthermore\b', r'\bmoreover\b', r'\bnevertheless\b',
        r'\bconsequently\b', r'\bsubsequently\b', r'\btherefore\b',
        r'\bhence\b', r'\bthus\b', r'\bwhereby\b',
        r'\bnotwithstanding\b', r'\binasmuch\b', r'\bheretofore\b'
    ]
    
    def __init__(
        self,
        whisper_model: str = "tiny",
        device: Optional[str] = None
    ):
        """
        Initialize the linguistic analyzer.
        
        Args:
            whisper_model: Whisper model size (default: 'large-v2' for best accuracy)
            device: Device for inference ('cuda', 'cpu', etc.)
        """
        self.whisper_model_name = whisper_model
        self.device = device
        self.whisper_model = None
        self._model_loaded = False
    
    def _load_whisper(self) -> bool:
        """Load Whisper model on first use."""
        if self._model_loaded:
            return True
        
        try:
            import whisper
            import torch
            
            # Auto-detect device if not specified
            if self.device:
                device = self.device
            else:
                # Whisper has limited MPS support, so prefer CUDA then CPU
                # MPS can cause sparse tensor errors, so we'll use CPU for stability
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"  # Use CPU even on MPS for Whisper stability
            
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=device)
            self._model_loaded = True
            logger.info(f"Whisper model loaded: {self.whisper_model_name} on {device}")
            return True
            
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return False
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def transcribe(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        if not self._load_whisper():
            return ""
        
        try:
            # Whisper expects 16kHz audio
            if sample_rate != 16000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            
            # Ensure float32
            waveform = waveform.astype(np.float32)
            
            # Transcribe
            result = self.whisper_model.transcribe(waveform)
            return result.get('text', '').strip()
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def analyze(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        transcription: Optional[str] = None
    ) -> LinguisticResult:
        """
        Perform linguistic analysis on audio.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            transcription: Optional pre-computed transcription
            
        Returns:
            LinguisticResult with anomaly score and features
        """
        # Get transcription if not provided
        if transcription is None:
            transcription = self.transcribe(waveform, sample_rate)
        
        if not transcription:
            # Return neutral result if transcription failed
            return LinguisticResult(
                anomaly_score=0.5,
                confidence="LOW",
                features=LinguisticFeatures(
                    transcription="",
                    word_count=0,
                    unique_words=0,
                    vocabulary_richness=0.0,
                    perplexity_score=0.5,
                    repetition_rate=0.0,
                    semantic_entropy=0.5,
                    formality_score=0.5,
                    disfluency_count=0,
                    disfluency_rate=0.0,
                    sentence_length_variance=0.0
                ),
                interpretation="Could not analyze - transcription failed"
            )
        
        # Extract features
        features = self._extract_features(transcription)
        
        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(features)
        
        # Determine confidence
        if features.word_count < 20:
            confidence = "LOW"
        elif anomaly_score > 0.7 or anomaly_score < 0.3:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"
        
        # Generate interpretation
        interpretation = self._generate_interpretation(features, anomaly_score)
        
        return LinguisticResult(
            anomaly_score=anomaly_score,
            confidence=confidence,
            features=features,
            interpretation=interpretation
        )
    
    def _extract_features(self, transcription: str) -> LinguisticFeatures:
        """Extract linguistic features from transcription."""
        text = transcription.lower()
        words = self._tokenize(text)
        word_count = len(words)
        
        if word_count == 0:
            return LinguisticFeatures(
                transcription=transcription,
                word_count=0,
                unique_words=0,
                vocabulary_richness=0.0,
                perplexity_score=0.5,
                repetition_rate=0.0,
                semantic_entropy=0.5,
                formality_score=0.5,
                disfluency_count=0,
                disfluency_rate=0.0,
                sentence_length_variance=0.0
            )
        
        # Basic statistics
        unique_words = len(set(words))
        vocabulary_richness = unique_words / word_count
        
        # Compute features
        perplexity_score = self._estimate_perplexity(words)
        repetition_rate = self._compute_repetition_rate(words)
        semantic_entropy = self._compute_semantic_entropy(words)
        formality_score = self._compute_formality_score(text)
        disfluency_count = self._count_disfluencies(text)
        disfluency_rate = disfluency_count / word_count
        sentence_length_variance = self._compute_sentence_variance(transcription)
        
        return LinguisticFeatures(
            transcription=transcription,
            word_count=word_count,
            unique_words=unique_words,
            vocabulary_richness=vocabulary_richness,
            perplexity_score=perplexity_score,
            repetition_rate=repetition_rate,
            semantic_entropy=semantic_entropy,
            formality_score=formality_score,
            disfluency_count=disfluency_count,
            disfluency_rate=disfluency_rate,
            sentence_length_variance=sentence_length_variance
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _estimate_perplexity(self, words: List[str]) -> float:
        """
        Estimate perplexity using simple n-gram model.
        AI-generated text often has lower perplexity (more predictable).
        """
        if len(words) < 3:
            return 0.5  # Neutral
        
        # Build bigram probabilities
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        word_counts = Counter(words)
        
        # Compute log probability
        log_prob = 0
        for w1, w2 in bigrams:
            # Smoothed probability
            prob = (bigram_counts[(w1, w2)] + 1) / (word_counts[w1] + len(word_counts))
            log_prob += np.log(prob + 1e-10)
        
        # Perplexity
        perplexity = np.exp(-log_prob / len(bigrams))
        
        # Normalize to 0-1 (lower perplexity = lower score = more suspicious)
        normalized = min(perplexity / 100, 1.0)
        return float(normalized)
    
    def _compute_repetition_rate(self, words: List[str]) -> float:
        """
        Compute token repetition rate.
        AI text often has higher repetition of certain phrases.
        """
        if len(words) < 2:
            return 0.0
        
        # Count repeated consecutive words
        repetitions = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
        
        # Count repeated bigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        
        total_repetition = repetitions + repeated_bigrams
        rate = total_repetition / len(words)
        
        return float(min(rate, 1.0))
    
    def _compute_semantic_entropy(self, words: List[str]) -> float:
        """
        Compute semantic entropy (word distribution entropy).
        Low entropy suggests limited vocabulary usage.
        """
        if len(words) < 5:
            return 0.5
        
        word_counts = Counter(words)
        total = len(words)
        
        # Compute entropy
        entropy = 0
        for count in word_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob + 1e-10)
        
        # Normalize by max possible entropy
        max_entropy = np.log2(len(word_counts))
        normalized = entropy / (max_entropy + 1e-10)
        
        return float(normalized)
    
    def _compute_formality_score(self, text: str) -> float:
        """
        Compute formality score.
        AI-generated text is often overly formal.
        """
        formal_count = 0
        for pattern in self.FORMAL_PATTERNS:
            formal_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.5
        
        # High formality rate is suspicious
        formality_rate = formal_count / word_count
        
        # Normalize (>1% formal words is considered high)
        return float(min(formality_rate * 100, 1.0))
    
    def _count_disfluencies(self, text: str) -> int:
        """Count disfluencies (hesitation words/sounds)."""
        count = 0
        for pattern in self.DISFLUENCIES:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def _compute_sentence_variance(self, text: str) -> float:
        """
        Compute variance in sentence lengths.
        Natural speech has more variable sentence lengths.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Compute word counts per sentence
        lengths = [len(s.split()) for s in sentences]
        
        # Coefficient of variation
        mean_length = np.mean(lengths)
        if mean_length == 0:
            return 0.0
        
        cv = np.std(lengths) / mean_length
        
        return float(min(cv, 1.0))
    
    def _compute_anomaly_score(self, features: LinguisticFeatures) -> float:
        """
        Compute linguistic anomaly score.
        Higher score = more likely to be AI-generated.
        """
        scores = []
        
        # Low perplexity is suspicious (too predictable)
        perplexity_score = 1.0 - features.perplexity_score
        scores.append(perplexity_score * 0.2)
        
        # High repetition is suspicious
        scores.append(features.repetition_rate * 0.15)
        
        # Low semantic entropy is suspicious
        entropy_score = 1.0 - features.semantic_entropy
        scores.append(entropy_score * 0.15)
        
        # High formality is suspicious
        scores.append(features.formality_score * 0.2)
        
        # Low disfluency rate is suspicious
        disfluency_score = 1.0 - min(features.disfluency_rate * 20, 1.0)
        scores.append(disfluency_score * 0.2)
        
        # Low sentence variance is suspicious
        variance_score = 1.0 - features.sentence_length_variance
        scores.append(variance_score * 0.1)
        
        return float(np.sum(scores))
    
    def _generate_interpretation(
        self,
        features: LinguisticFeatures,
        anomaly_score: float
    ) -> str:
        """Generate human-readable interpretation."""
        issues = []
        
        if features.perplexity_score < 0.3:
            issues.append("overly predictable word patterns")
        
        if features.repetition_rate > 0.1:
            issues.append("high phrase repetition")
        
        if features.semantic_entropy < 0.5:
            issues.append("limited vocabulary usage")
        
        if features.formality_score > 0.5:
            issues.append("unusually formal language")
        
        if features.disfluency_rate < 0.01 and features.word_count > 50:
            issues.append("absence of natural speech disfluencies")
        
        if features.sentence_length_variance < 0.2:
            issues.append("uniform sentence structure")
        
        if len(issues) == 0:
            return "Linguistic patterns appear natural"
        elif len(issues) <= 2:
            return f"Minor linguistic concerns: {', '.join(issues)}"
        else:
            return f"Multiple linguistic anomalies: {', '.join(issues)}"

