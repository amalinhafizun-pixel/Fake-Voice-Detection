# Fake Voice (AI-Generated Speech) Detection

This document outlines practical and research-backed methods to detect **fake voices**, including **AI-generated speech, voice cloning, and replay attacks**. The approach emphasizes **multi-layer detection**, suitable for systems such as **online examination integrity, surveillance, and authentication platforms**.

---

## 1. Audio Signal–Level Detection

These methods analyze **low-level acoustic artifacts** commonly left by text-to-speech (TTS) and voice cloning models.

### Key Features

| Feature | Description |
|------|------------|
| MFCCs | Fake voices show abnormal MFCC distributions |
| Spectral Flux | AI speech lacks natural spectral variation |
| Pitch Jitter & Shimmer | Synthesized voices are overly stable |
| Harmonic-to-Noise Ratio | AI audio is unnaturally clean |
| Phase Coherence | GAN/TTS models often break phase realism |

**Libraries**:
- `librosa`
- `pyAudioAnalysis`
- `torchaudio`

**Pros**: Lightweight, explainable  
**Cons**: Can be bypassed by advanced models

---

## 2. Deep Learning–Based Spoof Detection

This is the **most effective and widely used approach**.

### Popular Models

| Model | Notes |
|----|------|
| AASIST | State-of-the-art (ASVspoof 2021 winner) |
| RawNet2 | Operates directly on raw waveform |
| LFCC-CNN | Lightweight CNN-based detector |
| ECAPA-TDNN | Strong embeddings for anomaly detection |

### Training Datasets
- ASVspoof 2019 / 2021
- WaveFake
- Fake-or-Real (FoR)
- LibriSpeech + TTS-generated speech

**Typical Output**:
```json
{
  "spoof_probability": 0.87,
  "confidence": "HIGH"
}
```

---

## 3. Temporal & Behavioral Analysis

Analyzes **how speech behaves over time**, rather than just sound quality.

### Behavioral Indicators
- Absence of breathing patterns
- Uniform cadence and pacing
- No hesitation words ("um", "uh")
- Constant energy levels
- Lack of emotional variation

**Observation**:
- Human speech → chaotic micro-pauses
- AI speech → consistent, uniform timing

---

## 4. Liveness Detection (Anti-Replay & Anti-TTS)

Ensures the speaker is **present and responding in real time**.

### Challenge–Response Techniques
- Repeat randomly generated sentences
- Read numbers in reverse order
- Answer context-dependent questions

**Example Challenge**:
> "Say the third word of the previous sentence backwards"

AI replay systems struggle with **real-time adaptation**.

---

## 5. Cross-Modal Verification (Audio + Vision)

Combining voice with **camera-based analysis** significantly improves robustness.

| Check | Purpose |
|----|--------|
| Lip-sync vs Audio | Detects mismatch between speech and mouth movement |
| Head movement vs speech stress | AI lacks natural correlation |
| Eye blinking during speech | Humans blink more while talking |

**Tools**:
- MediaPipe Face Mesh
- SyncNet
- Optical Flow Analysis

---

## 6. Whisper-Based Semantic & Linguistic Analysis

Even if transcription is accurate, AI speech shows **linguistic anomalies**.

### Detectable Patterns
- Over-formal grammar
- No disfluencies
- High lexical repetition
- Low semantic entropy

**Metrics**:
- Perplexity score
- Token repetition rate
- Semantic entropy

---

## 7. Ensemble Scoring Strategy (Recommended)

Fake voice detection should **never rely on a single signal**.

### Example Scoring Algorithm
```python
cheating_score = (
    0.35 * spoof_model_score +
    0.20 * behavioral_anomaly +
    0.15 * liveness_fail +
    0.15 * gaze_mismatch +
    0.15 * whisper_anomaly
)
```

### Risk Interpretation
- `0.0 – 0.3` → Low Risk
- `0.3 – 0.6` → Medium Risk
- `0.6 – 1.0` → High Risk

---

## 8. Recommended Practical Stack

For a **Python + GPU-based system**:

- Audio Capture → PyAudio
- Fake Voice Detection → AASIST / RawNet2
- Speech Recognition → Whisper
- Behavioral Metrics → Custom Python logic
- Vision Sync → MediaPipe
- Decision Engine → Weighted Ensemble Algorithm

---

## 9. Limitations & Ethical Considerations

- 100% detection accuracy is impossible
- AI speech generation evolves rapidly
- Models require continuous retraining
- Detection should be **flag-based**, not accusatory
- Transparency is required for legal and academic use

---

## Conclusion

**Fake voice detection is most effective when voice is not treated in isolation.**  
A multi-modal approach combining **audio signals, behavior, vision, and context** provides robust and defensible detection suitable for high-stakes environments such as online examinations and identity verification systems.

