# Detection Methods

Detailed explanation of all detection algorithms used in the Fake Voice Detection System.

## Overview

The system uses a multi-modal approach combining five detection methods:

| Method | Weight | Description |
|--------|--------|-------------|
| Deep Learning | 35% | AASIST neural network |
| Behavioral | 20% | Temporal patterns |
| Signal Features | 15% | Acoustic analysis |
| Linguistic | 15% | Speech content |
| Anomaly Detection | 15% | Outlier detection |

## 1. Signal-Level Features

### MFCC (Mel-Frequency Cepstral Coefficients)

MFCCs capture the spectral envelope of audio, which differs between human and synthetic speech.

**What it detects:**
- Abnormal frequency distributions
- Unnatural spectral patterns
- Missing formant transitions

**Implementation:**
```python
mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc_delta)
```

### Spectral Flux

Measures the rate of change in the power spectrum frame-to-frame.

**What it detects:**
- AI-generated speech often has unnaturally low spectral flux
- Lack of natural variation in speech

**Formula:**
```
flux[t] = Σ max(0, |S[t,f]| - |S[t-1,f]|)²
```

### Pitch Jitter

Cycle-to-cycle variation in pitch (fundamental frequency).

**What it detects:**
- Synthetic voices have unnaturally stable pitch
- Natural speech has 0.5-1% jitter
- AI speech often has < 0.1% jitter

### Shimmer

Cycle-to-cycle variation in amplitude.

**What it detects:**
- Similar to jitter, synthetic voices are too stable
- Natural speech has amplitude variations

### Harmonic-to-Noise Ratio (HNR)

Ratio of harmonic energy to noise energy.

**What it detects:**
- AI-generated audio is often "too clean"
- Natural speech: HNR ~10-15 dB
- Synthetic speech: HNR > 20 dB (suspicious)

### Phase Coherence

Measures consistency of phase relationships across frequencies.

**What it detects:**
- GAN and TTS models often break phase realism
- Natural phase relationships are chaotic
- Synthetic audio shows unnatural coherence

## 2. Deep Learning (AASIST)

### Model Architecture

AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks):

1. **Sinc Convolution Layer**: Learns bandpass filters from raw waveform
2. **Residual Blocks**: Extract hierarchical features
3. **Graph Attention**: Models spectro-temporal relationships
4. **Classification Head**: Binary output (genuine/spoof)

### How It Works

```
Raw Waveform → Sinc Conv → ResBlocks → Graph Attention → Classifier
     ↓              ↓           ↓              ↓            ↓
  16kHz         Filters    Features      Relations    Probability
```

### What It Detects

- Artifacts from TTS synthesis
- Voice cloning signatures
- GAN-generated audio patterns
- Replay attack characteristics

## 3. Behavioral Analysis

### Energy Variation

Natural speech has dynamic energy patterns.

**What it detects:**
- Constant energy levels (suspicious)
- Lack of emphasis and stress patterns
- Missing prosodic variation

### Pause Detection

Analyzes pauses and micro-pauses in speech.

| Pause Type | Duration | Natural Speech |
|------------|----------|----------------|
| Micro-pause | 20-100ms | Frequent |
| Normal pause | 100-500ms | Regular |
| Long pause | 500ms-2s | Occasional |

**What it detects:**
- Absence of micro-pauses
- Unnaturally regular pausing
- Missing breathing pauses

### Cadence Uniformity

Measures regularity of speech rhythm.

**What it detects:**
- AI speech often has uniform cadence
- Natural speech is chaotic and variable
- High uniformity = suspicious

### Breathing Detection

Looks for audible breathing patterns.

**What it detects:**
- Absence of breathing sounds
- Unnatural breathing intervals
- Missing inhale/exhale patterns

## 4. Linguistic Analysis

### Whisper Integration

Uses OpenAI Whisper for transcription and analysis.

### Perplexity Analysis

Measures how "predictable" the text is.

**What it detects:**
- AI-generated text is often more predictable
- Lower perplexity = more suspicious
- Natural speech has higher perplexity

### Disfluency Detection

Looks for natural speech disfluencies.

**Expected disfluencies:**
- "um", "uh", "er", "ah"
- "like", "you know", "I mean"
- False starts and corrections

**What it detects:**
- Absence of disfluencies is suspicious
- Natural speech: 2-5% disfluency rate
- AI speech: often 0% disfluency

### Formality Analysis

Detects overly formal language patterns.

**Formal patterns (suspicious):**
- "Furthermore", "Moreover", "Nevertheless"
- "Consequently", "Subsequently"
- Lack of contractions

### Repetition Analysis

Measures token and phrase repetition.

**What it detects:**
- AI text often has higher repetition
- Repeated phrases or patterns
- Limited vocabulary usage

## 5. Anomaly Detection

### Isolation Forest

Unsupervised anomaly detection algorithm.

**How it works:**
1. Build random trees
2. Anomalies require fewer splits to isolate
3. Score based on path length

**Advantages:**
- Fast and scalable
- Handles high-dimensional data
- No need for labeled data

### One-Class SVM

Learns the boundary of "normal" data.

**How it works:**
1. Map data to high-dimensional space
2. Find hyperplane separating data from origin
3. Anomalies fall outside boundary

**Advantages:**
- Effective for novel spoofing techniques
- Works well with limited normal samples

### Local Outlier Factor (LOF)

Density-based anomaly detection.

**How it works:**
1. Compute local density for each point
2. Compare to neighbors' densities
3. Lower relative density = anomaly

**Advantages:**
- Detects local anomalies
- Handles varying densities
- Catches subtle spoofing artifacts

## Ensemble Scoring

### Weighted Average

```python
final_score = (
    0.35 * deep_learning_score +
    0.20 * behavioral_score +
    0.15 * signal_score +
    0.15 * linguistic_score +
    0.15 * anomaly_score
)
```

### Dynamic Weight Adjustment

Weights are adjusted based on confidence:

| Confidence | Multiplier |
|------------|------------|
| HIGH | 1.5x |
| MEDIUM | 1.0x |
| LOW | 0.5x |

### Risk Interpretation

| Score | Risk Level | Interpretation |
|-------|------------|----------------|
| 0.0 - 0.3 | LOW | Likely genuine |
| 0.3 - 0.6 | MEDIUM | Some concerns |
| 0.6 - 1.0 | HIGH | Likely synthetic |

## Model Calibration

### Why Calibrate?

Raw model outputs may not reflect true probabilities.

### Calibration Methods

1. **Platt Scaling**: Sigmoid fit
2. **Isotonic Regression**: Non-parametric
3. **Temperature Scaling**: Neural network specific

### Calibration Goal

After calibration:
- If model predicts 80% spoof, ~80% of such samples are actually spoof
- Better threshold selection
- More reliable confidence estimates

