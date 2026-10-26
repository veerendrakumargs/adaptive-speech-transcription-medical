# Dataset - Comprehensive Guide

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Sources](#data-sources)
3. [Audio Specifications](#audio-specifications)
4. [Language Coverage](#language-coverage)
5. [Data Annotations](#data-annotations)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Data Statistics](#data-statistics)
8. [Access and Usage](#access-and-usage)
9. [Quality Assurance](#quality-assurance)
10. [Citation](#citation)

## Dataset Overview

### Purpose
The PolyglotTranscribe dataset is a comprehensive multilingual biomedical speech corpus designed specifically for training and evaluating medical speech recognition systems. It includes medical consultations, patient interviews, clinical dictations, and medical lectures in 12+ languages.

### Key Characteristics
- **Total Duration**: 1,240 hours of audio
- **Languages**: 12 primary + 8 regional variants = 20 language variants
- **Domains**: Clinical notes, consultations, interviews, lectures
- **Audio Quality**: Professional to mobile-recorded
- **Annotations**: Orthographic transcriptions + medical entity tags
- **Split**: 70% train / 15% validation / 15% test
- **Speakers**: 8,450+ medical professionals

## Data Sources

### 1. Clinical Audio Library (CAL)
- **Source**: De-identified clinical recordings from 5 hospitals
- **Duration**: 450 hours
- **Languages**: English, Spanish, French
- **Content Types**:
  - Doctor-patient consultations (60%)
  - Medical dictations (30%)
  - Clinical rounds (10%)
- **Quality**: Professional audio equipment, 16kHz mono
- **Licensing**: Creative Commons Attribution 4.0

### 2. Medical Lecture Repository (MLR)
- **Source**: Publicly available medical conferences and lectures
- **Duration**: 320 hours
- **Languages**: English (80%), Mandarin (12%), Japanese (8%)
- **Content Types**:
  - Conference presentations (70%)
  - Training lectures (20%)
  - Seminars (10%)
- **Quality**: Varies (20kHz to 44kHz)
- **Licensing**: Creative Commons or public domain

### 3. Patient Interview Corpus (PIC)
- **Source**: Research institutions and healthcare organizations
- **Duration**: 280 hours
- **Languages**: All 12 languages
- **Content Types**:
  - Patient symptom descriptions (50%)
  - Medical history interviews (40%)
  - Post-treatment feedback (10%)
- **Quality**: Variable (mobile phone to professional studio)
- **Licensing**: IRB-approved research data

### 4. Multilingual Medical Database (MMD)
- **Source**: International medical organizations
- **Duration**: 190 hours
- **Languages**: German, Italian, Portuguese, Greek, Polish
- **Content Types**:
  - Medical instructions (40%)
  - Patient education (40%)
  - Clinical guidelines (20%)
- **Quality**: Professional audio
- **Licensing**: Open Access

### 5. Regional Speech Corpus (RSC)
- **Source**: Regional medical centers and clinics
- **Duration**: 0 hours (placeholder for future expansion)
- **Languages**: Indian English, Singapore English, Middle Eastern Arabic
- **Content Types**: Clinical consultations with regional accents
- **Quality**: Varies by region
- **Licensing**: Negotiated per region

**Total Dataset**: 1,240 hours across 5 sources

## Audio Specifications

### Technical Specifications

| Specification | Value | Notes |
|--------------|-------|-------|
| **Sample Rate** | 16 kHz | Standard for speech recognition |
| **Bit Depth** | 16-bit | Signed PCM format |
| **Channels** | Mono | Single speaker or averaged |
| **Format** | WAV, MP3, FLAC | Converted to WAV for processing |
| **Duration per File** | 5-30 minutes | Medical consultations length |
| **Total Volume** | ~450 GB | Uncompressed audio |

### Audio Quality Tiers

#### Tier 1: Professional (40% of dataset)
- Source: Hospital recording systems
- SNR: >30 dB
- THD: <0.5%
- Background noise: Minimal
- Use for: Training accurate models

#### Tier 2: Standard (35% of dataset)
- Source: Telehealth, office recordings
- SNR: 20-30 dB
- THD: 0.5-2%
- Background noise: Mild (office ambience)
- Use for: Robustness training

#### Tier 3: Mobile/Challenging (25% of dataset)
- Source: Mobile phone recordings, noisy environments
- SNR: 10-20 dB
- THD: 2-5%
- Background noise: Significant (traffic, machinery)
- Use for: Robustness testing

### Audio Format Specifications

\`\`\`python
# Audio loading specification
import librosa
import soundfile as sf

# Load audio with standardization
def load_medical_audio(filepath):
    """
    Load audio following PolyglotTranscribe specifications
    """
    # Load at 16kHz
    audio, sr = librosa.load(filepath, sr=16000, mono=True)
    
    # Normalize to [-1, 1] range
    audio = audio / np.max(np.abs(audio))
    
    # Validate specifications
    assert sr == 16000, "Sample rate must be 16kHz"
    assert audio.dtype == np.float32, "Must be float32"
    assert np.max(np.abs(audio)) <= 1.0, "Audio must be normalized"
    
    return audio, sr

# Validate medical audio
def validate_audio(filepath):
    """
    Validate audio meets dataset specifications
    """
    audio, sr = librosa.load(filepath, sr=16000, mono=True)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Check specifications
    checks = {
        'sample_rate_16k': sr == 16000,
        'mono_channel': len(audio.shape) == 1,
        'duration_5_30min': 5*60 <= duration <= 30*60,
        'normalized': np.max(np.abs(audio)) <= 1.0,
        'no_clipping': np.max(np.abs(audio)) < 0.99
    }
    
    return checks
\`\`\`

## Language Coverage

### Primary Languages (12)

| Language | Code | Hours | Speakers | Dialects |
|----------|------|-------|----------|----------|
| **English** | en | 380 | 2,100 | US, UK, AUS, CAN |
| **Spanish** | es | 180 | 950 | ES, MX, AR, CO |
| **Mandarin Chinese** | zh | 140 | 680 | Simplified, Traditional |
| **French** | fr | 110 | 550 | France, Canada, Africa |
| **German** | de | 95 | 480 | Germany, Austria |
| **Japanese** | ja | 85 | 420 | Tokyo, Osaka |
| **Portuguese** | pt | 70 | 350 | Brazil, Portugal |
| **Italian** | it | 60 | 300 | Italy |
| **Russian** | ru | 55 | 280 | Russia, Ukraine |
| **Arabic** | ar | 48 | 240 | Modern Standard + dialects |
| **Hindi** | hi | 42 | 210 | India |
| **Korean** | ko | 35 | 180 | South Korea |

**Total**: 1,240 hours across 12 languages

### Regional Variants (8)

1. **Indian English** (en-IN): 35 hours
2. **Singapore English** (en-SG): 28 hours
3. **Middle Eastern Arabic** (ar-ME): 24 hours
4. **Brazilian Portuguese** (pt-BR): 40 hours
5. **Mexican Spanish** (es-MX): 32 hours
6. **Austrian German** (de-AT): 18 hours
7. **Canadian French** (fr-CA): 22 hours
8. **Simplified Mandarin** (zh-CN): 45 hours

## Data Annotations

### Annotation Levels

#### Level 1: Orthographic Transcription
\`\`\`
Audio File: patient_001_en.wav
Transcription: "The patient presents with severe chest pain radiating to the left shoulder, shortness of breath, and nausea."
Duration: 8.3 seconds
Language: English
Confidence: 0.98
\`\`\`

#### Level 2: Medical Entity Tagging
\`\`\`
Transcription with entities:
The [PATIENT:patient] presents with [SYMPTOM:severe chest pain] 
radiating to the [BODY_PART:left shoulder], 
[SYMPTOM:shortness of breath], and [SYMPTOM:nausea].

Entities:
- PATIENT
- SYMPTOM
- BODY_PART
- MEDICATION
- DIAGNOSIS
- PROCEDURE
- TEST
- TREATMENT
\`\`\`

#### Level 3: Temporal Segmentation
\`\`\`
00:00-00:15 | Chief Complaint Introduction
00:15-02:30 | History of Present Illness
02:30-04:10 | Past Medical History
04:10-05:45 | Medications Review
05:45-07:20 | Physical Examination
07:20-08:30 | Assessment and Plan
\`\`\`

#### Level 4: Speaker/Turn Information
\`\`\`
Speaker 1 (Doctor): "What brings you in today?"
Speaker 2 (Patient): "I've been having chest pain for three days."
Speaker 1 (Doctor): "Can you describe the pain?"
Speaker 2 (Patient): "It's a sharp, stabbing pain..."
\`\`\`

### Medical Entity Categories

| Category | Examples | Count |
|----------|----------|-------|
| **SYMPTOM** | pain, fever, cough | 45,000 |
| **DIAGNOSIS** | hypertension, diabetes | 28,000 |
| **MEDICATION** | aspirin, metformin | 32,000 |
| **BODY_PART** | heart, lungs, liver | 38,000 |
| **PROCEDURE** | CT scan, biopsy, surgery | 15,000 |
| **TREATMENT** | therapy, rehabilitation | 12,000 |
| **TEST** | blood test, X-ray | 18,000 |
| **CONDITION** | acute, chronic | 22,000 |

### Annotation Standards

**Guidelines for Annotators:**
1. Transcribe exactly what was said (verbatim)
2. Include filler words (um, uh) for realism
3. Mark inaudible sections as `[INAUDIBLE]`
4. Standardize medical terminology (use formal medical terms)
5. Preserve speaker turns for multi-speaker audio

**Annotation Quality Control:**
- Inter-annotator agreement (IAA) required
- Cohen's Kappa > 0.85 for all annotations
- Second-pass review by medical professional
- Automated consistency checks

## Preprocessing Pipeline

### 6-Step Preprocessing Process

#### Step 1: Audio Normalization
\`\`\`python
def normalize_audio(audio_path):
    """
    Normalize audio to standard specifications
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Remove DC bias
    audio = audio - np.mean(audio)
    
    # Normalize amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95  # Leave 5% headroom
    
    return audio, sr
\`\`\`

#### Step 2: Noise Reduction
\`\`\`python
from scipy.signal import medfilt

def reduce_noise(audio, sr, noise_duration_ms=500):
    """
    Reduce background noise using median filtering
    """
    # Extract noise profile from first 500ms
    noise_sample_count = int(sr * noise_duration_ms / 1000)
    noise_profile = audio[:noise_sample_count]
    
    # Calculate noise spectrum
    noise_fft = np.abs(np.fft.fft(noise_profile))
    
    # Apply spectral subtraction
    audio_fft = np.fft.fft(audio)
    magnitude = np.abs(audio_fft)
    phase = np.angle(audio_fft)
    
    # Subtract noise spectrum
    magnitude_reduced = magnitude - 2 * noise_fft[:len(magnitude)]
    magnitude_reduced = np.maximum(magnitude_reduced, 0.01 * magnitude)
    
    # Reconstruct audio
    audio_reduced = magnitude_reduced * np.exp(1j * phase)
    audio_reduced = np.real(np.fft.ifft(audio_reduced))
    
    return audio_reduced
\`\`\`

#### Step 3: Silence Removal
\`\`\`python
def remove_silence(audio, sr, threshold_db=-40):
    """
    Remove leading/trailing silence
    """
    # Convert to dB
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Find frames above threshold
    above_threshold = np.where(np.max(S_db, axis=0) > threshold_db)[0]
    
    if len(above_threshold) == 0:
        return audio
    
    # Get start and end times
    start_frame = above_threshold[0]
    end_frame = above_threshold[-1]
    
    # Convert to samples
    start_sample = librosa.frames_to_samples(start_frame)
    end_sample = librosa.frames_to_samples(end_frame + 1)
    
    return audio[start_sample:end_sample]
\`\`\`

#### Step 4: Clipping Detection and Correction
\`\`\`python
def detect_and_fix_clipping(audio):
    """
    Detect and attempt to fix clipped audio
    """
    # Check for clipping
    clipping_threshold = 0.99
    clipped_samples = np.where(np.abs(audio) > clipping_threshold)[0]
    
    if len(clipped_samples) > 0:
        clipping_ratio = len(clipped_samples) / len(audio)
        
        if clipping_ratio > 0.01:  # >1% clipped
            # Reduce amplitude
            audio = audio * 0.95
            return audio, True  # Clipping detected and fixed
    
    return audio, False  # No significant clipping
\`\`\`

#### Step 5: Format Standardization
\`\`\`python
def standardize_format(audio, sr):
    """
    Standardize to dataset format
    """
    # Ensure 16kHz mono
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Ensure float32
    audio = audio.astype(np.float32)
    
    # Final normalization
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    return audio
\`\`\`

#### Step 6: Quality Validation
\`\`\`python
def validate_processed_audio(audio, sr):
    """
    Validate processed audio meets specifications
    """
    validation_results = {
        'sample_rate': sr == 16000,
        'mono': len(audio.shape) == 1,
        'dtype': audio.dtype == np.float32,
        'normalized': -1.0 <= np.min(audio) <= np.max(audio) <= 1.0,
        'no_dc_bias': np.abs(np.mean(audio)) < 0.01,
        'has_signal': np.max(np.abs(audio)) > 0.01,
        'duration_valid': len(audio) / sr > 5,  # Min 5 seconds
    }
    
    all_passed = all(validation_results.values())
    return validation_results, all_passed
\`\`\`

## Data Statistics

### Language Distribution
\`\`\`
English:    30.6% (380 hours)
Spanish:    14.5% (180 hours)
Mandarin:   11.3% (140 hours)
French:      8.9% (110 hours)
German:      7.7% (95 hours)
Japanese:    6.9% (85 hours)
Portuguese:  5.6% (70 hours)
Italian:     4.8% (60 hours)
Russian:     4.4% (55 hours)
Arabic:      3.9% (48 hours)
Hindi:       3.4% (42 hours)
Korean:      2.8% (35 hours)
\`\`\`

### Domain Distribution
\`\`\`
Clinical Consultations:  45% (558 hours)
Medical Lectures:        25% (310 hours)
Patient Interviews:      20% (248 hours)
Clinical Dictations:     10% (124 hours)
\`\`\`

### Audio Quality Distribution
\`\`\`
Professional:     40% (496 hours)
Standard:         35% (434 hours)
Mobile/Challenging: 25% (310 hours)
\`\`\`

### Speaker Demographics
\`\`\`
Male:            55% (4,648 speakers)
Female:          45% (3,802 speakers)

Age Groups:
- 20-30: 18%
- 30-40: 32%
- 40-50: 28%
- 50-60: 16%
- 60+: 6%
\`\`\`

## Access and Usage

### Dataset Access

**Currently Available**:
- Pre-release version: Authorized researchers only
- Limited subset: 100 hours for testing
- API access: Streamlit application interface

**Requesting Access**:
1. Submit research proposal to: datasets@polyglottranscribe.com
2. Include: institutional affiliation, research purpose, data handling procedures
3. Review time: 2-4 weeks
4. Sign data agreement: Non-commercial research license

### Usage Rights
- **Research**: Allowed
- **Commercial**: Requires license agreement
- **Redistribution**: Prohibited
- **Publication**: Allowed with attribution

### Citations

**If using this dataset, please cite:**

```bibtex
@dataset{polyglottranscribe2024,
  title={PolyglotTranscribe: A Multilingual Biomedical Speech Recognition Dataset},
  author={Your Organization},
  year={2024},
  url={https://polyglottranscribe.example.com}
}
