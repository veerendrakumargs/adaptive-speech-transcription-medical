# Adaptive Speech Transcription for Medical Environments

An AI-powered speech transcription project designed to explore speech-to-text processing in medical environments using OpenAI Whisper, deep learning, and natural language processing.

## Overview

Medical speech can contain specialised terminology that general-purpose transcription systems may not always recognise accurately. This project explores an adaptive transcription workflow that processes speech audio, generates text transcriptions, and evaluates transcription quality using standard performance metrics.

The project combines speech processing, machine learning, and an interactive application interface to demonstrate how AI can support transcription workflows in specialised domains.

## Key Features

- **Speech-to-Text Transcription:** Uses OpenAI Whisper for automatic speech recognition.
- **Audio Processing:** Uses Librosa to support audio preprocessing and analysis.
- **Medical NLP:** Includes a dedicated utility module for medical natural language processing.
- **Model Integration:** Uses PyTorch and Hugging Face technologies for deep learning workflows.
- **Performance Evaluation:** Includes metrics for evaluating transcription quality, including Word Error Rate (WER), precision, recall, and F1-score where applicable.
- **Interactive Interface:** Provides a Streamlit-based application with separate pages for transcription, evaluation, training, settings, and visualisation.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Speech Recognition | OpenAI Whisper |
| Deep Learning | PyTorch |
| NLP and Model Tools | Hugging Face |
| Audio Processing | Librosa |
| User Interface | Streamlit |
| Evaluation | WER, Precision, Recall, F1-score |
| Dependency Management | pip / uv |

## Project Structure

```text
adaptive-speech-transcription/
├── app.py
├── main.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── Dataset.md
├── Whisper.md
├── pages/
│   ├── home.py
│   ├── transcription.py
│   ├── evaluation.py
│   ├── training.py
│   ├── settings.py
│   └── visualization.py
└── utils/
    ├── audio_processor.py
    ├── medical_nlp.py
    ├── metrics.py
    ├── translator.py
    └── whisper_model.py
```

## Getting Started

### Prerequisites

- Python installed on your system
- Git
- pip or uv for dependency management

### 1. Clone the repository

```bash
git clone https://github.com/veerendrakumargs/adaptive-speech-transcription-medical.git
cd adaptive-speech-transcription-medical
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

*Note: These commands assume that `requirements.txt` contains the necessary dependencies and that `app.py` is the Streamlit entry point. Additional setup may be required depending on your environment and configuration.*

## Evaluation Metrics

The project includes utilities for evaluating transcription performance.

- **Word Error Rate (WER):** Measures transcription errors relative to a reference transcript.
- **Precision:** Measures the proportion of predicted positive results that are correct, where applicable.
- **Recall:** Measures the proportion of actual positive results correctly identified, where applicable.
- **F1-score:** Combines precision and recall into a single metric.

Actual performance should be reported using measured results on a documented test dataset. No accuracy figures are claimed here.

## Dataset and Model Documentation

Refer to the following project documentation for additional details:

- [`Dataset.md`](Dataset.md) — dataset-related documentation.
- [`Whisper.md`](Whisper.md) — Whisper-related documentation.

## Potential Applications

- Medical dictation and speech-to-text research
- Exploration of specialised terminology in transcription
- Evaluation of speech recognition systems
- Research into NLP-assisted transcription workflows

**Disclaimer:** This project is intended for educational and research purposes. It is not a medical diagnostic system, and its transcriptions should be reviewed by qualified personnel before use in clinical workflows.

## Future Improvements

- Evaluate transcription performance on domain-specific datasets.
- Improve recognition and handling of medical terminology.
- Compare transcription results across different speech recognition models.
- Expand multilingual transcription and translation capabilities.
- Improve reproducibility through documented experiments and evaluation results.

## Author

**Veerendrakumar G S**

B.E. in Artificial Intelligence and Machine Learning  
New Horizon College of Engineering, Bengaluru

- GitHub: [veerendrakumargs](https://github.com/veerendrakumargs)

---

*Built as an applied AI/ML project exploring speech recognition, audio processing, and medical-domain NLP.*
