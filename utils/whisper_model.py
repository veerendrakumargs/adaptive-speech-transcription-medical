"""
Whisper model loader and inference utilities
Based on OpenAI Whisper for medical speech recognition
"""
import whisper
import torch
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class WhisperTranscriber:
    """Advanced Whisper-based transcription with medical domain support"""
    
    AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    
    def __init__(self, model_size='base', device=None):
        """
        Initialize Whisper model
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: 'cuda' or 'cpu'
        """
        self.model_size = model_size
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load Whisper model"""
        try:
            print(f"Loading Whisper {self.model_size} model on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {e}")
    
    def transcribe(self, audio_path, language=None, task='transcribe',
                  word_timestamps=False, **kwargs):
        """
        Transcribe audio file
        Args:
            audio_path: Path to audio file
            language: Source language code (None for auto-detect)
            task: 'transcribe' or 'translate'
            word_timestamps: Enable word-level timestamps
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Failed to load Whisper model")

        try:
            options = {
                'language': language,
                'task': task,
                'word_timestamps': word_timestamps,
                'fp16': False if self.device == 'cpu' else True,
                **kwargs
            }
            
            result = self.model.transcribe(audio_path, **options)
            
            # Parse word-level timestamps if available
            word_segments = []
            if word_timestamps and 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        word_segments.extend(segment['words'])
            
            return {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'words': word_segments,
                'confidence': self._calculate_confidence(result)
            }
        except Exception as e:
            raise Exception(f"Transcription failed: {e}")
    
    def _calculate_confidence(self, result):
        """Calculate average confidence score from segments"""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        confidences = []
        for segment in result['segments']:
            if 'no_speech_prob' in segment:
                # Convert no_speech_prob to confidence
                confidence = 1.0 - segment['no_speech_prob']
                confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.85
    
    def detect_language(self, audio_path):
        """Detect spoken language"""
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            raise RuntimeError("Failed to load Whisper model")
        
        try:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            
            return detected_language, probs
        except Exception as e:
            return 'en', {}
    
    def batch_transcribe(self, audio_paths, **kwargs):
        """Batch transcribe multiple audio files"""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path, **kwargs)
                results.append({
                    'file': audio_path,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                results.append({
                    'file': audio_path,
                    'status': 'failed',
                    'error': str(e)
                })
        return results
    
    def get_model_info(self):
        """Get information about loaded model"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'parameters': self._count_parameters()
        }
    
    def _count_parameters(self):
        """Count model parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
