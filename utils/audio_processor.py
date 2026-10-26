"""
Audio processing utilities for biomedical speech transcription
Includes noise reduction, format conversion, and audio enhancement
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
import os

class AudioProcessor:
    """Advanced audio processing for medical speech transcription"""
    
    def __init__(self, sample_rate=16000, noise_threshold=0.02):
        self.sample_rate = sample_rate
        self.noise_threshold = noise_threshold
    
    def load_audio(self, file_path, target_sr=None):
        """Load audio file with automatic resampling"""
        if target_sr is None:
            target_sr = self.sample_rate
        
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    
    def reduce_noise(self, audio, sr):
        """Apply noise reduction using spectral gating"""
        try:
            # Simple noise reduction using spectral subtraction
            # Estimate noise from first 0.5 seconds
            noise_sample = audio[:int(0.5 * sr)]
            noise_profile = np.abs(librosa.stft(noise_sample)).mean(axis=1)
            
            # Apply spectral subtraction
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Subtract noise profile
            magnitude_clean = magnitude - noise_profile[:, np.newaxis] * self.noise_threshold
            magnitude_clean = np.maximum(magnitude_clean, 0)
            
            # Reconstruct audio
            D_clean = magnitude_clean * np.exp(1j * phase)
            audio_clean = librosa.istft(D_clean)
            
            return audio_clean
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
    
    def trim_silence(self, audio, sr, top_db=20):
        """Remove silence from audio"""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def normalize_audio(self, audio):
        """Normalize audio amplitude"""
        return librosa.util.normalize(audio)
    
    def resample_audio(self, audio, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def extract_features(self, audio, sr):
        """Extract audio features for analysis"""
        features = {
            'duration': len(audio) / sr,
            'rms': float(np.sqrt(np.mean(audio**2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
            'tempo': float(librosa.beat.tempo(y=audio, sr=sr)[0]) if len(audio) > sr else 0
        }
        return features
    
    def process_audio(self, file_path, enable_noise_reduction=True, 
                     trim_silence_enabled=True, target_sr=None):
        """
        Complete audio processing pipeline
        Returns: processed audio, sample rate, features
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        # Load audio
        audio, sr = self.load_audio(file_path, target_sr)
        
        # Apply processing steps
        if trim_silence_enabled:
            audio = self.trim_silence(audio, sr)
        
        if enable_noise_reduction:
            audio = self.reduce_noise(audio, sr)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Save processed audio to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio, sr)
        
        return temp_file.name, sr, features
    
    def convert_to_wav(self, input_file):
        """Convert any audio format to WAV"""
        try:
            audio, sr = librosa.load(input_file, sr=self.sample_rate)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio, sr)
            return temp_file.name
        except Exception as e:
            raise Exception(f"Audio conversion failed: {e}")
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            from moviepy.editor import VideoFileClip
            
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
            video.close()
            
            return temp_audio.name
        except Exception as e:
            raise Exception(f"Video audio extraction failed: {e}")
