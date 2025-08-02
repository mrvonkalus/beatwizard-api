"""
Audio Processor - Core audio file handling and preprocessing
Enhanced audio loading with professional-grade preprocessing
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from loguru import logger

from config.settings import audio_settings


class AudioProcessor:
    """
    Professional audio file processor with enhanced capabilities
    Handles loading, preprocessing, and basic audio operations
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Target sample rate (defaults to config)
            hop_length: Hop length for STFT analysis (defaults to config)
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        self.frame_size = audio_settings.DEFAULT_FRAME_SIZE
        
        logger.info(f"AudioProcessor initialized - SR: {self.sample_rate}, Hop: {self.hop_length}")
    
    def load_audio(self, file_path: str, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load audio file with professional preprocessing
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio to [-1, 1]
            
        Returns:
            Tuple of (audio_data, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if file_path.suffix.lower() not in audio_settings.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")
        
        try:
            # Load audio with librosa for professional handling
            audio, original_sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=False,  # Preserve stereo for stereo analysis
                duration=None
            )
            
            # Get additional metadata
            with sf.SoundFile(str(file_path)) as sf_file:
                metadata = {
                    'original_sample_rate': sf_file.samplerate,
                    'channels': sf_file.channels,
                    'frames': sf_file.frames,
                    'duration': sf_file.frames / sf_file.samplerate,
                    'format': sf_file.format,
                    'subtype': sf_file.subtype,
                    'file_size': file_path.stat().st_size,
                    'bit_depth': sf_file.subtype if 'PCM_' in sf_file.subtype else 'Variable'
                }
            
            # Handle stereo/mono conversion
            if len(audio.shape) == 2:
                # Stereo audio
                metadata['is_stereo'] = True
                metadata['channels'] = 2
            else:
                # Mono audio
                metadata['is_stereo'] = False
                metadata['channels'] = 1
            
            # Normalize if requested
            if normalize and np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                metadata['normalized'] = True
            else:
                metadata['normalized'] = False
            
            # Additional processing metadata
            metadata['processed_sample_rate'] = self.sample_rate
            metadata['hop_length'] = self.hop_length
            metadata['frame_size'] = self.frame_size
            
            logger.info(f"Loaded audio: {file_path.name} ({metadata['duration']:.2f}s, {metadata['channels']}ch)")
            
            return audio, metadata
            
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {str(e)}")
            raise
    
    def get_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono
        
        Args:
            audio: Audio data (can be stereo or mono)
            
        Returns:
            Mono audio data
        """
        if len(audio.shape) == 2:
            return librosa.to_mono(audio)
        return audio
    
    def get_stereo_channels(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right channels from stereo audio
        
        Args:
            audio: Stereo audio data
            
        Returns:
            Tuple of (left_channel, right_channel)
        """
        if len(audio.shape) != 2 or audio.shape[0] != 2:
            raise ValueError("Audio must be stereo (2 channels)")
        
        return audio[0], audio[1]
    
    def apply_high_pass_filter(self, audio: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise
        
        Args:
            audio: Audio data
            cutoff: Cutoff frequency in Hz
            
        Returns:
            Filtered audio
        """
        # Simple high-pass filter using librosa
        return librosa.effects.preemphasis(audio, coef=0.95)
    
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40.0) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Trim silence from beginning and end of audio
        
        Args:
            audio: Audio data
            threshold_db: Silence threshold in dB
            
        Returns:
            Tuple of (trimmed_audio, (start_sample, end_sample))
        """
        mono_audio = self.get_mono(audio) if len(audio.shape) == 2 else audio
        
        # Find non-silent regions
        intervals = librosa.effects.split(
            mono_audio,
            top_db=abs(threshold_db),
            hop_length=self.hop_length
        )
        
        if len(intervals) == 0:
            logger.warning("No non-silent regions found")
            return audio, (0, len(audio))
        
        start_sample = intervals[0][0]
        end_sample = intervals[-1][1]
        
        # Apply trimming to original audio (preserve stereo if present)
        if len(audio.shape) == 2:
            trimmed = audio[:, start_sample:end_sample]
        else:
            trimmed = audio[start_sample:end_sample]
        
        logger.debug(f"Trimmed silence: {start_sample}-{end_sample} samples")
        
        return trimmed, (start_sample, end_sample)
    
    def calculate_rms_energy(self, audio: np.ndarray, frame_length: int = None) -> np.ndarray:
        """
        Calculate RMS energy across time
        
        Args:
            audio: Audio data
            frame_length: Frame length for analysis
            
        Returns:
            RMS energy values
        """
        frame_length = frame_length or self.frame_size
        mono_audio = self.get_mono(audio) if len(audio.shape) == 2 else audio
        
        return librosa.feature.rms(
            y=mono_audio,
            frame_length=frame_length,
            hop_length=self.hop_length
        )[0]
    
    def detect_silence_regions(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """
        Detect silence regions in audio
        
        Args:
            audio: Audio data
            threshold_db: Silence threshold in dB
            
        Returns:
            Binary mask where True indicates silence
        """
        mono_audio = self.get_mono(audio) if len(audio.shape) == 2 else audio
        
        # Calculate RMS energy
        rms = self.calculate_rms_energy(mono_audio)
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Create silence mask
        silence_mask = rms_db < threshold_db
        
        return silence_mask
    
    def validate_audio_quality(self, audio: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate audio quality and detect potential issues
        
        Args:
            audio: Audio data
            metadata: Audio metadata
            
        Returns:
            Dictionary with quality analysis results
        """
        mono_audio = self.get_mono(audio) if len(audio.shape) == 2 else audio
        
        quality_analysis = {
            'sample_rate_adequate': metadata['processed_sample_rate'] >= 44100,
            'duration_adequate': metadata['duration'] >= 10.0,  # At least 10 seconds
            'dynamic_range': None,
            'clipping_detected': False,
            'dc_offset': None,
            'noise_floor': None,
            'overall_quality': 'unknown'
        }
        
        # Check for clipping
        peak_value = np.max(np.abs(mono_audio))
        quality_analysis['clipping_detected'] = peak_value >= 0.99
        
        # Calculate dynamic range (simplified)
        rms = np.sqrt(np.mean(mono_audio**2))
        if rms > 0:
            quality_analysis['dynamic_range'] = 20 * np.log10(peak_value / rms)
        
        # Check DC offset
        quality_analysis['dc_offset'] = abs(np.mean(mono_audio))
        
        # Estimate noise floor (bottom 10% of RMS values)
        rms_frames = self.calculate_rms_energy(mono_audio)
        noise_floor_estimate = np.percentile(rms_frames[rms_frames > 0], 10)
        quality_analysis['noise_floor'] = 20 * np.log10(noise_floor_estimate) if noise_floor_estimate > 0 else -np.inf
        
        # Overall quality assessment
        issues = []
        if not quality_analysis['sample_rate_adequate']:
            issues.append('low_sample_rate')
        if not quality_analysis['duration_adequate']:
            issues.append('short_duration')
        if quality_analysis['clipping_detected']:
            issues.append('clipping')
        if quality_analysis['dc_offset'] > 0.1:
            issues.append('dc_offset')
        
        if not issues:
            quality_analysis['overall_quality'] = 'good'
        elif len(issues) <= 1:
            quality_analysis['overall_quality'] = 'acceptable'
        else:
            quality_analysis['overall_quality'] = 'poor'
        
        quality_analysis['issues'] = issues
        
        logger.debug(f"Audio quality: {quality_analysis['overall_quality']} (issues: {issues})")
        
        return quality_analysis