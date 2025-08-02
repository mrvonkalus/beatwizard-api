"""
Enhanced Tempo Detection - Professional BPM analysis
Advanced tempo detection with confidence scoring and multiple algorithms
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class TempoDetector:
    """
    Professional tempo detection with multiple algorithms and confidence scoring
    Uses librosa's advanced beat tracking and tempo estimation
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """
        Initialize the tempo detector
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        
        # Tempo detection parameters
        self.tempo_range = (60, 200)  # BPM range for most music
        self.beat_track_units = 'time'
        
        logger.debug("TempoDetector initialized")
    
    def detect_tempo(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive tempo detection with multiple methods
        
        Args:
            audio: Audio data (mono)
            
        Returns:
            Dictionary with tempo analysis results
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        logger.debug("Starting comprehensive tempo analysis")
        
        # Method 1: Standard beat tracking
        tempo_standard, beats_standard = self._detect_tempo_standard(audio)
        
        # Method 2: Onset-based tempo
        tempo_onset = self._detect_tempo_onset_based(audio)
        
        # Method 3: Autocorrelation-based tempo
        tempo_autocorr = self._detect_tempo_autocorr(audio)
        
        # Method 4: Multi-scale tempo detection
        tempo_multiscale = self._detect_tempo_multiscale(audio)
        
        # Aggregate results and calculate confidence
        all_tempos = [tempo_standard, tempo_onset, tempo_autocorr, tempo_multiscale]
        valid_tempos = [t for t in all_tempos if t is not None and self.tempo_range[0] <= t <= self.tempo_range[1]]
        
        if not valid_tempos:
            logger.warning("No valid tempo detected")
            return self._create_empty_result()
        
        # Primary tempo (most common)
        primary_tempo = self._find_consensus_tempo(valid_tempos)
        
        # Calculate confidence based on agreement between methods
        confidence = self._calculate_tempo_confidence(valid_tempos, primary_tempo)
        
        # Beat analysis
        beat_analysis = self._analyze_beats(audio, primary_tempo, beats_standard)
        
        # Tempo stability analysis
        stability_analysis = self._analyze_tempo_stability(audio, primary_tempo)
        
        result = {
            'primary_tempo': primary_tempo,
            'confidence': confidence,
            'tempo_methods': {
                'standard': tempo_standard,
                'onset_based': tempo_onset,
                'autocorrelation': tempo_autocorr,
                'multiscale': tempo_multiscale
            },
            'beat_analysis': beat_analysis,
            'stability_analysis': stability_analysis,
            'tempo_category': self._categorize_tempo(primary_tempo),
            'all_detected_tempos': valid_tempos
        }
        
        logger.info(f"Tempo detected: {primary_tempo:.1f} BPM (confidence: {confidence:.2f})")
        
        return result
    
    def _detect_tempo_standard(self, audio: np.ndarray) -> Tuple[float, np.ndarray]:
        """Standard librosa beat tracking"""
        try:
            tempo, beats = librosa.beat.beat_track(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units=self.beat_track_units
            )
            return float(tempo), beats
        except Exception as e:
            logger.warning(f"Standard tempo detection failed: {e}")
            return None, np.array([])
    
    def _detect_tempo_onset_based(self, audio: np.ndarray) -> Optional[float]:
        """Onset-based tempo detection"""
        try:
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units='time'
            )
            
            if len(onset_frames) < 4:
                return None
            
            # Calculate inter-onset intervals
            intervals = np.diff(onset_frames)
            
            # Find most common interval (mode)
            # Convert to BPM
            if len(intervals) > 0:
                median_interval = np.median(intervals)
                tempo = 60.0 / median_interval if median_interval > 0 else None
                return tempo
            
            return None
            
        except Exception as e:
            logger.warning(f"Onset-based tempo detection failed: {e}")
            return None
    
    def _detect_tempo_autocorr(self, audio: np.ndarray) -> Optional[float]:
        """Autocorrelation-based tempo detection"""
        try:
            # Use onset strength for autocorrelation
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Compute tempogram
            tempogram = librosa.feature.fourier_tempogram(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Extract tempo from tempogram
            tempo_freqs = librosa.fourier_tempo_frequencies(
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Find peak in tempo spectrum
            tempo_power = np.mean(np.abs(tempogram), axis=1)
            valid_indices = np.where((tempo_freqs >= self.tempo_range[0]) & 
                                   (tempo_freqs <= self.tempo_range[1]))[0]
            
            if len(valid_indices) > 0:
                peak_idx = valid_indices[np.argmax(tempo_power[valid_indices])]
                return float(tempo_freqs[peak_idx])
            
            return None
            
        except Exception as e:
            logger.warning(f"Autocorrelation tempo detection failed: {e}")
            return None
    
    def _detect_tempo_multiscale(self, audio: np.ndarray) -> Optional[float]:
        """Multi-scale tempo detection"""
        try:
            # Use multiple hop lengths for different time scales
            hop_lengths = [256, 512, 1024]
            tempos = []
            
            for hop_len in hop_lengths:
                try:
                    tempo, _ = librosa.beat.beat_track(
                        y=audio,
                        sr=self.sample_rate,
                        hop_length=hop_len
                    )
                    if self.tempo_range[0] <= tempo <= self.tempo_range[1]:
                        tempos.append(tempo)
                except:
                    continue
            
            if tempos:
                return float(np.median(tempos))
            
            return None
            
        except Exception as e:
            logger.warning(f"Multi-scale tempo detection failed: {e}")
            return None
    
    def _find_consensus_tempo(self, tempos: List[float]) -> float:
        """Find consensus tempo from multiple detections"""
        if not tempos:
            return 120.0  # Default fallback
        
        # Group similar tempos (within 5% tolerance)
        tempo_groups = []
        
        for tempo in tempos:
            added_to_group = False
            for group in tempo_groups:
                if any(abs(tempo - t) / t < 0.05 for t in group):
                    group.append(tempo)
                    added_to_group = True
                    break
            
            if not added_to_group:
                tempo_groups.append([tempo])
        
        # Find largest group
        largest_group = max(tempo_groups, key=len)
        
        # Return median of largest group
        return float(np.median(largest_group))
    
    def _calculate_tempo_confidence(self, tempos: List[float], primary_tempo: float) -> float:
        """Calculate confidence score for tempo detection"""
        if not tempos:
            return 0.0
        
        # Count how many methods agree with primary tempo (within 5%)
        agreements = sum(1 for t in tempos if abs(t - primary_tempo) / primary_tempo < 0.05)
        
        # Base confidence on agreement ratio
        base_confidence = agreements / len(tempos)
        
        # Bonus for having multiple methods
        method_bonus = min(len(tempos) / 4.0, 1.0) * 0.2
        
        # Penalty for extreme tempos (less reliable)
        if primary_tempo < 80 or primary_tempo > 160:
            extreme_penalty = 0.1
        else:
            extreme_penalty = 0.0
        
        confidence = base_confidence + method_bonus - extreme_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _analyze_beats(self, audio: np.ndarray, tempo: float, beats: np.ndarray) -> Dict[str, any]:
        """Analyze beat characteristics"""
        if len(beats) < 2:
            return {'beat_count': 0, 'beat_consistency': 0.0, 'avg_beat_interval': 0.0}
        
        # Beat intervals
        beat_intervals = np.diff(beats)
        
        # Beat consistency (lower std = more consistent)
        if len(beat_intervals) > 0:
            consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.0
        
        return {
            'beat_count': len(beats),
            'beat_consistency': consistency,
            'avg_beat_interval': float(np.mean(beat_intervals)) if len(beat_intervals) > 0 else 0.0,
            'beat_interval_std': float(np.std(beat_intervals)) if len(beat_intervals) > 0 else 0.0,
            'beats_per_second': len(beats) / (len(audio) / self.sample_rate)
        }
    
    def _analyze_tempo_stability(self, audio: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze tempo stability throughout the track"""
        try:
            # Divide audio into segments for local tempo analysis
            segment_length = 10.0  # 10 seconds per segment
            segment_samples = int(segment_length * self.sample_rate)
            
            if len(audio) < segment_samples:
                return {'is_stable': True, 'tempo_variance': 0.0, 'tempo_drift': 0.0}
            
            # Analyze tempo in each segment
            segment_tempos = []
            
            for start in range(0, len(audio) - segment_samples, segment_samples // 2):
                segment = audio[start:start + segment_samples]
                try:
                    seg_tempo, _ = librosa.beat.beat_track(
                        y=segment,
                        sr=self.sample_rate,
                        hop_length=self.hop_length
                    )
                    if self.tempo_range[0] <= seg_tempo <= self.tempo_range[1]:
                        segment_tempos.append(seg_tempo)
                except:
                    continue
            
            if len(segment_tempos) < 2:
                return {'is_stable': True, 'tempo_variance': 0.0, 'tempo_drift': 0.0}
            
            # Calculate stability metrics
            tempo_variance = np.var(segment_tempos)
            tempo_drift = abs(segment_tempos[-1] - segment_tempos[0])
            
            # Stability threshold (2 BPM variance is considered stable)
            is_stable = tempo_variance < 4.0
            
            return {
                'is_stable': is_stable,
                'tempo_variance': float(tempo_variance),
                'tempo_drift': float(tempo_drift),
                'segment_tempos': segment_tempos,
                'tempo_range': (float(min(segment_tempos)), float(max(segment_tempos)))
            }
            
        except Exception as e:
            logger.warning(f"Tempo stability analysis failed: {e}")
            return {'is_stable': True, 'tempo_variance': 0.0, 'tempo_drift': 0.0}
    
    def _categorize_tempo(self, tempo: float) -> str:
        """Categorize tempo into musical terms"""
        if tempo < 70:
            return "very_slow"
        elif tempo < 90:
            return "slow"
        elif tempo < 110:
            return "moderate"
        elif tempo < 130:
            return "medium_fast"
        elif tempo < 150:
            return "fast"
        elif tempo < 180:
            return "very_fast"
        else:
            return "extremely_fast"
    
    def _create_empty_result(self) -> Dict[str, any]:
        """Create empty result when tempo detection fails"""
        return {
            'primary_tempo': None,
            'confidence': 0.0,
            'tempo_methods': {
                'standard': None,
                'onset_based': None,
                'autocorrelation': None,
                'multiscale': None
            },
            'beat_analysis': {'beat_count': 0, 'beat_consistency': 0.0, 'avg_beat_interval': 0.0},
            'stability_analysis': {'is_stable': False, 'tempo_variance': 0.0, 'tempo_drift': 0.0},
            'tempo_category': 'unknown',
            'all_detected_tempos': []
        }
    
    def suggest_tempo_adjustments(self, tempo_result: Dict[str, any]) -> Dict[str, any]:
        """Suggest tempo adjustments for better mix compatibility"""
        primary_tempo = tempo_result.get('primary_tempo')
        confidence = tempo_result.get('confidence', 0.0)
        
        if primary_tempo is None:
            return {'suggestions': [], 'reasoning': 'No valid tempo detected'}
        
        suggestions = []
        reasoning = []
        
        # Common DJ-friendly tempos
        dj_friendly_tempos = [120, 124, 126, 128, 130, 132, 134, 140]
        
        # Find closest DJ-friendly tempo
        closest_dj_tempo = min(dj_friendly_tempos, key=lambda x: abs(x - primary_tempo))
        
        if abs(primary_tempo - closest_dj_tempo) > 3:
            suggestions.append({
                'type': 'dj_friendly',
                'target_tempo': closest_dj_tempo,
                'adjustment': closest_dj_tempo - primary_tempo,
                'percentage_change': ((closest_dj_tempo - primary_tempo) / primary_tempo) * 100
            })
            reasoning.append(f"Adjust to {closest_dj_tempo} BPM for better DJ/mix compatibility")
        
        # Confidence-based suggestions
        if confidence < 0.7:
            suggestions.append({
                'type': 'confidence_improvement',
                'suggestion': 'Consider manual tempo adjustment - low detection confidence',
                'current_confidence': confidence
            })
            reasoning.append("Low tempo detection confidence suggests manual verification needed")
        
        # Genre-specific suggestions
        genre_suggestions = self._get_genre_tempo_suggestions(primary_tempo)
        if genre_suggestions:
            suggestions.extend(genre_suggestions)
            reasoning.extend([s['reasoning'] for s in genre_suggestions])
        
        return {
            'suggestions': suggestions,
            'reasoning': reasoning,
            'current_tempo': primary_tempo,
            'confidence': confidence
        }
    
    def _get_genre_tempo_suggestions(self, tempo: float) -> List[Dict[str, any]]:
        """Get genre-specific tempo suggestions"""
        suggestions = []
        
        # Define genre tempo ranges
        genre_ranges = {
            'house': (120, 130),
            'techno': (120, 135),
            'trance': (128, 140),
            'drum_and_bass': (160, 180),
            'dubstep': (140, 150),
            'hip_hop': (80, 100),
            'pop': (110, 130),
            'rock': (110, 140)
        }
        
        # Find matching genres
        matching_genres = [genre for genre, (min_t, max_t) in genre_ranges.items() 
                          if min_t <= tempo <= max_t]
        
        if matching_genres:
            suggestions.append({
                'type': 'genre_compatibility',
                'matching_genres': matching_genres,
                'reasoning': f"Tempo matches {', '.join(matching_genres)} genre characteristics"
            })
        
        return suggestions