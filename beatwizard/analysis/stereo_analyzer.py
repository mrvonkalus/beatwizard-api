"""
Enhanced Stereo Analysis - Professional stereo imaging analysis
Advanced stereo field analysis with width, phase correlation, and balance metrics
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class StereoAnalyzer:
    """
    Professional stereo analysis with comprehensive stereo imaging metrics
    Analyzes stereo width, phase correlation, balance, and spatial characteristics
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """
        Initialize the stereo analyzer
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        self.frame_size = audio_settings.DEFAULT_FRAME_SIZE
        
        logger.debug("StereoAnalyzer initialized")
    
    def analyze_stereo_image(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive stereo imaging analysis
        
        Args:
            audio: Stereo audio data (2 channels)
            
        Returns:
            Dictionary with stereo analysis results
        """
        # Ensure stereo input
        if len(audio.shape) != 2 or audio.shape[0] != 2:
            logger.warning("Input is not stereo - creating pseudo-stereo analysis")
            return self._analyze_mono_as_stereo(audio)
        
        logger.debug("Starting comprehensive stereo analysis")
        
        left_channel = audio[0]
        right_channel = audio[1]
        
        # Phase correlation analysis
        phase_analysis = self._analyze_phase_correlation(left_channel, right_channel)
        
        # Stereo width analysis
        width_analysis = self._analyze_stereo_width(left_channel, right_channel)
        
        # Balance analysis
        balance_analysis = self._analyze_stereo_balance(left_channel, right_channel)
        
        # Mid-Side analysis
        ms_analysis = self._analyze_mid_side(left_channel, right_channel)
        
        # Frequency-dependent stereo analysis
        frequency_stereo_analysis = self._analyze_frequency_dependent_stereo(left_channel, right_channel)
        
        # Stereo imaging quality assessment
        imaging_quality = self._assess_stereo_imaging_quality(phase_analysis, width_analysis, balance_analysis)
        
        # Professional mixing insights
        mixing_insights = self._generate_stereo_mixing_insights(phase_analysis, width_analysis, balance_analysis, ms_analysis)
        
        result = {
            'phase_analysis': phase_analysis,
            'width_analysis': width_analysis,
            'balance_analysis': balance_analysis,
            'mid_side_analysis': ms_analysis,
            'frequency_stereo_analysis': frequency_stereo_analysis,
            'imaging_quality': imaging_quality,
            'mixing_insights': mixing_insights,
            'overall_assessment': self._assess_overall_stereo_quality(phase_analysis, width_analysis, balance_analysis, imaging_quality)
        }
        
        logger.info("Stereo analysis completed")
        
        return result
    
    def _analyze_phase_correlation(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze phase correlation between left and right channels"""
        try:
            # Overall phase correlation
            correlation_coefficient = np.corrcoef(left, right)[0, 1]
            
            # Frame-wise phase correlation for temporal analysis
            frame_correlations = []
            frame_size = self.frame_size
            hop_size = self.hop_length
            
            for start in range(0, len(left) - frame_size, hop_size):
                left_frame = left[start:start + frame_size]
                right_frame = right[start:start + frame_size]
                
                if np.std(left_frame) > 1e-6 and np.std(right_frame) > 1e-6:
                    frame_corr = np.corrcoef(left_frame, right_frame)[0, 1]
                    if not np.isnan(frame_corr):
                        frame_correlations.append(frame_corr)
            
            # Phase correlation statistics
            if frame_correlations:
                mean_correlation = float(np.mean(frame_correlations))
                min_correlation = float(np.min(frame_correlations))
                max_correlation = float(np.max(frame_correlations))
                std_correlation = float(np.std(frame_correlations))
            else:
                mean_correlation = correlation_coefficient
                min_correlation = max_correlation = correlation_coefficient
                std_correlation = 0.0
            
            # Detect phase issues
            phase_issues = self._detect_phase_issues(frame_correlations)
            
            # Calculate phase coherence (frequency domain)
            phase_coherence = self._calculate_phase_coherence(left, right)
            
            return {
                'overall_correlation': float(correlation_coefficient) if not np.isnan(correlation_coefficient) else 0.0,
                'mean_correlation': mean_correlation,
                'min_correlation': min_correlation,
                'max_correlation': max_correlation,
                'correlation_stability': 1.0 - min(std_correlation, 1.0),  # Higher = more stable
                'phase_issues': phase_issues,
                'phase_coherence': phase_coherence,
                'correlation_quality': self._assess_phase_correlation_quality(mean_correlation),
                'frame_correlations': frame_correlations[:100]  # Limit for storage
            }
            
        except Exception as e:
            logger.error(f"Phase correlation analysis failed: {e}")
            return {
                'overall_correlation': 0.0,
                'mean_correlation': 0.0,
                'correlation_quality': 'unknown',
                'phase_issues': []
            }
    
    def _calculate_phase_coherence(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Calculate frequency-dependent phase coherence"""
        try:
            # Cross-power spectral density
            f, Pxy = signal.csd(left, right, fs=self.sample_rate, nperseg=2048)
            
            # Auto-power spectral densities
            _, Pxx = signal.welch(left, fs=self.sample_rate, nperseg=2048)
            _, Pyy = signal.welch(right, fs=self.sample_rate, nperseg=2048)
            
            # Coherence
            coherence = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-10)
            
            # Average coherence in different frequency bands
            frequency_bands = {
                'low': (20, 250),
                'mid': (250, 4000),
                'high': (4000, 20000)
            }
            
            band_coherence = {}
            for band_name, (low_freq, high_freq) in frequency_bands.items():
                band_indices = np.where((f >= low_freq) & (f <= high_freq))[0]
                if len(band_indices) > 0:
                    band_coherence[band_name] = float(np.mean(coherence[band_indices]))
                else:
                    band_coherence[band_name] = 0.0
            
            return {
                'overall_coherence': float(np.mean(coherence)),
                'band_coherence': band_coherence,
                'frequencies': f.tolist()[:100],  # Limit for storage
                'coherence_values': coherence.tolist()[:100]
            }
            
        except Exception as e:
            logger.warning(f"Phase coherence calculation failed: {e}")
            return {'overall_coherence': 0.0, 'band_coherence': {}}
    
    def _detect_phase_issues(self, correlations: List[float]) -> List[Dict[str, any]]:
        """Detect potential phase issues"""
        issues = []
        
        if not correlations:
            return issues
        
        # Out of phase detection (correlation < 0)
        negative_correlations = [c for c in correlations if c < 0]
        if negative_correlations:
            percentage = len(negative_correlations) / len(correlations) * 100
            if percentage > 5:  # More than 5% out of phase
                issues.append({
                    'type': 'out_of_phase',
                    'severity': 'high' if percentage > 20 else 'medium',
                    'percentage': float(percentage),
                    'description': f'{percentage:.1f}% of audio shows phase cancellation'
                })
        
        # Low correlation detection (mono compatibility issues)
        low_correlations = [c for c in correlations if 0 <= c < 0.3]
        if low_correlations:
            percentage = len(low_correlations) / len(correlations) * 100
            if percentage > 10:
                issues.append({
                    'type': 'low_correlation',
                    'severity': 'medium' if percentage > 30 else 'low',
                    'percentage': float(percentage),
                    'description': f'{percentage:.1f}% of audio has poor mono compatibility'
                })
        
        # Extreme correlation instability
        if len(correlations) > 1:
            correlation_range = max(correlations) - min(correlations)
            if correlation_range > 1.5:  # Very unstable
                issues.append({
                    'type': 'correlation_instability',
                    'severity': 'medium',
                    'range': float(correlation_range),
                    'description': 'Phase correlation varies significantly throughout track'
                })
        
        return issues
    
    def _assess_phase_correlation_quality(self, correlation: float) -> str:
        """Assess phase correlation quality"""
        if correlation >= 0.8:
            return 'excellent'
        elif correlation >= 0.6:
            return 'good'
        elif correlation >= 0.3:
            return 'acceptable'
        elif correlation >= 0:
            return 'poor'
        else:
            return 'out_of_phase'
    
    def _analyze_stereo_width(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze stereo width and imaging"""
        try:
            # Convert to Mid-Side
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Calculate stereo width using Mid-Side analysis
            mid_energy = np.mean(mid**2)
            side_energy = np.mean(side**2)
            
            # Stereo width ratio
            if mid_energy > 0:
                width_ratio = side_energy / mid_energy
                # Convert to more intuitive scale (0 = mono, 1 = normal stereo, >1 = wide)
                stereo_width = np.sqrt(width_ratio)
            else:
                stereo_width = 0.0
            
            # LCR (Left-Center-Right) analysis
            lcr_analysis = self._analyze_lcr_distribution(left, right)
            
            # Frequency-dependent width analysis
            frequency_width = self._analyze_frequency_dependent_width(left, right)
            
            # Stereo imaging stability
            width_stability = self._analyze_width_stability(left, right)
            
            # Assess width quality
            width_quality = self._assess_stereo_width_quality(stereo_width)
            
            return {
                'stereo_width': float(stereo_width),
                'width_ratio': float(width_ratio) if 'width_ratio' in locals() else 0.0,
                'mid_energy': float(mid_energy),
                'side_energy': float(side_energy),
                'lcr_analysis': lcr_analysis,
                'frequency_width': frequency_width,
                'width_stability': width_stability,
                'width_quality': width_quality,
                'width_category': self._categorize_stereo_width(stereo_width)
            }
            
        except Exception as e:
            logger.error(f"Stereo width analysis failed: {e}")
            return {
                'stereo_width': 0.0,
                'width_quality': 'unknown',
                'width_category': 'unknown'
            }
    
    def _analyze_lcr_distribution(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze Left-Center-Right distribution"""
        try:
            # Calculate energy in different positions
            total_energy = np.mean(left**2) + np.mean(right**2)
            
            if total_energy > 0:
                left_energy = np.mean(left**2) / total_energy
                right_energy = np.mean(right**2) / total_energy
                
                # Center energy (common content)
                center_energy = np.mean((left * right)) / (total_energy / 2 + 1e-10)
                center_energy = max(0, center_energy)  # Ensure non-negative
                
                # Side energy (difference content)
                side_energy = 1.0 - center_energy
                
                return {
                    'left_percentage': float(left_energy * 100),
                    'right_percentage': float(right_energy * 100),
                    'center_percentage': float(center_energy * 100),
                    'side_percentage': float(side_energy * 100),
                    'lr_balance': float(left_energy - right_energy),  # Positive = left-heavy
                    'distribution_quality': self._assess_lcr_distribution_quality(left_energy, right_energy, center_energy)
                }
            else:
                return {
                    'left_percentage': 0.0,
                    'right_percentage': 0.0,
                    'center_percentage': 0.0,
                    'side_percentage': 0.0,
                    'lr_balance': 0.0,
                    'distribution_quality': 'silent'
                }
                
        except Exception as e:
            logger.warning(f"LCR distribution analysis failed: {e}")
            return {'distribution_quality': 'unknown'}
    
    def _analyze_frequency_dependent_width(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze stereo width across frequency bands"""
        try:
            # Get spectrograms
            left_stft = librosa.stft(left, hop_length=self.hop_length)
            right_stft = librosa.stft(right, hop_length=self.hop_length)
            
            # Convert to Mid-Side in frequency domain
            mid_stft = (left_stft + right_stft) / 2
            side_stft = (left_stft - right_stft) / 2
            
            frequencies = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Calculate width for each frequency bin
            mid_magnitude = np.abs(mid_stft)
            side_magnitude = np.abs(side_stft)
            
            # Width ratio for each frequency
            width_per_freq = np.zeros(len(frequencies))
            for i in range(len(frequencies)):
                mid_energy = np.mean(mid_magnitude[i, :]**2)
                side_energy = np.mean(side_magnitude[i, :]**2)
                
                if mid_energy > 0:
                    width_per_freq[i] = np.sqrt(side_energy / mid_energy)
            
            # Average width in frequency bands
            frequency_bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 8000),
                'brilliance': (8000, 20000)
            }
            
            band_widths = {}
            for band_name, (low_freq, high_freq) in frequency_bands.items():
                band_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
                if len(band_indices) > 0:
                    band_widths[band_name] = float(np.mean(width_per_freq[band_indices]))
                else:
                    band_widths[band_name] = 0.0
            
            return {
                'band_widths': band_widths,
                'width_variance': float(np.var(width_per_freq)),
                'frequency_consistency': self._assess_frequency_width_consistency(band_widths)
            }
            
        except Exception as e:
            logger.warning(f"Frequency-dependent width analysis failed: {e}")
            return {'band_widths': {}, 'frequency_consistency': 'unknown'}
    
    def _analyze_width_stability(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze temporal stability of stereo width"""
        try:
            frame_size = self.frame_size
            hop_size = self.hop_length
            
            frame_widths = []
            
            for start in range(0, len(left) - frame_size, hop_size):
                left_frame = left[start:start + frame_size]
                right_frame = right[start:start + frame_size]
                
                # Calculate frame width
                mid_frame = (left_frame + right_frame) / 2
                side_frame = (left_frame - right_frame) / 2
                
                mid_energy = np.mean(mid_frame**2)
                side_energy = np.mean(side_frame**2)
                
                if mid_energy > 0:
                    frame_width = np.sqrt(side_energy / mid_energy)
                    frame_widths.append(frame_width)
            
            if frame_widths:
                width_stability = 1.0 - (np.std(frame_widths) / (np.mean(frame_widths) + 1e-10))
                width_stability = max(0.0, min(1.0, width_stability))
                
                return {
                    'mean_width': float(np.mean(frame_widths)),
                    'width_variance': float(np.var(frame_widths)),
                    'stability_score': float(width_stability),
                    'stability_quality': self._assess_stability_quality(width_stability),
                    'frame_widths': frame_widths[:50]  # Limit for storage
                }
            else:
                return {
                    'mean_width': 0.0,
                    'width_variance': 0.0,
                    'stability_score': 0.0,
                    'stability_quality': 'unknown'
                }
                
        except Exception as e:
            logger.warning(f"Width stability analysis failed: {e}")
            return {'stability_quality': 'unknown'}
    
    def _assess_stereo_width_quality(self, width: float) -> str:
        """Assess stereo width quality"""
        if width < 0.1:
            return 'mono'
        elif width < 0.5:
            return 'narrow'
        elif width <= 1.2:
            return 'good'
        elif width <= 2.0:
            return 'wide'
        else:
            return 'very_wide'
    
    def _categorize_stereo_width(self, width: float) -> str:
        """Categorize stereo width"""
        if width < 0.1:
            return 'mono'
        elif width < 0.7:
            return 'narrow_stereo'
        elif width <= 1.3:
            return 'normal_stereo'
        elif width <= 2.0:
            return 'wide_stereo'
        else:
            return 'extra_wide'
    
    def _analyze_stereo_balance(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze stereo balance and panning"""
        try:
            # Overall balance
            left_rms = np.sqrt(np.mean(left**2))
            right_rms = np.sqrt(np.mean(right**2))
            
            total_rms = left_rms + right_rms
            if total_rms > 0:
                left_percentage = (left_rms / total_rms) * 100
                right_percentage = (right_rms / total_rms) * 100
                balance_offset = left_percentage - 50.0  # Positive = left-heavy
            else:
                left_percentage = right_percentage = 50.0
                balance_offset = 0.0
            
            # Temporal balance analysis
            temporal_balance = self._analyze_temporal_balance(left, right)
            
            # Frequency-dependent balance
            frequency_balance = self._analyze_frequency_dependent_balance(left, right)
            
            # Assess balance quality
            balance_quality = self._assess_balance_quality(abs(balance_offset))
            
            return {
                'left_percentage': float(left_percentage),
                'right_percentage': float(right_percentage),
                'balance_offset': float(balance_offset),
                'balance_quality': balance_quality,
                'temporal_balance': temporal_balance,
                'frequency_balance': frequency_balance,
                'is_centered': abs(balance_offset) < 5.0  # Within 5% is considered centered
            }
            
        except Exception as e:
            logger.error(f"Stereo balance analysis failed: {e}")
            return {
                'balance_offset': 0.0,
                'balance_quality': 'unknown',
                'is_centered': True
            }
    
    def _analyze_temporal_balance(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze balance changes over time"""
        try:
            frame_size = self.frame_size
            hop_size = self.hop_length
            
            frame_balances = []
            
            for start in range(0, len(left) - frame_size, hop_size):
                left_frame = left[start:start + frame_size]
                right_frame = right[start:start + frame_size]
                
                left_rms = np.sqrt(np.mean(left_frame**2))
                right_rms = np.sqrt(np.mean(right_frame**2))
                
                total_rms = left_rms + right_rms
                if total_rms > 0:
                    balance = (left_rms / total_rms - 0.5) * 100  # -50 to +50
                    frame_balances.append(balance)
            
            if frame_balances:
                return {
                    'mean_balance': float(np.mean(frame_balances)),
                    'balance_variance': float(np.var(frame_balances)),
                    'balance_stability': 1.0 - min(np.std(frame_balances) / 25.0, 1.0),  # Normalize by max expected std
                    'max_left_bias': float(np.max(frame_balances)),
                    'max_right_bias': float(np.min(frame_balances)),
                    'frame_balances': frame_balances[:50]  # Limit for storage
                }
            else:
                return {
                    'mean_balance': 0.0,
                    'balance_variance': 0.0,
                    'balance_stability': 1.0
                }
                
        except Exception as e:
            logger.warning(f"Temporal balance analysis failed: {e}")
            return {'balance_stability': 1.0}
    
    def _analyze_frequency_dependent_balance(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze balance across frequency bands"""
        try:
            # Get power spectral densities
            f, left_psd = signal.welch(left, fs=self.sample_rate, nperseg=2048)
            _, right_psd = signal.welch(right, fs=self.sample_rate, nperseg=2048)
            
            frequency_bands = {
                'bass': (20, 250),
                'mid': (250, 4000),
                'high': (4000, 20000)
            }
            
            band_balances = {}
            
            for band_name, (low_freq, high_freq) in frequency_bands.items():
                band_indices = np.where((f >= low_freq) & (f <= high_freq))[0]
                
                if len(band_indices) > 0:
                    left_energy = np.sum(left_psd[band_indices])
                    right_energy = np.sum(right_psd[band_indices])
                    
                    total_energy = left_energy + right_energy
                    if total_energy > 0:
                        balance = (left_energy / total_energy - 0.5) * 100
                        band_balances[band_name] = float(balance)
                    else:
                        band_balances[band_name] = 0.0
                else:
                    band_balances[band_name] = 0.0
            
            return {
                'band_balances': band_balances,
                'frequency_balance_consistency': self._assess_frequency_balance_consistency(band_balances)
            }
            
        except Exception as e:
            logger.warning(f"Frequency-dependent balance analysis failed: {e}")
            return {'band_balances': {}}
    
    def _assess_balance_quality(self, balance_offset: float) -> str:
        """Assess balance quality"""
        if balance_offset < 2.0:
            return 'perfectly_centered'
        elif balance_offset < 5.0:
            return 'well_centered'
        elif balance_offset < 10.0:
            return 'slightly_off_center'
        elif balance_offset < 20.0:
            return 'off_center'
        else:
            return 'heavily_biased'
    
    def _analyze_mid_side(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Comprehensive Mid-Side analysis"""
        try:
            # Convert to Mid-Side
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Mid-Side energy analysis
            mid_rms = np.sqrt(np.mean(mid**2))
            side_rms = np.sqrt(np.mean(side**2))
            
            total_energy = mid_rms**2 + side_rms**2
            if total_energy > 0:
                mid_percentage = (mid_rms**2 / total_energy) * 100
                side_percentage = (side_rms**2 / total_energy) * 100
            else:
                mid_percentage = side_percentage = 50.0
            
            # Mid-Side ratio
            if mid_rms > 0:
                ms_ratio = side_rms / mid_rms
            else:
                ms_ratio = 0.0
            
            # Mid-Side correlation
            if np.std(mid) > 1e-6 and np.std(side) > 1e-6:
                ms_correlation = abs(np.corrcoef(mid, side)[0, 1])
            else:
                ms_correlation = 0.0
            
            # Mid-Side frequency analysis
            ms_frequency_analysis = self._analyze_ms_frequency_content(mid, side)
            
            return {
                'mid_percentage': float(mid_percentage),
                'side_percentage': float(side_percentage),
                'mid_side_ratio': float(ms_ratio),
                'mid_side_correlation': float(ms_correlation),
                'mid_rms': float(mid_rms),
                'side_rms': float(side_rms),
                'ms_frequency_analysis': ms_frequency_analysis,
                'stereo_information': self._assess_stereo_information(mid_percentage, side_percentage)
            }
            
        except Exception as e:
            logger.error(f"Mid-Side analysis failed: {e}")
            return {
                'mid_percentage': 50.0,
                'side_percentage': 50.0,
                'stereo_information': 'unknown'
            }
    
    def _analyze_ms_frequency_content(self, mid: np.ndarray, side: np.ndarray) -> Dict[str, any]:
        """Analyze frequency content of Mid and Side channels"""
        try:
            # Get spectrograms
            mid_stft = librosa.stft(mid, hop_length=self.hop_length)
            side_stft = librosa.stft(side, hop_length=self.hop_length)
            
            # Calculate energy distribution
            mid_energy = np.mean(np.abs(mid_stft)**2, axis=1)
            side_energy = np.mean(np.abs(side_stft)**2, axis=1)
            
            frequencies = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Find frequency ranges where side dominates
            side_dominant_freqs = []
            for i, freq in enumerate(frequencies):
                if side_energy[i] > mid_energy[i]:
                    side_dominant_freqs.append(freq)
            
            return {
                'side_dominant_frequencies': side_dominant_freqs[:20],  # Limit for storage
                'mid_energy_distribution': mid_energy.tolist()[:50],
                'side_energy_distribution': side_energy.tolist()[:50],
                'stereo_content_frequency_range': (
                    float(min(side_dominant_freqs)) if side_dominant_freqs else 0.0,
                    float(max(side_dominant_freqs)) if side_dominant_freqs else 0.0
                )
            }
            
        except Exception as e:
            logger.warning(f"M-S frequency analysis failed: {e}")
            return {}
    
    def _assess_stereo_information(self, mid_percentage: float, side_percentage: float) -> str:
        """Assess stereo information content"""
        if side_percentage < 5:
            return 'mono'
        elif side_percentage < 15:
            return 'minimal_stereo'
        elif side_percentage < 35:
            return 'moderate_stereo'
        elif side_percentage < 55:
            return 'rich_stereo'
        else:
            return 'very_wide_stereo'
    
    def _analyze_frequency_dependent_stereo(self, left: np.ndarray, right: np.ndarray) -> Dict[str, any]:
        """Analyze stereo characteristics across frequency ranges"""
        try:
            # This combines width and balance analysis across frequencies
            # Using STFT for frequency-time analysis
            left_stft = librosa.stft(left, hop_length=self.hop_length)
            right_stft = librosa.stft(right, hop_length=self.hop_length)
            
            frequencies = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Define professional frequency bands
            frequency_bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 8000),
                'brilliance': (8000, 20000)
            }
            
            band_stereo_analysis = {}
            
            for band_name, (low_freq, high_freq) in frequency_bands.items():
                band_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
                
                if len(band_indices) > 0:
                    # Extract band content
                    left_band = np.sum(np.abs(left_stft[band_indices, :]), axis=0)
                    right_band = np.sum(np.abs(right_stft[band_indices, :]), axis=0)
                    
                    # Calculate band-specific metrics
                    band_correlation = np.corrcoef(left_band, right_band)[0, 1] if np.std(left_band) > 1e-6 and np.std(right_band) > 1e-6 else 1.0
                    
                    # Band width
                    mid_band = (left_band + right_band) / 2
                    side_band = (left_band - right_band) / 2
                    
                    mid_energy = np.mean(mid_band**2)
                    side_energy = np.mean(side_band**2)
                    
                    band_width = np.sqrt(side_energy / (mid_energy + 1e-10))
                    
                    # Band balance
                    left_energy = np.mean(left_band**2)
                    right_energy = np.mean(right_band**2)
                    total_energy = left_energy + right_energy
                    
                    if total_energy > 0:
                        band_balance = (left_energy / total_energy - 0.5) * 100
                    else:
                        band_balance = 0.0
                    
                    band_stereo_analysis[band_name] = {
                        'correlation': float(band_correlation) if not np.isnan(band_correlation) else 1.0,
                        'width': float(band_width),
                        'balance_offset': float(band_balance),
                        'frequency_range': (low_freq, high_freq)
                    }
                else:
                    band_stereo_analysis[band_name] = {
                        'correlation': 1.0,
                        'width': 0.0,
                        'balance_offset': 0.0,
                        'frequency_range': (low_freq, high_freq)
                    }
            
            return band_stereo_analysis
            
        except Exception as e:
            logger.error(f"Frequency-dependent stereo analysis failed: {e}")
            return {}
    
    def _assess_stereo_imaging_quality(self, phase_analysis: Dict[str, any], width_analysis: Dict[str, any], balance_analysis: Dict[str, any]) -> Dict[str, any]:
        """Assess overall stereo imaging quality"""
        try:
            # Individual quality scores
            phase_quality = phase_analysis.get('correlation_quality', 'unknown')
            width_quality = width_analysis.get('width_quality', 'unknown')
            balance_quality = balance_analysis.get('balance_quality', 'unknown')
            
            # Convert to numeric scores
            quality_scores = {
                'excellent': 5, 'good': 4, 'acceptable': 3, 'poor': 2, 'out_of_phase': 1, 'unknown': 0
            }
            
            phase_score = quality_scores.get(phase_quality, 0)
            width_score = quality_scores.get(width_quality, 0)
            balance_score = quality_scores.get(balance_quality, 0)
            
            # Overall score (weighted average)
            overall_score = (phase_score * 0.4 + width_score * 0.3 + balance_score * 0.3)
            
            # Convert back to quality rating
            if overall_score >= 4.5:
                overall_quality = 'excellent'
            elif overall_score >= 3.5:
                overall_quality = 'good'
            elif overall_score >= 2.5:
                overall_quality = 'acceptable'
            elif overall_score >= 1.5:
                overall_quality = 'poor'
            else:
                overall_quality = 'problematic'
            
            # Specific quality aspects
            quality_aspects = {
                'phase_quality': phase_quality,
                'width_quality': width_quality,
                'balance_quality': balance_quality,
                'mono_compatibility': 'good' if phase_analysis.get('mean_correlation', 0) > 0.6 else 'poor',
                'stereo_effectiveness': 'good' if width_analysis.get('stereo_width', 0) > 0.5 else 'limited'
            }
            
            return {
                'overall_quality': overall_quality,
                'overall_score': float(overall_score),
                'quality_aspects': quality_aspects,
                'professional_grade': overall_quality in ['excellent', 'good'],
                'needs_attention': overall_quality in ['poor', 'problematic']
            }
            
        except Exception as e:
            logger.error(f"Stereo imaging quality assessment failed: {e}")
            return {
                'overall_quality': 'unknown',
                'professional_grade': False,
                'needs_attention': True
            }
    
    def _generate_stereo_mixing_insights(self, phase_analysis: Dict[str, any], width_analysis: Dict[str, any], balance_analysis: Dict[str, any], ms_analysis: Dict[str, any]) -> Dict[str, any]:
        """Generate professional stereo mixing insights"""
        insights = {
            'recommendations': [],
            'warnings': [],
            'enhancements': [],
            'technical_notes': []
        }
        
        try:
            # Phase-related insights
            phase_issues = phase_analysis.get('phase_issues', [])
            for issue in phase_issues:
                if issue['type'] == 'out_of_phase':
                    insights['warnings'].append(f"Phase cancellation detected in {issue['percentage']:.1f}% of audio")
                    insights['recommendations'].append("Check for phase inversion or alignment issues")
                elif issue['type'] == 'low_correlation':
                    insights['warnings'].append("Poor mono compatibility detected")
                    insights['recommendations'].append("Consider checking stereo spread and phase relationships")
            
            # Width-related insights
            stereo_width = width_analysis.get('stereo_width', 0)
            if stereo_width < 0.3:
                insights['recommendations'].append("Stereo image is narrow - consider stereo enhancement")
            elif stereo_width > 2.0:
                insights['warnings'].append("Extremely wide stereo image may cause mono compatibility issues")
                insights['recommendations'].append("Consider reducing stereo width for better compatibility")
            
            # Balance-related insights
            balance_offset = abs(balance_analysis.get('balance_offset', 0))
            if balance_offset > 10:
                insights['warnings'].append(f"Significant stereo imbalance detected ({balance_offset:.1f}% offset)")
                insights['recommendations'].append("Adjust channel balance for better center image")
            
            # Mid-Side insights
            side_percentage = ms_analysis.get('side_percentage', 50)
            if side_percentage < 10:
                insights['enhancements'].append("Limited stereo information - consider adding stereo elements")
            elif side_percentage > 60:
                insights['technical_notes'].append("High side content - verify mono compatibility")
            
            # Professional mixing tips
            phase_correlation = phase_analysis.get('mean_correlation', 1.0)
            if phase_correlation > 0.8:
                insights['enhancements'].append("Excellent phase correlation - good for mastering")
            
            if width_analysis.get('width_quality') == 'good' and balance_analysis.get('balance_quality') in ['perfectly_centered', 'well_centered']:
                insights['enhancements'].append("Professional stereo imaging - ready for mastering")
            
            return insights
            
        except Exception as e:
            logger.error(f"Stereo mixing insights generation failed: {e}")
            return insights
    
    def _assess_overall_stereo_quality(self, phase_analysis: Dict[str, any], width_analysis: Dict[str, any], balance_analysis: Dict[str, any], imaging_quality: Dict[str, any]) -> Dict[str, any]:
        """Provide overall stereo quality assessment"""
        try:
            overall_quality = imaging_quality.get('overall_quality', 'unknown')
            professional_grade = imaging_quality.get('professional_grade', False)
            
            # Count issues
            issues = []
            phase_issues = phase_analysis.get('phase_issues', [])
            if phase_issues:
                issues.extend([issue['type'] for issue in phase_issues])
            
            if not balance_analysis.get('is_centered', True):
                issues.append('balance_offset')
            
            # Stereo characteristics summary
            characteristics = {
                'phase_correlation': phase_analysis.get('mean_correlation', 0.0),
                'stereo_width': width_analysis.get('stereo_width', 0.0),
                'balance_offset': balance_analysis.get('balance_offset', 0.0),
                'mono_compatible': phase_analysis.get('mean_correlation', 0) > 0.6,
                'stereo_effective': width_analysis.get('stereo_width', 0) > 0.5
            }
            
            # Professional readiness
            ready_for_mastering = (
                professional_grade and 
                len(issues) <= 1 and
                characteristics['mono_compatible']
            )
            
            # Summary recommendations
            if overall_quality == 'excellent':
                summary = "Professional stereo imaging with excellent characteristics"
            elif overall_quality == 'good':
                summary = "Good stereo imaging with minor optimization opportunities"
            elif overall_quality == 'acceptable':
                summary = "Acceptable stereo imaging but improvements recommended"
            else:
                summary = "Stereo imaging needs significant attention"
            
            return {
                'overall_quality': overall_quality,
                'professional_grade': professional_grade,
                'ready_for_mastering': ready_for_mastering,
                'issues_detected': issues,
                'issue_count': len(issues),
                'characteristics': characteristics,
                'summary': summary,
                'confidence_score': imaging_quality.get('overall_score', 0.0) / 5.0  # Normalize to 0-1
            }
            
        except Exception as e:
            logger.error(f"Overall stereo quality assessment failed: {e}")
            return {
                'overall_quality': 'unknown',
                'professional_grade': False,
                'ready_for_mastering': False,
                'summary': 'Analysis failed'
            }
    
    def _analyze_mono_as_stereo(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze mono audio with stereo placeholders"""
        logger.info("Analyzing mono audio - creating pseudo-stereo analysis")
        
        return {
            'phase_analysis': {
                'overall_correlation': 1.0,
                'correlation_quality': 'mono',
                'phase_issues': []
            },
            'width_analysis': {
                'stereo_width': 0.0,
                'width_quality': 'mono',
                'width_category': 'mono'
            },
            'balance_analysis': {
                'balance_offset': 0.0,
                'balance_quality': 'perfectly_centered',
                'is_centered': True
            },
            'mid_side_analysis': {
                'mid_percentage': 100.0,
                'side_percentage': 0.0,
                'stereo_information': 'mono'
            },
            'imaging_quality': {
                'overall_quality': 'mono',
                'professional_grade': False,
                'needs_attention': False
            },
            'overall_assessment': {
                'overall_quality': 'mono',
                'professional_grade': False,
                'ready_for_mastering': True,  # Mono can be fine for mastering
                'summary': 'Mono audio - no stereo imaging present'
            }
        }
    
    # Helper methods for assessment functions
    def _assess_lcr_distribution_quality(self, left_energy: float, right_energy: float, center_energy: float) -> str:
        """Assess Left-Center-Right distribution quality"""
        balance = abs(left_energy - right_energy)
        if balance < 0.1 and center_energy > 0.3:
            return 'well_distributed'
        elif balance < 0.2:
            return 'acceptable'
        else:
            return 'imbalanced'
    
    def _assess_frequency_width_consistency(self, band_widths: Dict[str, float]) -> str:
        """Assess consistency of width across frequency bands"""
        if not band_widths:
            return 'unknown'
        
        widths = list(band_widths.values())
        if len(widths) > 1:
            width_variance = np.var(widths)
            if width_variance < 0.1:
                return 'very_consistent'
            elif width_variance < 0.3:
                return 'consistent'
            elif width_variance < 0.6:
                return 'moderately_consistent'
            else:
                return 'inconsistent'
        else:
            return 'single_band'
    
    def _assess_stability_quality(self, stability_score: float) -> str:
        """Assess stability quality"""
        if stability_score >= 0.9:
            return 'very_stable'
        elif stability_score >= 0.7:
            return 'stable'
        elif stability_score >= 0.5:
            return 'moderately_stable'
        else:
            return 'unstable'
    
    def _assess_frequency_balance_consistency(self, band_balances: Dict[str, float]) -> str:
        """Assess consistency of balance across frequency bands"""
        if not band_balances:
            return 'unknown'
        
        balances = list(band_balances.values())
        if len(balances) > 1:
            balance_variance = np.var(balances)
            if balance_variance < 25:  # Low variance in balance offsets
                return 'consistent'
            elif balance_variance < 100:
                return 'moderately_consistent'
            else:
                return 'inconsistent'
        else:
            return 'single_band'