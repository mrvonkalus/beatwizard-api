"""
Enhanced Frequency Analysis - Professional 7-band EQ analysis
Advanced frequency domain analysis with professional mixing insights
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class FrequencyAnalyzer:
    """
    Professional frequency analysis with 7-band EQ breakdown
    Industry-standard frequency analysis for mixing and mastering
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """
        Initialize the frequency analyzer
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        self.frame_size = audio_settings.DEFAULT_FRAME_SIZE
        
        # Professional 7-band EQ frequency ranges
        self.frequency_bands = audio_settings.FREQUENCY_BANDS
        
        # Initialize analysis parameters
        self.window_size = 4096  # Higher resolution for frequency analysis
        self.overlap = 0.75
        
        logger.debug("FrequencyAnalyzer initialized with 7-band professional EQ")
    
    def analyze_frequency_spectrum(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive frequency spectrum analysis
        
        Args:
            audio: Audio data (mono)
            
        Returns:
            Dictionary with frequency analysis results
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        logger.debug("Starting comprehensive frequency analysis")
        
        # Overall spectral analysis
        spectral_features = self._extract_spectral_features(audio)
        
        # 7-band frequency analysis
        band_analysis = self._analyze_frequency_bands(audio)
        
        # Spectral balance analysis
        balance_analysis = self._analyze_spectral_balance(audio, band_analysis)
        
        # Dynamic frequency analysis
        dynamic_analysis = self._analyze_frequency_dynamics(audio)
        
        # Masking and clarity analysis
        clarity_analysis = self._analyze_spectral_clarity(audio)
        
        # Professional mixing insights
        mixing_insights = self._generate_mixing_insights(band_analysis, balance_analysis)
        
        result = {
            'spectral_features': spectral_features,
            'band_analysis': band_analysis,
            'balance_analysis': balance_analysis,
            'dynamic_analysis': dynamic_analysis,
            'clarity_analysis': clarity_analysis,
            'mixing_insights': mixing_insights,
            'overall_assessment': self._assess_overall_frequency_balance(band_analysis, balance_analysis)
        }
        
        logger.info("Frequency analysis completed")
        
        return result
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract comprehensive spectral features"""
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff (energy distribution)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                roll_percent=0.85
            )[0]
            
            # Spectral bandwidth (spread)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.frame_size,
                hop_length=self.hop_length
            )[0]
            
            # MFCC for timbral analysis
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_mfcc=13
            )
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            return {
                'spectral_centroid': {
                    'mean': float(np.mean(spectral_centroids)),
                    'std': float(np.std(spectral_centroids)),
                    'range': (float(np.min(spectral_centroids)), float(np.max(spectral_centroids)))
                },
                'spectral_rolloff': {
                    'mean': float(np.mean(spectral_rolloff)),
                    'std': float(np.std(spectral_rolloff))
                },
                'spectral_bandwidth': {
                    'mean': float(np.mean(spectral_bandwidth)),
                    'std': float(np.std(spectral_bandwidth))
                },
                'zero_crossing_rate': {
                    'mean': float(np.mean(zcr)),
                    'std': float(np.std(zcr))
                },
                'mfcc_features': {
                    'mean': np.mean(mfcc, axis=1).tolist(),
                    'std': np.std(mfcc, axis=1).tolist()
                },
                'spectral_contrast': {
                    'mean': np.mean(spectral_contrast, axis=1).tolist(),
                    'std': np.std(spectral_contrast, axis=1).tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return {}
    
    def _analyze_frequency_bands(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze energy distribution across 7 professional frequency bands"""
        try:
            # Calculate power spectral density
            frequencies, psd = signal.welch(
                audio,
                fs=self.sample_rate,
                window='hann',
                nperseg=self.window_size,
                noverlap=int(self.window_size * self.overlap)
            )
            
            band_analysis = {}
            total_energy = np.sum(psd)
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Find frequency indices for this band
                band_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
                
                if len(band_indices) > 0:
                    # Calculate band energy
                    band_energy = np.sum(psd[band_indices])
                    band_percentage = (band_energy / total_energy) * 100 if total_energy > 0 else 0
                    
                    # Peak frequency in band
                    band_psd = psd[band_indices]
                    band_freqs = frequencies[band_indices]
                    peak_idx = np.argmax(band_psd)
                    peak_frequency = band_freqs[peak_idx] if len(band_freqs) > 0 else low_freq
                    
                    # Band statistics
                    band_analysis[band_name] = {
                        'frequency_range': (low_freq, high_freq),
                        'energy_percentage': float(band_percentage),
                        'energy_db': float(10 * np.log10(band_energy + 1e-10)),
                        'peak_frequency': float(peak_frequency),
                        'peak_amplitude_db': float(10 * np.log10(np.max(band_psd) + 1e-10)),
                        'band_width': high_freq - low_freq,
                        'relative_strength': self._categorize_band_strength(band_percentage, band_name)
                    }
                else:
                    # Empty band
                    band_analysis[band_name] = {
                        'frequency_range': (low_freq, high_freq),
                        'energy_percentage': 0.0,
                        'energy_db': -np.inf,
                        'peak_frequency': low_freq,
                        'peak_amplitude_db': -np.inf,
                        'band_width': high_freq - low_freq,
                        'relative_strength': 'silent'
                    }
            
            return band_analysis
            
        except Exception as e:
            logger.error(f"Frequency band analysis failed: {e}")
            return {}
    
    def _categorize_band_strength(self, percentage: float, band_name: str) -> str:
        """Categorize the strength of a frequency band"""
        # Different thresholds for different bands based on typical music content
        thresholds = {
            'sub_bass': {'weak': 2, 'moderate': 8, 'strong': 15},
            'bass': {'weak': 5, 'moderate': 12, 'strong': 20},
            'low_mid': {'weak': 8, 'moderate': 15, 'strong': 25},
            'mid': {'weak': 10, 'moderate': 18, 'strong': 30},
            'high_mid': {'weak': 8, 'moderate': 15, 'strong': 25},
            'presence': {'weak': 5, 'moderate': 12, 'strong': 20},
            'brilliance': {'weak': 2, 'moderate': 8, 'strong': 15}
        }
        
        band_thresholds = thresholds.get(band_name, {'weak': 5, 'moderate': 10, 'strong': 20})
        
        if percentage < band_thresholds['weak']:
            return 'weak'
        elif percentage < band_thresholds['moderate']:
            return 'moderate'
        elif percentage < band_thresholds['strong']:
            return 'strong'
        else:
            return 'very_strong'
    
    def _analyze_spectral_balance(self, audio: np.ndarray, band_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze overall spectral balance and professional mixing standards"""
        try:
            # Calculate frequency distribution ratios
            low_energy = sum(band_analysis[band]['energy_percentage'] 
                           for band in ['sub_bass', 'bass', 'low_mid'] 
                           if band in band_analysis)
            
            mid_energy = band_analysis.get('mid', {}).get('energy_percentage', 0)
            
            high_energy = sum(band_analysis[band]['energy_percentage'] 
                            for band in ['high_mid', 'presence', 'brilliance'] 
                            if band in band_analysis)
            
            # Professional balance ratios
            total_energy = low_energy + mid_energy + high_energy
            
            if total_energy > 0:
                low_ratio = low_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
            else:
                low_ratio = mid_ratio = high_ratio = 0.0
            
            # Assess balance according to professional standards
            balance_assessment = self._assess_frequency_balance(low_ratio, mid_ratio, high_ratio)
            
            # Calculate spectral tilt (overall frequency slope)
            frequencies = [band_analysis[band]['peak_frequency'] 
                          for band in self.frequency_bands.keys() 
                          if band in band_analysis]
            energies = [band_analysis[band]['energy_db'] 
                       for band in self.frequency_bands.keys() 
                       if band in band_analysis and not np.isinf(band_analysis[band]['energy_db'])]
            
            spectral_tilt = self._calculate_spectral_tilt(frequencies, energies)
            
            return {
                'low_frequency_ratio': float(low_ratio),
                'mid_frequency_ratio': float(mid_ratio),
                'high_frequency_ratio': float(high_ratio),
                'balance_assessment': balance_assessment,
                'spectral_tilt': spectral_tilt,
                'energy_distribution': {
                    'low': float(low_energy),
                    'mid': float(mid_energy),
                    'high': float(high_energy)
                },
                'professional_balance_score': self._calculate_balance_score(low_ratio, mid_ratio, high_ratio)
            }
            
        except Exception as e:
            logger.error(f"Spectral balance analysis failed: {e}")
            return {}
    
    def _assess_frequency_balance(self, low_ratio: float, mid_ratio: float, high_ratio: float) -> Dict[str, any]:
        """Assess frequency balance according to professional mixing standards"""
        # Professional mixing guidelines (approximate)
        ideal_ranges = {
            'low': (0.25, 0.40),    # 25-40% low frequencies
            'mid': (0.35, 0.50),    # 35-50% midrange
            'high': (0.15, 0.30)    # 15-30% high frequencies
        }
        
        assessments = {}
        
        # Check each range
        if ideal_ranges['low'][0] <= low_ratio <= ideal_ranges['low'][1]:
            assessments['low_assessment'] = 'balanced'
        elif low_ratio < ideal_ranges['low'][0]:
            assessments['low_assessment'] = 'thin'
        else:
            assessments['low_assessment'] = 'heavy'
        
        if ideal_ranges['mid'][0] <= mid_ratio <= ideal_ranges['mid'][1]:
            assessments['mid_assessment'] = 'balanced'
        elif mid_ratio < ideal_ranges['mid'][0]:
            assessments['mid_assessment'] = 'hollow'
        else:
            assessments['mid_assessment'] = 'prominent'
        
        if ideal_ranges['high'][0] <= high_ratio <= ideal_ranges['high'][1]:
            assessments['high_assessment'] = 'balanced'
        elif high_ratio < ideal_ranges['high'][0]:
            assessments['high_assessment'] = 'dull'
        else:
            assessments['high_assessment'] = 'bright'
        
        # Overall assessment
        balanced_ranges = sum(1 for assessment in assessments.values() if assessment == 'balanced')
        
        if balanced_ranges >= 2:
            overall = 'well_balanced'
        elif balanced_ranges == 1:
            overall = 'moderately_balanced'
        else:
            overall = 'imbalanced'
        
        assessments['overall_balance'] = overall
        
        return assessments
    
    def _calculate_spectral_tilt(self, frequencies: List[float], energies: List[float]) -> Dict[str, any]:
        """Calculate spectral tilt (frequency slope)"""
        if len(frequencies) < 2 or len(energies) < 2:
            return {'tilt_db_per_octave': 0.0, 'tilt_direction': 'neutral'}
        
        try:
            # Convert to log scale for octave-based calculation
            log_frequencies = np.log2(np.array(frequencies))
            
            # Linear regression to find slope
            slope, intercept = np.polyfit(log_frequencies, energies, 1)
            
            # Convert slope to dB per octave
            tilt_db_per_octave = float(slope)
            
            # Categorize tilt direction
            if tilt_db_per_octave > 3:
                tilt_direction = 'bright'
            elif tilt_db_per_octave > 1:
                tilt_direction = 'slightly_bright'
            elif tilt_db_per_octave > -1:
                tilt_direction = 'neutral'
            elif tilt_db_per_octave > -3:
                tilt_direction = 'slightly_dark'
            else:
                tilt_direction = 'dark'
            
            return {
                'tilt_db_per_octave': tilt_db_per_octave,
                'tilt_direction': tilt_direction,
                'slope': float(slope),
                'intercept': float(intercept)
            }
            
        except Exception as e:
            logger.warning(f"Spectral tilt calculation failed: {e}")
            return {'tilt_db_per_octave': 0.0, 'tilt_direction': 'neutral'}
    
    def _calculate_balance_score(self, low_ratio: float, mid_ratio: float, high_ratio: float) -> float:
        """Calculate a professional balance score (0-1)"""
        # Ideal ratios for professional mixes
        ideal_low = 0.33
        ideal_mid = 0.42
        ideal_high = 0.25
        
        # Calculate deviations
        low_deviation = abs(low_ratio - ideal_low)
        mid_deviation = abs(mid_ratio - ideal_mid)
        high_deviation = abs(high_ratio - ideal_high)
        
        # Average deviation
        avg_deviation = (low_deviation + mid_deviation + high_deviation) / 3
        
        # Convert to score (lower deviation = higher score)
        score = max(0.0, 1.0 - (avg_deviation * 2))  # Scale factor of 2
        
        return float(score)
    
    def _analyze_frequency_dynamics(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze frequency content dynamics over time"""
        try:
            # Short-time Fourier transform for time-frequency analysis
            stft = librosa.stft(
                audio,
                hop_length=self.hop_length,
                n_fft=self.window_size
            )
            
            magnitude = np.abs(stft)
            frequencies = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.window_size)
            
            # Analyze dynamics for each frequency band
            band_dynamics = {}
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Find frequency indices for this band
                band_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
                
                if len(band_indices) > 0:
                    # Band energy over time
                    band_energy = np.sum(magnitude[band_indices, :], axis=0)
                    
                    # Dynamic range in this band
                    if len(band_energy) > 0 and np.max(band_energy) > 0:
                        dynamic_range = 20 * np.log10(np.max(band_energy) / (np.mean(band_energy) + 1e-10))
                        
                        # Variability (coefficient of variation)
                        variability = np.std(band_energy) / (np.mean(band_energy) + 1e-10)
                        
                        band_dynamics[band_name] = {
                            'dynamic_range_db': float(dynamic_range),
                            'variability': float(variability),
                            'peak_to_average_ratio': float(np.max(band_energy) / (np.mean(band_energy) + 1e-10))
                        }
                    else:
                        band_dynamics[band_name] = {
                            'dynamic_range_db': 0.0,
                            'variability': 0.0,
                            'peak_to_average_ratio': 1.0
                        }
            
            return band_dynamics
            
        except Exception as e:
            logger.error(f"Frequency dynamics analysis failed: {e}")
            return {}
    
    def _analyze_spectral_clarity(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze spectral clarity and potential masking issues"""
        try:
            # Calculate spectral flux (measure of spectral change)
            stft = librosa.stft(audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Spectral flux
            spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
            
            # Spectral irregularity (roughness)
            spectral_irregularity = []
            for frame in range(magnitude.shape[1]):
                frame_magnitude = magnitude[:, frame]
                if np.sum(frame_magnitude) > 0:
                    # Calculate roughness as sum of differences between adjacent bins
                    roughness = np.sum(np.abs(np.diff(frame_magnitude)))
                    normalized_roughness = roughness / np.sum(frame_magnitude)
                    spectral_irregularity.append(normalized_roughness)
            
            # Spectral clarity metrics
            clarity_metrics = {
                'spectral_flux': {
                    'mean': float(np.mean(spectral_flux)),
                    'std': float(np.std(spectral_flux))
                },
                'spectral_irregularity': {
                    'mean': float(np.mean(spectral_irregularity)) if spectral_irregularity else 0.0,
                    'std': float(np.std(spectral_irregularity)) if spectral_irregularity else 0.0
                },
                'clarity_assessment': self._assess_spectral_clarity(spectral_flux, spectral_irregularity)
            }
            
            return clarity_metrics
            
        except Exception as e:
            logger.error(f"Spectral clarity analysis failed: {e}")
            return {}
    
    def _assess_spectral_clarity(self, spectral_flux: np.ndarray, spectral_irregularity: List[float]) -> str:
        """Assess overall spectral clarity"""
        if len(spectral_flux) == 0:
            return 'unknown'
        
        # High flux = more spectral changes (could indicate clarity or chaos)
        flux_mean = np.mean(spectral_flux)
        
        # High irregularity = more rough/unclear sound
        irreg_mean = np.mean(spectral_irregularity) if spectral_irregularity else 0
        
        # Simple assessment logic
        if irreg_mean < 0.1 and flux_mean > 0:
            return 'clear'
        elif irreg_mean < 0.2:
            return 'moderately_clear'
        elif irreg_mean < 0.3:
            return 'somewhat_muddy'
        else:
            return 'muddy'
    
    def _generate_mixing_insights(self, band_analysis: Dict[str, any], balance_analysis: Dict[str, any]) -> Dict[str, any]:
        """Generate professional mixing insights and recommendations"""
        insights = {
            'eq_suggestions': [],
            'mixing_notes': [],
            'problem_frequencies': [],
            'enhancement_opportunities': []
        }
        
        try:
            # Analyze each band for potential issues and improvements
            for band_name, band_data in band_analysis.items():
                strength = band_data.get('relative_strength', 'moderate')
                percentage = band_data.get('energy_percentage', 0)
                
                # Band-specific insights
                if band_name == 'sub_bass':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"High-pass filter around 30-40Hz to remove excessive sub-bass")
                        insights['problem_frequencies'].append(f"Excessive sub-bass energy ({percentage:.1f}%)")
                    elif strength == 'weak':
                        insights['enhancement_opportunities'].append(f"Consider gentle boost around 40-60Hz for warmth")
                
                elif band_name == 'bass':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"Reduce bass around 80-150Hz to prevent muddiness")
                        insights['problem_frequencies'].append(f"Bass buildup detected ({percentage:.1f}%)")
                    elif strength == 'weak':
                        insights['enhancement_opportunities'].append(f"Bass presence could be enhanced around 80-120Hz")
                
                elif band_name == 'low_mid':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"Cut around 250-400Hz to reduce boxiness")
                        insights['problem_frequencies'].append(f"Low-mid buildup ({percentage:.1f}%)")
                
                elif band_name == 'mid':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"Gentle cut around 1-2kHz to reduce harshness")
                    elif strength == 'weak':
                        insights['enhancement_opportunities'].append(f"Midrange clarity could benefit from gentle boost around 1-3kHz")
                
                elif band_name == 'high_mid':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"Reduce presence around 2-4kHz if too forward")
                    elif strength == 'weak':
                        insights['enhancement_opportunities'].append(f"Presence boost around 2-4kHz for clarity")
                
                elif band_name == 'presence':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"High-cut around 8kHz to reduce harshness")
                    elif strength == 'weak':
                        insights['enhancement_opportunities'].append(f"Gentle high-frequency boost for air and presence")
                
                elif band_name == 'brilliance':
                    if strength == 'very_strong':
                        insights['eq_suggestions'].append(f"Gentle high-cut above 12kHz to reduce sibilance")
                    elif strength == 'weak':
                        insights['enhancement_opportunities'].append(f"Subtle brilliance boost above 10kHz for sparkle")
            
            # Overall balance insights
            balance_assessment = balance_analysis.get('balance_assessment', {})
            
            if balance_assessment.get('overall_balance') == 'imbalanced':
                insights['mixing_notes'].append("Overall frequency balance needs attention")
                
                if balance_assessment.get('low_assessment') == 'heavy':
                    insights['mixing_notes'].append("Mix is bottom-heavy - consider high-pass filtering")
                elif balance_assessment.get('low_assessment') == 'thin':
                    insights['mixing_notes'].append("Mix lacks low-end weight - check bass elements")
                
                if balance_assessment.get('high_assessment') == 'dull':
                    insights['mixing_notes'].append("Mix lacks high-frequency energy - consider gentle HF boost")
                elif balance_assessment.get('high_assessment') == 'bright':
                    insights['mixing_notes'].append("Mix may be too bright - consider gentle HF reduction")
            
            # Professional balance score insights
            balance_score = balance_analysis.get('professional_balance_score', 0.5)
            if balance_score < 0.3:
                insights['mixing_notes'].append("Frequency balance significantly deviates from professional standards")
            elif balance_score < 0.6:
                insights['mixing_notes'].append("Frequency balance could be improved for more professional sound")
            
            return insights
            
        except Exception as e:
            logger.error(f"Mixing insights generation failed: {e}")
            return insights
    
    def _assess_overall_frequency_balance(self, band_analysis: Dict[str, any], balance_analysis: Dict[str, any]) -> Dict[str, any]:
        """Provide overall assessment of frequency balance"""
        try:
            balance_score = balance_analysis.get('professional_balance_score', 0.5)
            balance_assessment = balance_analysis.get('balance_assessment', {})
            
            # Determine overall quality rating
            if balance_score >= 0.8:
                quality_rating = 'excellent'
            elif balance_score >= 0.65:
                quality_rating = 'good'
            elif balance_score >= 0.45:
                quality_rating = 'fair'
            else:
                quality_rating = 'needs_improvement'
            
            # Count problematic bands
            problematic_bands = []
            for band_name, band_data in band_analysis.items():
                strength = band_data.get('relative_strength', 'moderate')
                if strength in ['very_strong', 'weak']:
                    problematic_bands.append(band_name)
            
            # Generate summary
            summary_notes = []
            
            if quality_rating == 'excellent':
                summary_notes.append("Frequency balance is professional and well-suited for commercial release")
            elif quality_rating == 'good':
                summary_notes.append("Good frequency balance with minor adjustments needed")
            elif quality_rating == 'fair':
                summary_notes.append("Frequency balance is acceptable but could benefit from EQ adjustments")
            else:
                summary_notes.append("Significant frequency balance issues detected - EQ work recommended")
            
            if len(problematic_bands) > 0:
                summary_notes.append(f"Focus attention on: {', '.join(problematic_bands)}")
            
            return {
                'quality_rating': quality_rating,
                'balance_score': balance_score,
                'problematic_bands': problematic_bands,
                'summary_notes': summary_notes,
                'overall_balance': balance_assessment.get('overall_balance', 'unknown'),
                'ready_for_mastering': quality_rating in ['excellent', 'good'] and len(problematic_bands) <= 1
            }
            
        except Exception as e:
            logger.error(f"Overall assessment failed: {e}")
            return {
                'quality_rating': 'unknown',
                'balance_score': 0.0,
                'problematic_bands': [],
                'summary_notes': ['Analysis failed'],
                'overall_balance': 'unknown',
                'ready_for_mastering': False
            }