"""
Enhanced Loudness Analysis - Professional LUFS measurement
Advanced loudness analysis using pyloudnorm for broadcasting and streaming standards
"""

import numpy as np
import librosa
import pyloudnorm as pyln
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class LoudnessAnalyzer:
    """
    Professional loudness analysis with LUFS measurement
    Implements ITU-R BS.1770-4 and EBU R128 standards
    """
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize the loudness analyzer
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        
        # LUFS targets for different platforms
        self.lufs_targets = audio_settings.LUFS_TARGETS
        
        # Dynamic range targets
        self.dynamic_range_targets = audio_settings.DYNAMIC_RANGE_TARGETS
        
        # Initialize loudness meter
        try:
            self.meter = pyln.Meter(self.sample_rate)
            logger.debug("LoudnessAnalyzer initialized with ITU-R BS.1770-4 meter")
        except Exception as e:
            logger.error(f"Failed to initialize loudness meter: {e}")
            self.meter = None
    
    def analyze_loudness(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive loudness analysis
        
        Args:
            audio: Audio data (mono or stereo)
            
        Returns:
            Dictionary with loudness analysis results
        """
        if self.meter is None:
            logger.error("Loudness meter not initialized")
            return self._create_empty_result()
        
        logger.debug("Starting comprehensive loudness analysis")
        
        try:
            # Ensure proper audio format for pyloudnorm
            if len(audio.shape) == 1:
                # Mono audio - duplicate to stereo for proper LUFS measurement
                audio_for_lufs = np.stack([audio, audio], axis=0).T
            else:
                # Stereo audio
                audio_for_lufs = audio.T
            
            # Integrated loudness (LUFS)
            integrated_loudness = self._measure_integrated_loudness(audio_for_lufs)
            
            # Momentary loudness measurements
            momentary_analysis = self._analyze_momentary_loudness(audio_for_lufs)
            
            # Short-term loudness measurements
            short_term_analysis = self._analyze_short_term_loudness(audio_for_lufs)
            
            # Loudness range (LRA)
            loudness_range = self._measure_loudness_range(audio_for_lufs)
            
            # Peak analysis
            peak_analysis = self._analyze_peaks(audio_for_lufs)
            
            # Dynamic range analysis
            dynamic_range_analysis = self._analyze_dynamic_range(audio, integrated_loudness, peak_analysis)
            
            # Platform compliance analysis
            compliance_analysis = self._analyze_platform_compliance(integrated_loudness, peak_analysis, loudness_range)
            
            # Gating analysis (for technical users)
            gating_analysis = self._analyze_gating_behavior(audio_for_lufs)
            
            result = {
                'integrated_loudness': integrated_loudness,
                'momentary_analysis': momentary_analysis,
                'short_term_analysis': short_term_analysis,
                'loudness_range': loudness_range,
                'peak_analysis': peak_analysis,
                'dynamic_range_analysis': dynamic_range_analysis,
                'compliance_analysis': compliance_analysis,
                'gating_analysis': gating_analysis,
                'overall_assessment': self._assess_overall_loudness(integrated_loudness, dynamic_range_analysis, compliance_analysis)
            }
            
            logger.info(f"Loudness analysis completed - LUFS: {integrated_loudness:.1f}, LRA: {loudness_range:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Loudness analysis failed: {e}")
            return self._create_empty_result()
    
    def _measure_integrated_loudness(self, audio: np.ndarray) -> float:
        """Measure integrated loudness (LUFS)"""
        try:
            # Integrated loudness measurement
            loudness = self.meter.integrated_loudness(audio)
            return float(loudness)
        except Exception as e:
            logger.warning(f"Integrated loudness measurement failed: {e}")
            return -np.inf
    
    def _analyze_momentary_loudness(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze momentary loudness (400ms windows)"""
        try:
            # Calculate momentary loudness values
            # Note: pyloudnorm doesn't have direct momentary loudness, so we'll estimate
            
            window_size = int(0.4 * self.sample_rate)  # 400ms window
            hop_size = int(0.1 * self.sample_rate)  # 100ms hop for overlap
            
            momentary_values = []
            
            for start in range(0, len(audio) - window_size, hop_size):
                window = audio[start:start + window_size]
                if len(window) == window_size:
                    try:
                        momentary_lufs = self.meter.integrated_loudness(window)
                        if not np.isneginf(momentary_lufs):
                            momentary_values.append(momentary_lufs)
                    except:
                        continue
            
            if momentary_values:
                return {
                    'max_momentary': float(np.max(momentary_values)),
                    'min_momentary': float(np.min(momentary_values)),
                    'mean_momentary': float(np.mean(momentary_values)),
                    'std_momentary': float(np.std(momentary_values)),
                    'values': momentary_values[:50]  # Limit for storage
                }
            else:
                return {
                    'max_momentary': -np.inf,
                    'min_momentary': -np.inf,
                    'mean_momentary': -np.inf,
                    'std_momentary': 0.0,
                    'values': []
                }
                
        except Exception as e:
            logger.warning(f"Momentary loudness analysis failed: {e}")
            return {
                'max_momentary': -np.inf,
                'min_momentary': -np.inf,
                'mean_momentary': -np.inf,
                'std_momentary': 0.0,
                'values': []
            }
    
    def _analyze_short_term_loudness(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze short-term loudness (3 second windows)"""
        try:
            window_size = int(3.0 * self.sample_rate)  # 3 second window
            hop_size = int(1.0 * self.sample_rate)  # 1 second hop
            
            short_term_values = []
            
            for start in range(0, len(audio) - window_size, hop_size):
                window = audio[start:start + window_size]
                if len(window) == window_size:
                    try:
                        st_lufs = self.meter.integrated_loudness(window)
                        if not np.isneginf(st_lufs):
                            short_term_values.append(st_lufs)
                    except:
                        continue
            
            if short_term_values:
                return {
                    'max_short_term': float(np.max(short_term_values)),
                    'min_short_term': float(np.min(short_term_values)),
                    'mean_short_term': float(np.mean(short_term_values)),
                    'std_short_term': float(np.std(short_term_values)),
                    'values': short_term_values[:50]  # Limit for storage
                }
            else:
                return {
                    'max_short_term': -np.inf,
                    'min_short_term': -np.inf,
                    'mean_short_term': -np.inf,
                    'std_short_term': 0.0,
                    'values': []
                }
                
        except Exception as e:
            logger.warning(f"Short-term loudness analysis failed: {e}")
            return {
                'max_short_term': -np.inf,
                'min_short_term': -np.inf,
                'mean_short_term': -np.inf,
                'std_short_term': 0.0,
                'values': []
            }
    
    def _measure_loudness_range(self, audio: np.ndarray) -> float:
        """Measure loudness range (LRA)"""
        try:
            # Calculate short-term loudness for LRA
            window_size = int(3.0 * self.sample_rate)
            hop_size = int(1.0 * self.sample_rate)
            
            short_term_values = []
            
            for start in range(0, len(audio) - window_size, hop_size):
                window = audio[start:start + window_size]
                if len(window) == window_size:
                    try:
                        st_lufs = self.meter.integrated_loudness(window)
                        if not np.isneginf(st_lufs):
                            short_term_values.append(st_lufs)
                    except:
                        continue
            
            if len(short_term_values) > 0:
                # LRA is 95th percentile - 10th percentile
                lra = np.percentile(short_term_values, 95) - np.percentile(short_term_values, 10)
                return float(lra)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Loudness range measurement failed: {e}")
            return 0.0
    
    def _analyze_peaks(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze peak levels (True Peak and Sample Peak)"""
        try:
            # Sample peaks
            if len(audio.shape) == 2:
                sample_peak_left = float(np.max(np.abs(audio[:, 0])))
                sample_peak_right = float(np.max(np.abs(audio[:, 1])))
                sample_peak_max = max(sample_peak_left, sample_peak_right)
            else:
                sample_peak_max = float(np.max(np.abs(audio)))
                sample_peak_left = sample_peak_right = sample_peak_max
            
            # Convert to dBFS
            sample_peak_dbfs = 20 * np.log10(sample_peak_max + 1e-10)
            
            # True peaks (approximation using upsampling)
            try:
                # Upsample by 4x for better peak detection
                upsampled = librosa.resample(audio.flatten(), orig_sr=self.sample_rate, target_sr=self.sample_rate*4)
                true_peak = float(np.max(np.abs(upsampled)))
                true_peak_dbfs = 20 * np.log10(true_peak + 1e-10)
            except:
                true_peak = sample_peak_max
                true_peak_dbfs = sample_peak_dbfs
            
            # Peak analysis
            peak_analysis = {
                'sample_peak_dbfs': sample_peak_dbfs,
                'true_peak_dbfs': true_peak_dbfs,
                'sample_peak_linear': sample_peak_max,
                'true_peak_linear': true_peak,
                'headroom_db': -true_peak_dbfs,  # Headroom to 0 dBFS
                'clipping_detected': true_peak_dbfs > -0.1,  # Very close to 0 dBFS
                'peak_quality': self._assess_peak_quality(true_peak_dbfs)
            }
            
            if len(audio.shape) == 2:
                peak_analysis.update({
                    'sample_peak_left_dbfs': 20 * np.log10(sample_peak_left + 1e-10),
                    'sample_peak_right_dbfs': 20 * np.log10(sample_peak_right + 1e-10),
                    'stereo_balance': abs(sample_peak_left - sample_peak_right) / max(sample_peak_left, sample_peak_right, 1e-10)
                })
            
            return peak_analysis
            
        except Exception as e:
            logger.warning(f"Peak analysis failed: {e}")
            return {
                'sample_peak_dbfs': -np.inf,
                'true_peak_dbfs': -np.inf,
                'sample_peak_linear': 0.0,
                'true_peak_linear': 0.0,
                'headroom_db': np.inf,
                'clipping_detected': False,
                'peak_quality': 'unknown'
            }
    
    def _assess_peak_quality(self, true_peak_dbfs: float) -> str:
        """Assess peak level quality"""
        if true_peak_dbfs > -0.1:
            return 'clipping_risk'
        elif true_peak_dbfs > -1.0:
            return 'very_hot'
        elif true_peak_dbfs > -3.0:
            return 'hot'
        elif true_peak_dbfs > -6.0:
            return 'good'
        elif true_peak_dbfs > -12.0:
            return 'conservative'
        else:
            return 'very_quiet'
    
    def _analyze_dynamic_range(self, audio: np.ndarray, integrated_loudness: float, peak_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze dynamic range characteristics"""
        try:
            # Get mono version for analysis
            if len(audio.shape) == 2:
                mono_audio = librosa.to_mono(audio.T)
            else:
                mono_audio = audio
            
            # RMS-based dynamic range
            rms_values = librosa.feature.rms(
                y=mono_audio,
                frame_length=2048,
                hop_length=512
            )[0]
            
            if len(rms_values) > 0:
                rms_db = 20 * np.log10(rms_values + 1e-10)
                rms_dynamic_range = float(np.max(rms_db) - np.mean(rms_db))
            else:
                rms_dynamic_range = 0.0
            
            # Peak-to-RMS ratio
            peak_linear = peak_analysis.get('true_peak_linear', 0.0)
            rms_mean = np.sqrt(np.mean(mono_audio**2))
            
            if rms_mean > 0:
                peak_to_rms_ratio = peak_linear / rms_mean
                peak_to_rms_db = 20 * np.log10(peak_to_rms_ratio)
            else:
                peak_to_rms_ratio = 0.0
                peak_to_rms_db = -np.inf
            
            # Crest factor
            crest_factor = peak_to_rms_db
            
            # PLR (Peak to Loudness Ratio) - approximation
            if not np.isneginf(integrated_loudness):
                peak_dbfs = peak_analysis.get('true_peak_dbfs', -np.inf)
                plr = peak_dbfs - integrated_loudness
            else:
                plr = 0.0
            
            # Categorize dynamic range
            dr_category = self._categorize_dynamic_range(rms_dynamic_range)
            
            return {
                'rms_dynamic_range_db': rms_dynamic_range,
                'peak_to_rms_ratio_db': peak_to_rms_db,
                'crest_factor_db': crest_factor,
                'peak_loudness_ratio_db': float(plr),
                'dynamic_range_category': dr_category,
                'compression_detected': rms_dynamic_range < 6.0,  # Heavily compressed
                'dynamic_range_quality': self._assess_dynamic_range_quality(rms_dynamic_range)
            }
            
        except Exception as e:
            logger.warning(f"Dynamic range analysis failed: {e}")
            return {
                'rms_dynamic_range_db': 0.0,
                'peak_to_rms_ratio_db': 0.0,
                'crest_factor_db': 0.0,
                'peak_loudness_ratio_db': 0.0,
                'dynamic_range_category': 'unknown',
                'compression_detected': False,
                'dynamic_range_quality': 'unknown'
            }
    
    def _categorize_dynamic_range(self, dr_db: float) -> str:
        """Categorize dynamic range based on professional standards"""
        for category, (min_dr, max_dr) in self.dynamic_range_targets.items():
            if min_dr <= dr_db <= max_dr:
                return category
        
        if dr_db < 3.0:
            return 'over_compressed'
        else:
            return 'excellent'
    
    def _assess_dynamic_range_quality(self, dr_db: float) -> str:
        """Assess dynamic range quality"""
        if dr_db >= 14.0:
            return 'excellent'
        elif dr_db >= 9.0:
            return 'good'
        elif dr_db >= 6.0:
            return 'acceptable'
        elif dr_db >= 3.0:
            return 'compressed'
        else:
            return 'over_compressed'
    
    def _analyze_platform_compliance(self, integrated_loudness: float, peak_analysis: Dict[str, any], loudness_range: float) -> Dict[str, any]:
        """Analyze compliance with streaming platform standards"""
        compliance = {}
        
        true_peak_dbfs = peak_analysis.get('true_peak_dbfs', -np.inf)
        
        for platform, target_lufs in self.lufs_targets.items():
            # Check LUFS compliance
            lufs_difference = integrated_loudness - target_lufs
            lufs_compliant = abs(lufs_difference) <= 2.0  # ±2 LU tolerance
            
            # Check peak compliance (general -1 dBFS limit for most platforms)
            peak_compliant = true_peak_dbfs <= -1.0
            
            # Overall compliance
            overall_compliant = lufs_compliant and peak_compliant
            
            compliance[platform] = {
                'target_lufs': target_lufs,
                'current_lufs': integrated_loudness,
                'lufs_difference': float(lufs_difference),
                'lufs_compliant': lufs_compliant,
                'peak_compliant': peak_compliant,
                'overall_compliant': overall_compliant,
                'adjustment_needed': {
                    'gain_adjustment_db': -lufs_difference if not lufs_compliant else 0.0,
                    'limiting_needed': not peak_compliant
                }
            }
        
        # Find best matching platform
        best_match = min(compliance.items(), 
                        key=lambda x: abs(x[1]['lufs_difference']),
                        default=(None, None))
        
        return {
            'platform_compliance': compliance,
            'best_matching_platform': best_match[0] if best_match[0] else 'none',
            'loudness_range_quality': self._assess_lra_quality(loudness_range)
        }
    
    def _assess_lra_quality(self, lra: float) -> str:
        """Assess loudness range quality"""
        if lra < 3.0:
            return 'very_compressed'
        elif lra < 6.0:
            return 'compressed'
        elif lra < 10.0:
            return 'moderate'
        elif lra < 15.0:
            return 'dynamic'
        else:
            return 'very_dynamic'
    
    def _analyze_gating_behavior(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze loudness gating behavior (technical analysis)"""
        try:
            # This is a simplified version - real implementation would require
            # access to pyloudnorm's internal gating mechanism
            
            # Calculate percentage of audio above relative gate
            window_size = int(0.4 * self.sample_rate)  # 400ms blocks
            hop_size = int(0.1 * self.sample_rate)
            
            block_loudnesses = []
            for start in range(0, len(audio) - window_size, hop_size):
                block = audio[start:start + window_size]
                if len(block) == window_size:
                    try:
                        block_loudness = self.meter.integrated_loudness(block)
                        if not np.isneginf(block_loudness):
                            block_loudnesses.append(block_loudness)
                    except:
                        continue
            
            if block_loudnesses:
                # Estimate gated percentage (blocks above -70 LUFS absolute gate)
                absolute_gate = -70.0
                above_absolute_gate = [bl for bl in block_loudnesses if bl > absolute_gate]
                gated_percentage = len(above_absolute_gate) / len(block_loudnesses) * 100
                
                return {
                    'total_blocks': len(block_loudnesses),
                    'gated_blocks': len(above_absolute_gate),
                    'gated_percentage': float(gated_percentage),
                    'mean_gated_loudness': float(np.mean(above_absolute_gate)) if above_absolute_gate else -np.inf
                }
            else:
                return {
                    'total_blocks': 0,
                    'gated_blocks': 0,
                    'gated_percentage': 0.0,
                    'mean_gated_loudness': -np.inf
                }
                
        except Exception as e:
            logger.warning(f"Gating analysis failed: {e}")
            return {
                'total_blocks': 0,
                'gated_blocks': 0,
                'gated_percentage': 0.0,
                'mean_gated_loudness': -np.inf
            }
    
    def _assess_overall_loudness(self, integrated_loudness: float, dynamic_range_analysis: Dict[str, any], compliance_analysis: Dict[str, any]) -> Dict[str, any]:
        """Provide overall loudness assessment"""
        try:
            # Determine loudness level category
            if integrated_loudness > -8.0:
                loudness_category = 'very_loud'
            elif integrated_loudness > -12.0:
                loudness_category = 'loud'
            elif integrated_loudness > -18.0:
                loudness_category = 'moderate'
            elif integrated_loudness > -25.0:
                loudness_category = 'quiet'
            else:
                loudness_category = 'very_quiet'
            
            # Count compliant platforms
            platform_compliance = compliance_analysis.get('platform_compliance', {})
            compliant_platforms = [p for p, data in platform_compliance.items() if data.get('overall_compliant', False)]
            
            # Overall quality assessment
            dr_quality = dynamic_range_analysis.get('dynamic_range_quality', 'unknown')
            
            # Combine assessments
            if len(compliant_platforms) >= 2 and dr_quality in ['excellent', 'good']:
                overall_quality = 'excellent'
            elif len(compliant_platforms) >= 1 and dr_quality in ['excellent', 'good', 'acceptable']:
                overall_quality = 'good'
            elif len(compliant_platforms) >= 1 or dr_quality in ['acceptable']:
                overall_quality = 'acceptable'
            else:
                overall_quality = 'needs_improvement'
            
            # Generate recommendations
            recommendations = []
            
            if integrated_loudness < -25.0:
                recommendations.append("Track is very quiet - consider raising overall level")
            elif integrated_loudness > -6.0:
                recommendations.append("Track is very loud - consider reducing level to prevent distortion")
            
            if dr_quality == 'over_compressed':
                recommendations.append("Dynamic range is heavily compressed - consider reducing compression")
            elif dr_quality == 'compressed':
                recommendations.append("Dynamic range is quite compressed - verify this is intentional")
            
            if len(compliant_platforms) == 0:
                recommendations.append("Not compliant with any streaming platforms - mastering adjustment needed")
            
            return {
                'loudness_category': loudness_category,
                'overall_quality': overall_quality,
                'compliant_platforms': compliant_platforms,
                'compliance_count': len(compliant_platforms),
                'dynamic_range_quality': dr_quality,
                'recommendations': recommendations,
                'ready_for_distribution': overall_quality in ['excellent', 'good'] and len(compliant_platforms) >= 1
            }
            
        except Exception as e:
            logger.error(f"Overall loudness assessment failed: {e}")
            return {
                'loudness_category': 'unknown',
                'overall_quality': 'unknown',
                'compliant_platforms': [],
                'compliance_count': 0,
                'dynamic_range_quality': 'unknown',
                'recommendations': ['Analysis failed'],
                'ready_for_distribution': False
            }
    
    def _create_empty_result(self) -> Dict[str, any]:
        """Create empty result when analysis fails"""
        return {
            'integrated_loudness': -np.inf,
            'momentary_analysis': {'max_momentary': -np.inf, 'values': []},
            'short_term_analysis': {'max_short_term': -np.inf, 'values': []},
            'loudness_range': 0.0,
            'peak_analysis': {'true_peak_dbfs': -np.inf, 'clipping_detected': False},
            'dynamic_range_analysis': {'dynamic_range_quality': 'unknown'},
            'compliance_analysis': {'platform_compliance': {}, 'best_matching_platform': 'none'},
            'gating_analysis': {'gated_percentage': 0.0},
            'overall_assessment': {
                'overall_quality': 'unknown',
                'ready_for_distribution': False,
                'recommendations': ['Analysis failed']
            }
        }
    
    def suggest_loudness_adjustments(self, loudness_result: Dict[str, any]) -> Dict[str, any]:
        """Suggest loudness adjustments for different platforms"""
        integrated_loudness = loudness_result.get('integrated_loudness', -np.inf)
        compliance_analysis = loudness_result.get('compliance_analysis', {})
        peak_analysis = loudness_result.get('peak_analysis', {})
        
        if np.isneginf(integrated_loudness):
            return {'suggestions': [], 'reasoning': 'No valid loudness measurement'}
        
        suggestions = []
        reasoning = []
        
        # Platform-specific suggestions
        platform_compliance = compliance_analysis.get('platform_compliance', {})
        
        for platform, data in platform_compliance.items():
            if not data.get('overall_compliant', False):
                adjustment = data.get('adjustment_needed', {})
                gain_adjustment = adjustment.get('gain_adjustment_db', 0.0)
                
                if abs(gain_adjustment) > 0.5:
                    suggestions.append({
                        'platform': platform,
                        'type': 'gain_adjustment',
                        'adjustment_db': gain_adjustment,
                        'target_lufs': data.get('target_lufs'),
                        'current_lufs': integrated_loudness
                    })
                    reasoning.append(f"Adjust gain by {gain_adjustment:+.1f} dB for {platform} compliance")
                
                if adjustment.get('limiting_needed', False):
                    suggestions.append({
                        'platform': platform,
                        'type': 'peak_limiting',
                        'target_peak_dbfs': -1.0,
                        'current_peak_dbfs': peak_analysis.get('true_peak_dbfs')
                    })
                    reasoning.append(f"Apply peak limiting for {platform} compliance")
        
        # General mastering suggestions
        overall_assessment = loudness_result.get('overall_assessment', {})
        
        if overall_assessment.get('overall_quality') == 'needs_improvement':
            suggestions.append({
                'type': 'mastering_review',
                'suggestion': 'Complete mastering review recommended',
                'issues': overall_assessment.get('recommendations', [])
            })
            reasoning.extend(overall_assessment.get('recommendations', []))
        
        return {
            'suggestions': suggestions,
            'reasoning': reasoning,
            'current_loudness': integrated_loudness,
            'compliance_summary': {
                'compliant_platforms': overall_assessment.get('compliant_platforms', []),
                'total_platforms': len(platform_compliance)
            }
        }