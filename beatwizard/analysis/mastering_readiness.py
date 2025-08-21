"""
Mastering Readiness Assessment - Professional pre-mastering evaluation
Comprehensive analysis to determine if a track is ready for mastering
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from config.settings import audio_settings


class MasteringReadinessAnalyzer:
    """
    Professional mastering readiness assessment system
    Evaluates technical and musical factors to determine mastering readiness
    """
    
    def __init__(self):
        """Initialize the mastering readiness analyzer"""
        
        # Scoring weights for different aspects
        self.scoring_weights = {
            'frequency_balance': 0.25,      # EQ and frequency distribution
            'dynamic_range': 0.20,          # Dynamics and compression
            'loudness_standards': 0.20,     # LUFS and peak levels
            'stereo_imaging': 0.15,         # Stereo field quality
            'technical_quality': 0.10,      # Overall technical issues
            'musical_cohesion': 0.10        # Musical and arrangement factors
        }
        
        # Platform-specific requirements
        self.platform_requirements = {
            'spotify': {
                'target_lufs': -14.0,
                'max_peak': -1.0,
                'max_lufs': -7.0,
                'name': 'Spotify'
            },
            'apple_music': {
                'target_lufs': -16.0,
                'max_peak': -1.0,
                'max_lufs': -8.0,
                'name': 'Apple Music'
            },
            'youtube': {
                'target_lufs': -13.0,
                'max_peak': -1.0,
                'max_lufs': -6.0,
                'name': 'YouTube'
            },
            'soundcloud': {
                'target_lufs': -14.0,
                'max_peak': -0.2,
                'max_lufs': -8.0,
                'name': 'SoundCloud'
            },
            'tidal': {
                'target_lufs': -14.0,
                'max_peak': -1.0,
                'max_lufs': -7.0,
                'name': 'Tidal'
            },
            'cd_physical': {
                'target_lufs': -9.0,
                'max_peak': -0.1,
                'max_lufs': -6.0,
                'name': 'CD/Physical'
            }
        }
        
        # Readiness categories
        self.readiness_categories = {
            'ready_for_mastering': (85, 100),      # Ready to master
            'minor_adjustments': (70, 84),         # Small fixes needed
            'moderate_work_needed': (50, 69),      # Some work required
            'significant_work_needed': (30, 49),   # Major issues
            'not_ready': (0, 29)                   # Not ready for mastering
        }
        
        logger.debug("MasteringReadinessAnalyzer initialized")
    
    def assess_mastering_readiness(self, 
                                 frequency_analysis: Dict[str, Any],
                                 loudness_analysis: Dict[str, Any],
                                 stereo_analysis: Dict[str, Any],
                                 tempo_analysis: Optional[Dict[str, Any]] = None,
                                 mood_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive mastering readiness assessment
        
        Args:
            frequency_analysis: Frequency spectrum analysis results
            loudness_analysis: Loudness and dynamics analysis results
            stereo_analysis: Stereo imaging analysis results
            tempo_analysis: Optional tempo analysis for musical context
            mood_analysis: Optional mood analysis for musical context
            
        Returns:
            Dictionary with comprehensive readiness assessment
        """
        logger.debug("Starting comprehensive mastering readiness assessment")
        
        # Individual aspect assessments
        frequency_score = self._assess_frequency_balance(frequency_analysis)
        dynamic_score = self._assess_dynamic_range(loudness_analysis)
        loudness_score = self._assess_loudness_standards(loudness_analysis)
        stereo_score = self._assess_stereo_imaging(stereo_analysis)
        technical_score = self._assess_technical_quality(frequency_analysis, loudness_analysis, stereo_analysis)
        musical_score = self._assess_musical_cohesion(tempo_analysis, mood_analysis, frequency_analysis)
        
        # Calculate overall readiness score
        overall_score = self._calculate_overall_score({
            'frequency_balance': frequency_score,
            'dynamic_range': dynamic_score,
            'loudness_standards': loudness_score,
            'stereo_imaging': stereo_score,
            'technical_quality': technical_score,
            'musical_cohesion': musical_score
        })
        
        # Determine readiness category
        readiness_category = self._determine_readiness_category(overall_score)
        
        # Platform-specific assessments
        platform_assessments = self._assess_platform_readiness(loudness_analysis, overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations({
            'frequency_balance': frequency_score,
            'dynamic_range': dynamic_score,
            'loudness_standards': loudness_score,
            'stereo_imaging': stereo_score,
            'technical_quality': technical_score,
            'musical_cohesion': musical_score
        }, readiness_category)
        
        # Create detailed report
        detailed_assessment = self._create_detailed_assessment({
            'frequency_balance': frequency_score,
            'dynamic_range': dynamic_score,
            'loudness_standards': loudness_score,
            'stereo_imaging': stereo_score,
            'technical_quality': technical_score,
            'musical_cohesion': musical_score
        })
        
        result = {
            'overall_score': overall_score,
            'readiness_category': readiness_category,
            'readiness_description': self._get_readiness_description(readiness_category),
            'individual_scores': {
                'frequency_balance': frequency_score,
                'dynamic_range': dynamic_score,
                'loudness_standards': loudness_score,
                'stereo_imaging': stereo_score,
                'technical_quality': technical_score,
                'musical_cohesion': musical_score
            },
            'platform_assessments': platform_assessments,
            'recommendations': recommendations,
            'detailed_assessment': detailed_assessment,
            'priority_actions': self._get_priority_actions(detailed_assessment, readiness_category),
            'estimated_work_time': self._estimate_work_time(readiness_category, detailed_assessment)
        }
        
        logger.info(f"Mastering readiness: {readiness_category} (score: {overall_score:.1f}/100)")
        
        return result
    
    def _assess_frequency_balance(self, frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess frequency balance readiness for mastering"""
        score = 0
        issues = []
        strengths = []
        
        if 'error' in frequency_analysis:
            return {'score': 0, 'issues': ['Frequency analysis failed'], 'strengths': []}
        
        # Check overall frequency balance
        balance_analysis = frequency_analysis.get('balance_analysis', {})
        overall_balance = balance_analysis.get('balance_score', 0)
        
        if overall_balance >= 8.5:
            score += 30
            strengths.append("Excellent frequency balance")
        elif overall_balance >= 7.0:
            score += 25
            strengths.append("Good frequency balance")
        elif overall_balance >= 5.0:
            score += 15
            issues.append("Frequency balance needs minor adjustment")
        else:
            score += 5
            issues.append("Frequency balance needs significant work")
        
        # Check for frequency masking
        band_analysis = frequency_analysis.get('band_analysis', {})
        if band_analysis:
            frequency_bands = band_analysis.get('frequency_bands', {})
            
            # Check bass buildup
            bass_energy = frequency_bands.get('bass', {}).get('energy', 0)
            sub_bass_energy = frequency_bands.get('sub_bass', {}).get('energy', 0)
            
            if bass_energy > 0 and sub_bass_energy > 0:
                bass_ratio = bass_energy / (sub_bass_energy + 1e-10)
                if 0.5 <= bass_ratio <= 2.0:
                    score += 20
                    strengths.append("Good bass balance")
                else:
                    score += 10
                    issues.append("Bass frequencies need balancing")
            
            # Check mid-range clarity
            mid_energy = frequency_bands.get('mid', {}).get('energy', 0)
            low_mid_energy = frequency_bands.get('low_mid', {}).get('energy', 0)
            
            if mid_energy > 0 and low_mid_energy > 0:
                mid_ratio = mid_energy / (low_mid_energy + 1e-10)
                if 0.3 <= mid_ratio <= 1.5:
                    score += 15
                    strengths.append("Clear mid-range")
                else:
                    score += 5
                    issues.append("Mid-range clarity needs attention")
            
            # Check high-frequency presence
            presence_energy = frequency_bands.get('presence', {}).get('energy', 0)
            brilliance_energy = frequency_bands.get('brilliance', {}).get('energy', 0)
            
            if presence_energy > 10:  # Sufficient presence
                score += 15
                strengths.append("Good high-frequency presence")
            else:
                score += 5
                issues.append("Needs more high-frequency presence")
        
        # Check mixing insights
        mixing_insights = frequency_analysis.get('mixing_insights', {})
        eq_suggestions = mixing_insights.get('eq_suggestions', [])
        
        if len(eq_suggestions) <= 2:
            score += 20
            strengths.append("Minimal EQ corrections needed")
        elif len(eq_suggestions) <= 4:
            score += 10
            issues.append("Some EQ adjustments recommended")
        else:
            score += 0
            issues.append("Multiple EQ corrections needed")
        
        return {
            'score': min(score, 100),
            'issues': issues,
            'strengths': strengths,
            'recommendations': eq_suggestions[:3]  # Top 3 recommendations
        }
    
    def _assess_dynamic_range(self, loudness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess dynamic range readiness for mastering"""
        score = 0
        issues = []
        strengths = []
        
        if 'error' in loudness_analysis:
            return {'score': 0, 'issues': ['Loudness analysis failed'], 'strengths': []}
        
        # Check dynamic range
        dr_analysis = loudness_analysis.get('dynamic_range_analysis', {})
        dynamic_range = dr_analysis.get('dynamic_range_db', 0)
        dr_quality = dr_analysis.get('dynamic_range_quality', 'unknown')
        
        if dr_quality == 'excellent':
            score += 40
            strengths.append(f"Excellent dynamic range ({dynamic_range:.1f} dB)")
        elif dr_quality == 'good':
            score += 30
            strengths.append(f"Good dynamic range ({dynamic_range:.1f} dB)")
        elif dr_quality == 'acceptable':
            score += 20
            issues.append(f"Dynamic range acceptable but could be improved ({dynamic_range:.1f} dB)")
        elif dr_quality == 'compressed':
            score += 10
            issues.append(f"Track is over-compressed ({dynamic_range:.1f} dB)")
        else:
            score += 5
            issues.append(f"Dynamic range severely limited ({dynamic_range:.1f} dB)")
        
        # Check peak levels
        peak_level = loudness_analysis.get('peak_level', 0)
        if peak_level <= -1.0:
            score += 25
            strengths.append(f"Good headroom ({peak_level:.1f} dB peak)")
        elif peak_level <= -0.5:
            score += 15
            issues.append(f"Limited headroom ({peak_level:.1f} dB peak)")
        elif peak_level <= -0.1:
            score += 5
            issues.append(f"Very limited headroom ({peak_level:.1f} dB peak)")
        else:
            score += 0
            issues.append("No headroom - peak levels too high")
        
        # Check for clipping
        clipping_detected = loudness_analysis.get('clipping_detected', False)
        if clipping_detected:
            score = max(score - 30, 0)
            issues.append("Clipping detected - must be fixed before mastering")
        else:
            score += 10
            strengths.append("No clipping detected")
        
        # Check RMS levels
        rms_level = loudness_analysis.get('rms_level', -float('inf'))
        if not np.isinf(rms_level) and rms_level > -30:
            if -20 <= rms_level <= -12:
                score += 25
                strengths.append("Good RMS level for mastering")
            elif -25 <= rms_level <= -8:
                score += 15
                issues.append("RMS level acceptable but could be optimized")
            else:
                score += 5
                issues.append("RMS level needs adjustment")
        
        return {
            'score': min(score, 100),
            'issues': issues,
            'strengths': strengths,
            'recommendations': self._get_dynamic_recommendations(dr_quality, peak_level, clipping_detected)
        }
    
    def _assess_loudness_standards(self, loudness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess loudness standards compliance for mastering"""
        score = 0
        issues = []
        strengths = []
        
        if 'error' in loudness_analysis:
            return {'score': 0, 'issues': ['Loudness analysis failed'], 'strengths': []}
        
        # Check LUFS measurement
        integrated_loudness = loudness_analysis.get('integrated_loudness', -float('inf'))
        
        if not np.isinf(integrated_loudness):
            # Score based on how close to mastering-friendly levels
            if -20 <= integrated_loudness <= -12:
                score += 40
                strengths.append(f"Good pre-mastering loudness ({integrated_loudness:.1f} LUFS)")
            elif -25 <= integrated_loudness <= -8:
                score += 25
                strengths.append(f"Acceptable loudness level ({integrated_loudness:.1f} LUFS)")
            elif -30 <= integrated_loudness <= -6:
                score += 15
                issues.append(f"Loudness level needs adjustment ({integrated_loudness:.1f} LUFS)")
            else:
                score += 5
                issues.append(f"Loudness level inappropriate for mastering ({integrated_loudness:.1f} LUFS)")
        else:
            score += 0
            issues.append("Unable to measure loudness")
        
        # Check platform compliance
        compliance_analysis = loudness_analysis.get('compliance_analysis', {})
        platform_compliance = compliance_analysis.get('platform_compliance', {})
        
        compliant_platforms = sum(1 for platform_data in platform_compliance.values() 
                                if platform_data.get('overall_compliant', False))
        total_platforms = len(platform_compliance)
        
        if total_platforms > 0:
            compliance_ratio = compliant_platforms / total_platforms
            if compliance_ratio >= 0.8:
                score += 30
                strengths.append(f"Compliant with {compliant_platforms}/{total_platforms} platforms")
            elif compliance_ratio >= 0.5:
                score += 20
                strengths.append(f"Compliant with {compliant_platforms}/{total_platforms} platforms")
            elif compliance_ratio >= 0.2:
                score += 10
                issues.append(f"Limited platform compliance ({compliant_platforms}/{total_platforms})")
            else:
                score += 0
                issues.append("Poor platform compliance")
        
        # Check loudness range
        loudness_range = loudness_analysis.get('loudness_range', 0)
        if loudness_range > 0:
            if 6 <= loudness_range <= 20:
                score += 20
                strengths.append(f"Good loudness range ({loudness_range:.1f} LU)")
            elif 3 <= loudness_range <= 25:
                score += 10
                issues.append(f"Loudness range acceptable ({loudness_range:.1f} LU)")
            else:
                score += 5
                issues.append(f"Loudness range needs attention ({loudness_range:.1f} LU)")
        
        # Check for distribution readiness
        distribution_ready = loudness_analysis.get('overall_assessment', {}).get('ready_for_distribution', False)
        if distribution_ready:
            score += 10
            strengths.append("Ready for distribution")
        else:
            issues.append("Not ready for distribution")
        
        return {
            'score': min(score, 100),
            'issues': issues,
            'strengths': strengths,
            'recommendations': self._get_loudness_recommendations(integrated_loudness, compliance_analysis)
        }
    
    def _assess_stereo_imaging(self, stereo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess stereo imaging readiness for mastering"""
        score = 0
        issues = []
        strengths = []
        
        if 'error' in stereo_analysis:
            return {'score': 0, 'issues': ['Stereo analysis failed'], 'strengths': []}
        
        # Check stereo width
        stereo_width = stereo_analysis.get('stereo_width', 0)
        if stereo_width > 0:
            if 0.4 <= stereo_width <= 0.9:
                score += 30
                strengths.append(f"Good stereo width ({stereo_width:.2f})")
            elif 0.2 <= stereo_width <= 1.0:
                score += 20
                strengths.append(f"Acceptable stereo width ({stereo_width:.2f})")
            else:
                score += 10
                issues.append(f"Stereo width needs adjustment ({stereo_width:.2f})")
        
        # Check correlation
        correlation = stereo_analysis.get('correlation', 1.0)
        if correlation >= 0.7:
            score += 25
            strengths.append(f"Good stereo correlation ({correlation:.2f})")
        elif correlation >= 0.5:
            score += 15
            issues.append(f"Stereo correlation acceptable ({correlation:.2f})")
        elif correlation >= 0.3:
            score += 5
            issues.append(f"Poor stereo correlation ({correlation:.2f})")
        else:
            score += 0
            issues.append("Stereo correlation indicates phase problems")
        
        # Check for phase issues
        phase_issues = stereo_analysis.get('phase_issues', True)
        if not phase_issues:
            score += 25
            strengths.append("No phase issues detected")
        else:
            score += 0
            issues.append("Phase issues detected")
        
        # Check for mono compatibility
        mono_compatibility = stereo_analysis.get('mono_compatibility', {})
        mono_level_loss = mono_compatibility.get('level_loss_db', 0)
        
        if mono_level_loss <= 1.0:
            score += 20
            strengths.append("Excellent mono compatibility")
        elif mono_level_loss <= 3.0:
            score += 10
            strengths.append("Good mono compatibility")
        else:
            score += 0
            issues.append(f"Mono compatibility issues ({mono_level_loss:.1f} dB loss)")
        
        return {
            'score': min(score, 100),
            'issues': issues,
            'strengths': strengths,
            'recommendations': self._get_stereo_recommendations(stereo_width, correlation, phase_issues)
        }
    
    def _assess_technical_quality(self, 
                                frequency_analysis: Dict[str, Any],
                                loudness_analysis: Dict[str, Any],
                                stereo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall technical quality for mastering"""
        score = 100  # Start at perfect, deduct for issues
        issues = []
        strengths = []
        
        # Check for major technical issues
        if 'error' in frequency_analysis:
            score -= 30
            issues.append("Frequency analysis failed")
        
        if 'error' in loudness_analysis:
            score -= 30
            issues.append("Loudness analysis failed")
        
        if 'error' in stereo_analysis:
            score -= 20
            issues.append("Stereo analysis failed")
        
        # Check for noise floor issues
        if 'noise_floor' in loudness_analysis:
            noise_floor = loudness_analysis['noise_floor']
            if noise_floor > -50:
                score -= 15
                issues.append(f"High noise floor ({noise_floor:.1f} dB)")
            elif noise_floor > -60:
                score -= 5
                issues.append(f"Moderate noise floor ({noise_floor:.1f} dB)")
            else:
                strengths.append("Low noise floor")
        
        # Check for distortion
        if 'distortion_analysis' in loudness_analysis:
            thd = loudness_analysis['distortion_analysis'].get('thd_percent', 0)
            if thd > 1.0:
                score -= 20
                issues.append(f"High distortion ({thd:.1f}% THD)")
            elif thd > 0.5:
                score -= 10
                issues.append(f"Moderate distortion ({thd:.1f}% THD)")
            else:
                strengths.append("Low distortion")
        
        # Check frequency response consistency
        balance_analysis = frequency_analysis.get('balance_analysis', {})
        frequency_consistency = balance_analysis.get('consistency_score', 0)
        
        if frequency_consistency >= 8:
            strengths.append("Consistent frequency response")
        elif frequency_consistency >= 6:
            score -= 5
            issues.append("Minor frequency response inconsistencies")
        else:
            score -= 15
            issues.append("Significant frequency response issues")
        
        # Check for automation issues
        if 'automation_analysis' in loudness_analysis:
            automation_smoothness = loudness_analysis['automation_analysis'].get('smoothness', 1.0)
            if automation_smoothness < 0.7:
                score -= 10
                issues.append("Automation discontinuities detected")
            else:
                strengths.append("Smooth automation")
        
        return {
            'score': max(score, 0),
            'issues': issues,
            'strengths': strengths,
            'recommendations': self._get_technical_recommendations(issues)
        }
    
    def _assess_musical_cohesion(self, 
                               tempo_analysis: Optional[Dict[str, Any]],
                               mood_analysis: Optional[Dict[str, Any]],
                               frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess musical cohesion and arrangement readiness"""
        score = 50  # Neutral starting point
        issues = []
        strengths = []
        
        # Tempo stability
        if tempo_analysis:
            tempo_confidence = tempo_analysis.get('confidence', 0)
            if tempo_confidence >= 0.8:
                score += 20
                strengths.append("Stable tempo")
            elif tempo_confidence >= 0.6:
                score += 10
                strengths.append("Generally stable tempo")
            else:
                score -= 10
                issues.append("Tempo instability detected")
            
            # Beat consistency
            stability_analysis = tempo_analysis.get('stability_analysis', {})
            beat_consistency = stability_analysis.get('overall_stability', 0)
            if beat_consistency >= 0.8:
                score += 15
                strengths.append("Consistent rhythm")
            elif beat_consistency >= 0.6:
                score += 5
                strengths.append("Generally consistent rhythm")
            else:
                score -= 5
                issues.append("Rhythmic inconsistencies")
        
        # Mood consistency
        if mood_analysis:
            mood_confidence = mood_analysis.get('mood_confidence', 0)
            if mood_confidence >= 0.8:
                score += 15
                strengths.append("Clear emotional direction")
            elif mood_confidence >= 0.6:
                score += 5
                strengths.append("Generally clear mood")
            else:
                score -= 5
                issues.append("Unclear emotional direction")
            
            # Mood dynamics
            mood_dynamics = mood_analysis.get('mood_dynamics', {})
            dynamics_type = mood_dynamics.get('dynamics', 'stable')
            if dynamics_type == 'stable':
                score += 10
                strengths.append("Consistent emotional arc")
            elif dynamics_type == 'moderately_variable':
                score += 15
                strengths.append("Good emotional variation")
            else:
                score -= 5
                issues.append("Inconsistent emotional development")
        
        # Harmonic content consistency
        harmonic_features = frequency_analysis.get('clarity_analysis', {})
        spectral_consistency = harmonic_features.get('spectral_consistency', 0)
        
        if spectral_consistency >= 0.8:
            score += 15
            strengths.append("Consistent harmonic content")
        elif spectral_consistency >= 0.6:
            score += 5
            strengths.append("Generally consistent harmonics")
        else:
            score -= 5
            issues.append("Inconsistent harmonic content")
        
        return {
            'score': max(min(score, 100), 0),
            'issues': issues,
            'strengths': strengths,
            'recommendations': self._get_musical_recommendations(tempo_analysis, mood_analysis)
        }
    
    def _calculate_overall_score(self, scores: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted overall mastering readiness score"""
        total_score = 0
        
        for aspect, weight in self.scoring_weights.items():
            aspect_score = scores[aspect]['score']
            total_score += aspect_score * weight
        
        return round(total_score, 1)
    
    def _determine_readiness_category(self, score: float) -> str:
        """Determine readiness category based on score"""
        for category, (min_score, max_score) in self.readiness_categories.items():
            if min_score <= score <= max_score:
                return category
        return 'not_ready'
    
    def _assess_platform_readiness(self, loudness_analysis: Dict[str, Any], overall_score: float) -> Dict[str, Any]:
        """Assess readiness for specific platforms"""
        platform_scores = {}
        
        integrated_loudness = loudness_analysis.get('integrated_loudness', -float('inf'))
        peak_level = loudness_analysis.get('peak_level', 0)
        
        for platform_id, requirements in self.platform_requirements.items():
            platform_score = overall_score  # Start with overall score
            platform_issues = []
            
            if not np.isinf(integrated_loudness):
                # Check LUFS compliance
                target_lufs = requirements['target_lufs']
                lufs_diff = abs(integrated_loudness - target_lufs)
                
                if lufs_diff <= 1.0:
                    platform_score += 5
                elif lufs_diff <= 3.0:
                    platform_score -= 5
                    platform_issues.append(f"LUFS slightly off target ({integrated_loudness:.1f} vs {target_lufs:.1f})")
                else:
                    platform_score -= 15
                    platform_issues.append(f"LUFS needs significant adjustment ({integrated_loudness:.1f} vs {target_lufs:.1f})")
                
                # Check if too loud
                if integrated_loudness > requirements['max_lufs']:
                    platform_score -= 20
                    platform_issues.append(f"Too loud for {requirements['name']} ({integrated_loudness:.1f} > {requirements['max_lufs']:.1f})")
            
            # Check peak levels
            if peak_level > requirements['max_peak']:
                platform_score -= 10
                platform_issues.append(f"Peak level too high ({peak_level:.1f} > {requirements['max_peak']:.1f})")
            
            platform_scores[platform_id] = {
                'score': max(min(platform_score, 100), 0),
                'requirements': requirements,
                'issues': platform_issues,
                'ready': len(platform_issues) == 0 and platform_score >= 70
            }
        
        return platform_scores
    
    def _generate_recommendations(self, scores: Dict[str, Dict[str, Any]], category: str) -> Dict[str, List[str]]:
        """Generate specific recommendations based on assessment"""
        recommendations = {
            'immediate': [],
            'before_mastering': [],
            'during_mastering': [],
            'general': []
        }
        
        # Priority recommendations based on lowest scores
        lowest_scores = sorted(scores.items(), key=lambda x: x[1]['score'])
        
        for aspect, score_data in lowest_scores[:3]:  # Top 3 issues
            if score_data['score'] < 70:
                recommendations['immediate'].extend(score_data.get('recommendations', []))
        
        # Category-specific recommendations
        if category == 'not_ready':
            recommendations['immediate'].extend([
                "Address major technical issues before proceeding",
                "Consider remix or significant mix adjustments",
                "Focus on frequency balance and dynamic range"
            ])
        elif category == 'significant_work_needed':
            recommendations['before_mastering'].extend([
                "Fix identified technical issues",
                "Adjust levels and balance",
                "Verify mono compatibility"
            ])
        elif category == 'moderate_work_needed':
            recommendations['before_mastering'].extend([
                "Make minor mix adjustments",
                "Check peak levels and headroom",
                "Fine-tune problematic frequency ranges"
            ])
        elif category == 'minor_adjustments':
            recommendations['during_mastering'].extend([
                "Minor EQ adjustments during mastering",
                "Optimize for target platforms",
                "Final level and stereo adjustments"
            ])
        else:  # ready_for_mastering
            recommendations['during_mastering'].extend([
                "Proceed with mastering",
                "Focus on platform optimization",
                "Preserve the existing balance"
            ])
        
        return recommendations
    
    def _create_detailed_assessment(self, scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create detailed assessment breakdown"""
        return {
            'strengths': self._collect_all_strengths(scores),
            'issues': self._collect_all_issues(scores),
            'critical_issues': self._identify_critical_issues(scores),
            'score_breakdown': {aspect: data['score'] for aspect, data in scores.items()},
            'weighted_contributions': {
                aspect: data['score'] * self.scoring_weights[aspect] 
                for aspect, data in scores.items()
            }
        }
    
    def _get_priority_actions(self, detailed_assessment: Dict[str, Any], category: str) -> List[str]:
        """Get priority actions based on assessment"""
        actions = []
        
        critical_issues = detailed_assessment['critical_issues']
        if critical_issues:
            actions.extend([f"CRITICAL: {issue}" for issue in critical_issues[:3]])
        
        # Add category-specific actions
        if category in ['not_ready', 'significant_work_needed']:
            actions.extend([
                "Review and fix mix fundamentals",
                "Check for technical problems",
                "Ensure proper gain staging"
            ])
        elif category == 'moderate_work_needed':
            actions.extend([
                "Address identified frequency issues",
                "Optimize dynamic range",
                "Check stereo imaging"
            ])
        elif category == 'minor_adjustments':
            actions.extend([
                "Make final mix polish adjustments",
                "Verify platform compliance",
                "Check final levels"
            ])
        
        return actions[:5]  # Top 5 priority actions
    
    def _estimate_work_time(self, category: str, detailed_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate time needed to address issues"""
        time_estimates = {
            'ready_for_mastering': {'hours': 0, 'description': 'Ready to proceed'},
            'minor_adjustments': {'hours': 1, 'description': '1-2 hours of minor tweaks'},
            'moderate_work_needed': {'hours': 4, 'description': '3-6 hours of focused work'},
            'significant_work_needed': {'hours': 12, 'description': '1-2 days of major revisions'},
            'not_ready': {'hours': 24, 'description': '2+ days of fundamental work'}
        }
        
        base_estimate = time_estimates.get(category, time_estimates['not_ready'])
        
        # Adjust based on number of critical issues
        critical_count = len(detailed_assessment.get('critical_issues', []))
        additional_hours = critical_count * 2
        
        return {
            'base_hours': base_estimate['hours'],
            'additional_hours': additional_hours,
            'total_hours': base_estimate['hours'] + additional_hours,
            'description': base_estimate['description'],
            'factors': f"Base time plus {additional_hours}h for {critical_count} critical issues"
        }
    
    # Helper methods for generating specific recommendations
    def _get_dynamic_recommendations(self, dr_quality: str, peak_level: float, clipping: bool) -> List[str]:
        """Get dynamic range specific recommendations"""
        recommendations = []
        
        if clipping:
            recommendations.append("Remove clipping - reduce input levels or use limiting")
        
        if peak_level > -1.0:
            recommendations.append("Leave more headroom - aim for -1dB to -3dB peaks")
        
        if dr_quality in ['compressed', 'over_compressed']:
            recommendations.append("Reduce compression or use parallel compression")
            recommendations.append("Consider re-mixing with less aggressive dynamics processing")
        
        return recommendations
    
    def _get_loudness_recommendations(self, lufs: float, compliance: Dict[str, Any]) -> List[str]:
        """Get loudness specific recommendations"""
        recommendations = []
        
        if not np.isinf(lufs):
            if lufs > -8:
                recommendations.append("Track is too loud - reduce overall level before mastering")
            elif lufs < -25:
                recommendations.append("Track is too quiet - increase overall level")
            elif -12 <= lufs <= -8:
                recommendations.append("Good pre-mastering level - proceed with caution")
        
        return recommendations
    
    def _get_stereo_recommendations(self, width: float, correlation: float, phase_issues: bool) -> List[str]:
        """Get stereo imaging specific recommendations"""
        recommendations = []
        
        if phase_issues:
            recommendations.append("Fix phase issues before mastering")
            recommendations.append("Check for out-of-phase elements")
        
        if correlation < 0.5:
            recommendations.append("Improve stereo correlation - check for phase cancellation")
        
        if width > 1.0:
            recommendations.append("Stereo width too wide - may cause mono compatibility issues")
        elif width < 0.3:
            recommendations.append("Consider widening stereo image for better spatial impact")
        
        return recommendations
    
    def _get_technical_recommendations(self, issues: List[str]) -> List[str]:
        """Get technical quality recommendations"""
        recommendations = []
        
        for issue in issues:
            if 'noise' in issue.lower():
                recommendations.append("Use noise reduction or gate to clean up noise floor")
            elif 'distortion' in issue.lower():
                recommendations.append("Reduce gain staging or processing intensity")
            elif 'frequency' in issue.lower():
                recommendations.append("Review EQ choices and frequency balance")
            elif 'automation' in issue.lower():
                recommendations.append("Smooth automation curves and transitions")
        
        return recommendations
    
    def _get_musical_recommendations(self, tempo_analysis: Optional[Dict], mood_analysis: Optional[Dict]) -> List[str]:
        """Get musical cohesion recommendations"""
        recommendations = []
        
        if tempo_analysis and tempo_analysis.get('confidence', 0) < 0.6:
            recommendations.append("Improve timing consistency - consider quantizing or tempo correction")
        
        if mood_analysis and mood_analysis.get('mood_confidence', 0) < 0.6:
            recommendations.append("Clarify emotional direction through arrangement and instrumentation")
        
        return recommendations
    
    def _collect_all_strengths(self, scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Collect all strengths from individual assessments"""
        strengths = []
        for aspect_data in scores.values():
            strengths.extend(aspect_data.get('strengths', []))
        return strengths
    
    def _collect_all_issues(self, scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Collect all issues from individual assessments"""
        issues = []
        for aspect_data in scores.values():
            issues.extend(aspect_data.get('issues', []))
        return issues
    
    def _identify_critical_issues(self, scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify critical issues that must be addressed"""
        critical = []
        
        for aspect, data in scores.items():
            if data['score'] < 30:  # Very low scores are critical
                critical.extend([f"{aspect}: {issue}" for issue in data.get('issues', [])])
        
        return critical
    
    def _get_readiness_description(self, category: str) -> str:
        """Get human-readable description of readiness category"""
        descriptions = {
            'ready_for_mastering': "Track is ready for professional mastering. All technical standards met.",
            'minor_adjustments': "Track is nearly ready with only minor adjustments needed before mastering.",
            'moderate_work_needed': "Track needs some work on technical and musical aspects before mastering.",
            'significant_work_needed': "Track requires significant improvement in multiple areas before mastering.",
            'not_ready': "Track is not ready for mastering and needs fundamental work on mix and technical issues."
        }
        return descriptions.get(category, "Unable to determine readiness level.")
