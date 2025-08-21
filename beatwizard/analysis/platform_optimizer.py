"""
Platform Optimization - Enhanced streaming platform targeting
Optimizes tracks for specific streaming platforms and distribution channels
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from config.settings import audio_settings


class PlatformOptimizer:
    """
    Enhanced platform optimization for streaming services
    Provides detailed recommendations for platform-specific requirements
    """
    
    def __init__(self):
        """Initialize the platform optimizer"""
        
        # Enhanced platform specifications
        self.platforms = {
            'spotify': {
                'name': 'Spotify',
                'target_lufs': -14.0,
                'max_peak': -1.0,
                'max_lufs': -7.0,
                'preferred_lufs_range': (-16.0, -11.0),
                'dynamic_range_preference': 'moderate',  # 8-15 dB
                'frequency_emphasis': 'balanced',
                'stereo_width_preference': 'wide',  # 0.6-0.9
                'genre_adjustments': {
                    'hip_hop': {'target_lufs': -12.0, 'bass_boost': True},
                    'electronic': {'target_lufs': -13.0, 'high_end_clarity': True},
                    'rock': {'target_lufs': -11.0, 'mid_presence': True},
                    'pop': {'target_lufs': -14.0, 'vocal_clarity': True},
                    'classical': {'target_lufs': -18.0, 'preserve_dynamics': True}
                },
                'technical_requirements': {
                    'sample_rate': [44100, 48000],
                    'bit_depth': [16, 24],
                    'formats': ['OGG Vorbis', 'AAC'],
                    'encoding_quality': 'high'
                }
            },
            'apple_music': {
                'name': 'Apple Music',
                'target_lufs': -16.0,
                'max_peak': -1.0,
                'max_lufs': -8.0,
                'preferred_lufs_range': (-18.0, -13.0),
                'dynamic_range_preference': 'high',  # 10-20 dB
                'frequency_emphasis': 'slightly_warm',
                'stereo_width_preference': 'moderate',  # 0.5-0.8
                'genre_adjustments': {
                    'hip_hop': {'target_lufs': -14.0, 'bass_detail': True},
                    'electronic': {'target_lufs': -15.0, 'stereo_width': True},
                    'rock': {'target_lufs': -13.0, 'guitar_clarity': True},
                    'pop': {'target_lufs': -16.0, 'vocal_warmth': True},
                    'classical': {'target_lufs': -20.0, 'natural_dynamics': True}
                },
                'technical_requirements': {
                    'sample_rate': [44100, 48000, 96000],
                    'bit_depth': [16, 24],
                    'formats': ['AAC', 'ALAC'],
                    'encoding_quality': 'very_high'
                }
            },
            'youtube': {
                'name': 'YouTube',
                'target_lufs': -13.0,
                'max_peak': -1.0,
                'max_lufs': -6.0,
                'preferred_lufs_range': (-15.0, -10.0),
                'dynamic_range_preference': 'low',  # 6-12 dB
                'frequency_emphasis': 'bright',
                'stereo_width_preference': 'narrow',  # 0.4-0.7 (mobile speakers)
                'genre_adjustments': {
                    'hip_hop': {'target_lufs': -11.0, 'punch': True},
                    'electronic': {'target_lufs': -12.0, 'energy': True},
                    'rock': {'target_lufs': -10.0, 'loudness': True},
                    'pop': {'target_lufs': -13.0, 'clarity': True},
                    'classical': {'target_lufs': -16.0, 'compressed_dynamics': True}
                },
                'technical_requirements': {
                    'sample_rate': [44100, 48000],
                    'bit_depth': [16, 24],
                    'formats': ['AAC', 'MP3'],
                    'encoding_quality': 'medium_high'
                }
            },
            'soundcloud': {
                'name': 'SoundCloud',
                'target_lufs': -14.0,
                'max_peak': -0.2,
                'max_lufs': -8.0,
                'preferred_lufs_range': (-16.0, -11.0),
                'dynamic_range_preference': 'moderate',
                'frequency_emphasis': 'balanced',
                'stereo_width_preference': 'wide',
                'genre_adjustments': {
                    'hip_hop': {'target_lufs': -12.0, 'bass_heavy': True},
                    'electronic': {'target_lufs': -13.0, 'creative_freedom': True},
                    'rock': {'target_lufs': -11.0, 'raw_energy': True},
                    'pop': {'target_lufs': -14.0, 'mainstream_appeal': True},
                    'experimental': {'target_lufs': -15.0, 'artistic_integrity': True}
                },
                'technical_requirements': {
                    'sample_rate': [44100, 48000],
                    'bit_depth': [16, 24],
                    'formats': ['MP3', 'AAC'],
                    'encoding_quality': 'medium'
                }
            },
            'tidal': {
                'name': 'Tidal HiFi',
                'target_lufs': -14.0,
                'max_peak': -1.0,
                'max_lufs': -7.0,
                'preferred_lufs_range': (-17.0, -12.0),
                'dynamic_range_preference': 'very_high',  # 12-25 dB
                'frequency_emphasis': 'audiophile',
                'stereo_width_preference': 'wide',
                'genre_adjustments': {
                    'hip_hop': {'target_lufs': -13.0, 'detail_preservation': True},
                    'electronic': {'target_lufs': -14.0, 'spatial_accuracy': True},
                    'rock': {'target_lufs': -12.0, 'instrument_separation': True},
                    'pop': {'target_lufs': -14.0, 'vocal_detail': True},
                    'jazz': {'target_lufs': -18.0, 'audiophile_quality': True}
                },
                'technical_requirements': {
                    'sample_rate': [44100, 48000, 96000, 192000],
                    'bit_depth': [16, 24, 32],
                    'formats': ['FLAC', 'MQA', 'AAC'],
                    'encoding_quality': 'lossless'
                }
            },
            'cd_physical': {
                'name': 'CD/Physical',
                'target_lufs': -9.0,
                'max_peak': -0.1,
                'max_lufs': -6.0,
                'preferred_lufs_range': (-12.0, -8.0),
                'dynamic_range_preference': 'variable',  # Depends on genre
                'frequency_emphasis': 'full_range',
                'stereo_width_preference': 'optimal',
                'genre_adjustments': {
                    'hip_hop': {'target_lufs': -8.0, 'impact': True},
                    'electronic': {'target_lufs': -8.5, 'club_ready': True},
                    'rock': {'target_lufs': -7.0, 'maximum_impact': True},
                    'pop': {'target_lufs': -9.0, 'radio_ready': True},
                    'classical': {'target_lufs': -15.0, 'full_dynamics': True}
                },
                'technical_requirements': {
                    'sample_rate': [44100],
                    'bit_depth': [16],
                    'formats': ['WAV', 'AIFF'],
                    'encoding_quality': 'uncompressed'
                }
            }
        }
        
        # Genre detection patterns (simple keywords)
        self.genre_keywords = {
            'hip_hop': ['hip', 'hop', 'rap', 'trap', 'drill'],
            'electronic': ['electronic', 'edm', 'house', 'techno', 'dubstep', 'ambient'],
            'rock': ['rock', 'metal', 'punk', 'grunge', 'alternative'],
            'pop': ['pop', 'mainstream', 'commercial', 'radio'],
            'classical': ['classical', 'orchestral', 'symphony', 'piano', 'violin'],
            'jazz': ['jazz', 'blues', 'swing', 'bebop'],
            'experimental': ['experimental', 'avant', 'noise', 'abstract']
        }
        
        logger.debug("PlatformOptimizer initialized")
    
    def optimize_for_platforms(self, 
                             loudness_analysis: Dict[str, Any],
                             frequency_analysis: Dict[str, Any],
                             stereo_analysis: Dict[str, Any],
                             mood_analysis: Optional[Dict[str, Any]] = None,
                             target_genre: Optional[str] = None,
                             priority_platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive platform optimization analysis
        
        Args:
            loudness_analysis: Loudness and dynamics analysis
            frequency_analysis: Frequency spectrum analysis  
            stereo_analysis: Stereo imaging analysis
            mood_analysis: Optional mood analysis for context
            target_genre: Optional target genre
            priority_platforms: Optional list of priority platforms
            
        Returns:
            Dictionary with platform optimization recommendations
        """
        logger.debug("Starting comprehensive platform optimization")
        
        # Detect genre if not provided
        detected_genre = self._detect_genre(frequency_analysis, mood_analysis, target_genre)
        
        # Determine priority platforms
        if priority_platforms is None:
            priority_platforms = ['spotify', 'apple_music', 'youtube']
        
        # Analyze current track characteristics
        track_characteristics = self._analyze_track_characteristics(
            loudness_analysis, frequency_analysis, stereo_analysis
        )
        
        # Generate platform-specific recommendations
        platform_recommendations = {}
        for platform_id in priority_platforms:
            if platform_id in self.platforms:
                platform_recommendations[platform_id] = self._optimize_for_platform(
                    platform_id, track_characteristics, detected_genre
                )
        
        # Generate optimization strategy
        optimization_strategy = self._generate_optimization_strategy(
            platform_recommendations, track_characteristics, detected_genre
        )
        
        # Create optimization roadmap
        optimization_roadmap = self._create_optimization_roadmap(
            platform_recommendations, optimization_strategy
        )
        
        result = {
            'detected_genre': detected_genre,
            'track_characteristics': track_characteristics,
            'platform_recommendations': platform_recommendations,
            'optimization_strategy': optimization_strategy,
            'optimization_roadmap': optimization_roadmap,
            'priority_adjustments': self._get_priority_adjustments(platform_recommendations),
            'multi_platform_approach': self._generate_multi_platform_approach(platform_recommendations)
        }
        
        logger.info(f"Platform optimization completed for {len(platform_recommendations)} platforms")
        
        return result
    
    def _detect_genre(self, 
                     frequency_analysis: Dict[str, Any],
                     mood_analysis: Optional[Dict[str, Any]],
                     provided_genre: Optional[str]) -> str:
        """Detect or validate genre from analysis"""
        if provided_genre:
            return provided_genre.lower()
        
        # Simple genre detection based on frequency characteristics
        band_analysis = frequency_analysis.get('band_analysis', {})
        frequency_bands = band_analysis.get('frequency_bands', {})
        
        if not frequency_bands:
            return 'unknown'
        
        # Analyze frequency characteristics
        bass_energy = frequency_bands.get('bass', {}).get('energy', 0)
        sub_bass_energy = frequency_bands.get('sub_bass', {}).get('energy', 0)
        mid_energy = frequency_bands.get('mid', {}).get('energy', 0)
        high_energy = frequency_bands.get('presence', {}).get('energy', 0)
        
        total_energy = bass_energy + sub_bass_energy + mid_energy + high_energy
        
        if total_energy == 0:
            return 'unknown'
        
        # Calculate energy ratios
        bass_ratio = (bass_energy + sub_bass_energy) / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Simple genre classification
        if bass_ratio > 0.6:  # Heavy bass
            return 'hip_hop'
        elif high_ratio > 0.3:  # Bright, electronic characteristics
            return 'electronic'
        elif mid_ratio > 0.4:  # Mid-heavy, likely rock/pop
            if mood_analysis:
                mood = mood_analysis.get('primary_mood', '')
                if mood in ['angry', 'intense']:
                    return 'rock'
                else:
                    return 'pop'
            return 'pop'
        else:
            return 'pop'  # Default fallback
    
    def _analyze_track_characteristics(self,
                                     loudness_analysis: Dict[str, Any],
                                     frequency_analysis: Dict[str, Any],
                                     stereo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current track characteristics"""
        characteristics = {}
        
        # Loudness characteristics
        characteristics['current_lufs'] = loudness_analysis.get('integrated_loudness', -float('inf'))
        characteristics['current_peak'] = loudness_analysis.get('peak_level', 0)
        characteristics['dynamic_range'] = loudness_analysis.get('dynamic_range_analysis', {}).get('dynamic_range_db', 0)
        
        # Frequency characteristics
        balance_analysis = frequency_analysis.get('balance_analysis', {})
        characteristics['frequency_balance'] = balance_analysis.get('balance_score', 0)
        characteristics['frequency_emphasis'] = self._determine_frequency_emphasis(frequency_analysis)
        
        # Stereo characteristics
        characteristics['stereo_width'] = stereo_analysis.get('stereo_width', 0)
        characteristics['stereo_correlation'] = stereo_analysis.get('correlation', 1.0)
        characteristics['mono_compatible'] = stereo_analysis.get('mono_compatibility', {}).get('level_loss_db', 0) <= 3.0
        
        return characteristics
    
    def _determine_frequency_emphasis(self, frequency_analysis: Dict[str, Any]) -> str:
        """Determine the frequency emphasis of the track"""
        band_analysis = frequency_analysis.get('band_analysis', {})
        frequency_bands = band_analysis.get('frequency_bands', {})
        
        if not frequency_bands:
            return 'unknown'
        
        # Get energy levels
        bass_energy = frequency_bands.get('bass', {}).get('energy', 0)
        mid_energy = frequency_bands.get('mid', {}).get('energy', 0)
        high_energy = frequency_bands.get('presence', {}).get('energy', 0)
        
        total_energy = bass_energy + mid_energy + high_energy
        
        if total_energy == 0:
            return 'unknown'
        
        # Determine emphasis
        bass_ratio = bass_energy / total_energy
        high_ratio = high_energy / total_energy
        
        if bass_ratio > 0.5:
            return 'bass_heavy'
        elif high_ratio > 0.3:
            return 'bright'
        elif bass_ratio > 0.3 and high_ratio > 0.2:
            return 'balanced'
        else:
            return 'mid_focused'
    
    def _optimize_for_platform(self,
                              platform_id: str,
                              track_characteristics: Dict[str, Any],
                              genre: str) -> Dict[str, Any]:
        """Generate optimization recommendations for specific platform"""
        platform = self.platforms[platform_id]
        
        # Get genre-specific adjustments
        genre_adjustments = platform.get('genre_adjustments', {}).get(genre, {})
        
        # Calculate target LUFS (with genre adjustment)
        target_lufs = genre_adjustments.get('target_lufs', platform['target_lufs'])
        current_lufs = track_characteristics['current_lufs']
        
        recommendations = {
            'platform_name': platform['name'],
            'compliance_score': 0,
            'adjustments_needed': [],
            'technical_specs': platform['technical_requirements'],
            'genre_specific': genre_adjustments
        }
        
        # LUFS adjustments
        if not np.isinf(current_lufs):
            lufs_diff = current_lufs - target_lufs
            
            if abs(lufs_diff) <= 1.0:
                recommendations['compliance_score'] += 30
            elif abs(lufs_diff) <= 3.0:
                recommendations['compliance_score'] += 20
                recommendations['adjustments_needed'].append({
                    'type': 'loudness',
                    'description': f"Adjust LUFS by {-lufs_diff:+.1f} dB (current: {current_lufs:.1f}, target: {target_lufs:.1f})",
                    'priority': 'medium',
                    'method': 'gain_adjustment'
                })
            else:
                recommendations['compliance_score'] += 5
                recommendations['adjustments_needed'].append({
                    'type': 'loudness',
                    'description': f"Significant LUFS adjustment needed: {-lufs_diff:+.1f} dB",
                    'priority': 'high',
                    'method': 'remaster_required'
                })
        
        # Peak level adjustments
        current_peak = track_characteristics['current_peak']
        max_peak = platform['max_peak']
        
        if current_peak <= max_peak:
            recommendations['compliance_score'] += 25
        else:
            peak_reduction = current_peak - max_peak
            recommendations['adjustments_needed'].append({
                'type': 'peak_limiting',
                'description': f"Reduce peak level by {peak_reduction:.1f} dB (current: {current_peak:.1f}, max: {max_peak:.1f})",
                'priority': 'high',
                'method': 'limiting'
            })
        
        # Dynamic range recommendations
        current_dr = track_characteristics['dynamic_range']
        dr_preference = platform['dynamic_range_preference']
        
        dr_recommendations = self._get_dynamic_range_recommendations(current_dr, dr_preference, genre)
        if dr_recommendations:
            recommendations['adjustments_needed'].extend(dr_recommendations)
            recommendations['compliance_score'] += 15
        else:
            recommendations['compliance_score'] += 25
        
        # Frequency emphasis recommendations
        current_emphasis = track_characteristics['frequency_emphasis']
        preferred_emphasis = platform['frequency_emphasis']
        
        freq_recommendations = self._get_frequency_recommendations(current_emphasis, preferred_emphasis, genre_adjustments)
        if freq_recommendations:
            recommendations['adjustments_needed'].extend(freq_recommendations)
        else:
            recommendations['compliance_score'] += 10
        
        # Stereo width recommendations
        current_width = track_characteristics['stereo_width']
        width_preference = platform['stereo_width_preference']
        
        width_recommendations = self._get_stereo_width_recommendations(current_width, width_preference)
        if width_recommendations:
            recommendations['adjustments_needed'].extend(width_recommendations)
        else:
            recommendations['compliance_score'] += 10
        
        # Calculate final compliance score
        recommendations['compliance_score'] = min(recommendations['compliance_score'], 100)
        
        return recommendations
    
    def _get_dynamic_range_recommendations(self,
                                         current_dr: float,
                                         preference: str,
                                         genre: str) -> List[Dict[str, Any]]:
        """Get dynamic range recommendations"""
        recommendations = []
        
        # Define ideal ranges based on preference
        ideal_ranges = {
            'very_high': (15, 25),
            'high': (10, 20),
            'moderate': (8, 15),
            'low': (6, 12),
            'variable': (8, 20)  # Depends on genre
        }
        
        min_dr, max_dr = ideal_ranges.get(preference, (8, 15))
        
        if current_dr < min_dr:
            recommendations.append({
                'type': 'dynamics',
                'description': f"Increase dynamic range - currently {current_dr:.1f} dB, ideal: {min_dr}-{max_dr} dB",
                'priority': 'medium',
                'method': 'reduce_compression'
            })
        elif current_dr > max_dr:
            recommendations.append({
                'type': 'dynamics',
                'description': f"Consider gentle compression - current DR {current_dr:.1f} dB may be too wide for platform",
                'priority': 'low',
                'method': 'gentle_compression'
            })
        
        return recommendations
    
    def _get_frequency_recommendations(self,
                                     current_emphasis: str,
                                     preferred_emphasis: str,
                                     genre_adjustments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get frequency emphasis recommendations"""
        recommendations = []
        
        # Platform-specific frequency adjustments
        if preferred_emphasis == 'bright' and current_emphasis != 'bright':
            recommendations.append({
                'type': 'eq',
                'description': "Add brightness with high-frequency enhancement (8-16kHz)",
                'priority': 'medium',
                'method': 'high_shelf_boost'
            })
        elif preferred_emphasis == 'slightly_warm' and current_emphasis == 'bright':
            recommendations.append({
                'type': 'eq',
                'description': "Reduce excessive brightness, add low-mid warmth",
                'priority': 'medium',
                'method': 'warm_adjustment'
            })
        
        # Genre-specific adjustments
        for adjustment_type, enabled in genre_adjustments.items():
            if enabled:
                if adjustment_type == 'bass_boost':
                    recommendations.append({
                        'type': 'eq',
                        'description': "Enhance bass presence for hip-hop impact",
                        'priority': 'medium',
                        'method': 'bass_enhancement'
                    })
                elif adjustment_type == 'vocal_clarity':
                    recommendations.append({
                        'type': 'eq',
                        'description': "Enhance vocal clarity in 2-5kHz range",
                        'priority': 'medium',
                        'method': 'vocal_enhancement'
                    })
                elif adjustment_type == 'high_end_clarity':
                    recommendations.append({
                        'type': 'eq',
                        'description': "Enhance high-frequency clarity for electronic music",
                        'priority': 'medium',
                        'method': 'high_frequency_clarity'
                    })
        
        return recommendations
    
    def _get_stereo_width_recommendations(self,
                                        current_width: float,
                                        preference: str) -> List[Dict[str, Any]]:
        """Get stereo width recommendations"""
        recommendations = []
        
        # Define ideal ranges
        width_ranges = {
            'narrow': (0.3, 0.6),
            'moderate': (0.5, 0.8),
            'wide': (0.6, 0.9),
            'optimal': (0.5, 0.8)
        }
        
        min_width, max_width = width_ranges.get(preference, (0.5, 0.8))
        
        if current_width < min_width:
            recommendations.append({
                'type': 'stereo',
                'description': f"Increase stereo width - currently {current_width:.2f}, ideal: {min_width:.2f}-{max_width:.2f}",
                'priority': 'low',
                'method': 'stereo_widening'
            })
        elif current_width > max_width:
            recommendations.append({
                'type': 'stereo',
                'description': f"Reduce stereo width for platform compatibility - currently {current_width:.2f}",
                'priority': 'medium',
                'method': 'stereo_narrowing'
            })
        
        return recommendations
    
    def _generate_optimization_strategy(self,
                                      platform_recommendations: Dict[str, Any],
                                      track_characteristics: Dict[str, Any],
                                      genre: str) -> Dict[str, Any]:
        """Generate overall optimization strategy"""
        strategy = {
            'primary_target': None,
            'secondary_targets': [],
            'conflicting_requirements': [],
            'universal_improvements': [],
            'trade_offs': []
        }
        
        # Find highest compliance score platform
        if platform_recommendations:
            sorted_platforms = sorted(
                platform_recommendations.items(),
                key=lambda x: x[1]['compliance_score'],
                reverse=True
            )
            
            strategy['primary_target'] = sorted_platforms[0][0]
            strategy['secondary_targets'] = [p[0] for p in sorted_platforms[1:3]]
        
        # Identify universal improvements (common across platforms)
        all_adjustments = []
        for platform_data in platform_recommendations.values():
            all_adjustments.extend(platform_data['adjustments_needed'])
        
        # Count adjustment types
        adjustment_counts = {}
        for adj in all_adjustments:
            adj_type = adj['type']
            adj_desc = adj['description']
            key = f"{adj_type}:{adj_desc}"
            adjustment_counts[key] = adjustment_counts.get(key, 0) + 1
        
        # Universal improvements (appear in multiple platforms)
        min_platforms = max(2, len(platform_recommendations) // 2)
        for adj_key, count in adjustment_counts.items():
            if count >= min_platforms:
                adj_type, adj_desc = adj_key.split(':', 1)
                strategy['universal_improvements'].append({
                    'type': adj_type,
                    'description': adj_desc,
                    'platforms_affected': count
                })
        
        return strategy
    
    def _create_optimization_roadmap(self,
                                   platform_recommendations: Dict[str, Any],
                                   strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create step-by-step optimization roadmap"""
        roadmap = {
            'phase_1_universal': [],
            'phase_2_primary_target': [],
            'phase_3_secondary_targets': [],
            'estimated_time_hours': 0
        }
        
        # Phase 1: Universal improvements
        roadmap['phase_1_universal'] = strategy['universal_improvements']
        roadmap['estimated_time_hours'] += len(strategy['universal_improvements']) * 0.5
        
        # Phase 2: Primary target optimization
        if strategy['primary_target']:
            primary_adjustments = platform_recommendations[strategy['primary_target']]['adjustments_needed']
            # Filter out universal adjustments
            universal_descs = [ui['description'] for ui in strategy['universal_improvements']]
            unique_adjustments = [adj for adj in primary_adjustments if adj['description'] not in universal_descs]
            roadmap['phase_2_primary_target'] = unique_adjustments
            roadmap['estimated_time_hours'] += len(unique_adjustments) * 0.3
        
        # Phase 3: Secondary target adjustments
        secondary_adjustments = []
        for target in strategy['secondary_targets']:
            if target in platform_recommendations:
                target_adjustments = platform_recommendations[target]['adjustments_needed']
                # Filter out already handled adjustments
                handled_descs = ([ui['description'] for ui in strategy['universal_improvements']] +
                               [adj['description'] for adj in roadmap['phase_2_primary_target']])
                unique_adjustments = [adj for adj in target_adjustments if adj['description'] not in handled_descs]
                secondary_adjustments.extend(unique_adjustments)
        
        roadmap['phase_3_secondary_targets'] = secondary_adjustments
        roadmap['estimated_time_hours'] += len(secondary_adjustments) * 0.2
        
        return roadmap
    
    def _get_priority_adjustments(self, platform_recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top priority adjustments across all platforms"""
        all_adjustments = []
        
        for platform_id, platform_data in platform_recommendations.items():
            for adjustment in platform_data['adjustments_needed']:
                adjustment_copy = adjustment.copy()
                adjustment_copy['platform'] = platform_data['platform_name']
                all_adjustments.append(adjustment_copy)
        
        # Sort by priority (high > medium > low)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        sorted_adjustments = sorted(
            all_adjustments,
            key=lambda x: priority_order.get(x['priority'], 0),
            reverse=True
        )
        
        return sorted_adjustments[:5]  # Top 5 priority adjustments
    
    def _generate_multi_platform_approach(self, platform_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for multi-platform releases"""
        approach = {
            'single_master_feasible': True,
            'recommended_approach': 'single_master',
            'compromise_settings': {},
            'platform_specific_masters': []
        }
        
        if not platform_recommendations:
            return approach
        
        # Check if single master can work for all platforms
        lufs_targets = []
        peak_limits = []
        
        for platform_data in platform_recommendations.values():
            # Extract target values from adjustments
            for adjustment in platform_data['adjustments_needed']:
                if adjustment['type'] == 'loudness' and 'target:' in adjustment['description']:
                    # Parse target LUFS from description
                    try:
                        target_str = adjustment['description'].split('target: ')[1].split(')')[0]
                        target_lufs = float(target_str)
                        lufs_targets.append(target_lufs)
                    except:
                        pass
        
        # If LUFS targets vary significantly, recommend platform-specific masters
        if lufs_targets and (max(lufs_targets) - min(lufs_targets) > 3.0):
            approach['single_master_feasible'] = False
            approach['recommended_approach'] = 'platform_specific'
            
            # Recommend specific masters for platforms with very different requirements
            for platform_id, platform_data in platform_recommendations.items():
                if platform_data['compliance_score'] < 70:
                    approach['platform_specific_masters'].append({
                        'platform': platform_data['platform_name'],
                        'reason': 'Low compliance score',
                        'adjustments_needed': len(platform_data['adjustments_needed'])
                    })
        else:
            # Calculate compromise settings
            if lufs_targets:
                approach['compromise_settings']['target_lufs'] = np.mean(lufs_targets)
            
            approach['compromise_settings']['optimization_priority'] = [
                platform_id for platform_id, platform_data 
                in sorted(platform_recommendations.items(), 
                         key=lambda x: x[1]['compliance_score'], reverse=True)
            ]
        
        return approach
