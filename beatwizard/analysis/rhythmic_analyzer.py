"""
Rhythmic Analyzer - Advanced rhythm pattern and groove analysis
Analyzes rhythmic elements, patterns, and provides groove improvement suggestions
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class RhythmicAnalyzer:
    """
    Analyze rhythmic elements, patterns, and groove characteristics
    Provides specific feedback for improving rhythm and feel
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """Initialize the rhythmic analyzer"""
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        
        # Rhythm pattern templates
        self.rhythm_patterns = self._load_rhythm_patterns()
        
        logger.debug("RhythmicAnalyzer initialized")
    
    def analyze_rhythm(self, audio: np.ndarray, tempo: Optional[float] = None) -> Dict[str, any]:
        """
        Comprehensive rhythmic analysis
        
        Args:
            audio: Audio data
            tempo: Detected tempo (if available)
            
        Returns:
            Dictionary with rhythm analysis results
        """
        logger.debug("Starting rhythmic analysis")
        
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Detect beats and tempo if not provided
        if tempo is None:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        else:
            _, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate, bpm=tempo)
        
        # Analyze different rhythmic aspects
        analysis_results = {}
        
        # Beat grid analysis
        analysis_results['beat_grid'] = self._analyze_beat_grid(audio, beats, tempo)
        
        # Groove analysis
        analysis_results['groove'] = self._analyze_groove(audio, beats, tempo)
        
        # Rhythm pattern recognition
        analysis_results['patterns'] = self._analyze_rhythm_patterns(audio, beats, tempo)
        
        # Syncopation analysis
        analysis_results['syncopation'] = self._analyze_syncopation(audio, beats, tempo)
        
        # Fill and variation analysis
        analysis_results['variations'] = self._analyze_rhythmic_variations(audio, beats, tempo)
        
        # Overall rhythm assessment
        analysis_results['overall_rhythm'] = self._assess_overall_rhythm(analysis_results)
        
        # Generate rhythm recommendations
        analysis_results['recommendations'] = self._generate_rhythm_recommendations(analysis_results, tempo)
        
        return analysis_results
    
    def _analyze_beat_grid(self, audio: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze beat grid accuracy and timing"""
        if len(beats) < 4:
            return {
                'quality': 'poor',
                'issues': ['Not enough beats detected for analysis'],
                'beat_consistency': 0.0
            }
        
        # Calculate beat intervals
        beat_intervals = np.diff(beats)
        
        # Expected beat interval
        expected_interval = 60.0 / tempo
        
        # Calculate timing deviations
        timing_deviations = np.abs(beat_intervals - expected_interval)
        avg_deviation = np.mean(timing_deviations)
        max_deviation = np.max(timing_deviations)
        
        # Beat consistency score
        consistency = 1.0 - (avg_deviation / expected_interval)
        consistency = max(0.0, min(1.0, consistency))
        
        # Quality assessment
        if consistency > 0.95:
            quality = 'excellent'
            issues = []
        elif consistency > 0.85:
            quality = 'good'
            issues = ['Minor timing variations']
        elif consistency > 0.7:
            quality = 'fair'
            issues = ['Noticeable timing inconsistencies']
        else:
            quality = 'poor'
            issues = ['Significant timing problems', 'Consider using a metronome or quantization']
        
        return {
            'quality': quality,
            'beat_consistency': float(consistency),
            'avg_timing_deviation': float(avg_deviation),
            'max_timing_deviation': float(max_deviation),
            'issues': issues,
            'beat_count': len(beats)
        }
    
    def _analyze_groove(self, audio: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze groove characteristics and feel"""
        # Extract onset strength
        onset_strength = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=self.sample_rate)
        
        # Analyze micro-timing (groove)
        groove_characteristics = self._detect_groove_characteristics(onset_strength, times, beats, tempo)
        
        # Analyze swing
        swing_analysis = self._analyze_swing(onset_strength, times, tempo)
        
        # Analyze pocket (rhythmic feel)
        pocket_analysis = self._analyze_pocket(onset_strength, beats, tempo)
        
        return {
            'groove_type': groove_characteristics['type'],
            'groove_strength': groove_characteristics['strength'],
            'swing': swing_analysis,
            'pocket': pocket_analysis,
            'micro_timing': groove_characteristics['micro_timing']
        }
    
    def _analyze_rhythm_patterns(self, audio: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze and recognize rhythm patterns"""
        # Separate percussive elements
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Create beat-synchronized feature matrix
        beat_features = self._extract_beat_features(percussive, beats)
        
        # Pattern recognition
        recognized_patterns = self._recognize_patterns(beat_features, tempo)
        
        # Pattern complexity analysis
        complexity = self._analyze_pattern_complexity(beat_features)
        
        return {
            'recognized_patterns': recognized_patterns,
            'pattern_complexity': complexity,
            'dominant_pattern': recognized_patterns[0] if recognized_patterns else 'unknown',
            'pattern_variations': len(set(recognized_patterns))
        }
    
    def _analyze_syncopation(self, audio: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze syncopation and off-beat elements"""
        # Detect onsets
        onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, units='time')
        
        if len(beats) < 2 or len(onsets) < 2:
            return {
                'syncopation_level': 'low',
                'off_beat_ratio': 0.0,
                'syncopated_events': 0
            }
        
        # Create beat grid
        beat_interval = 60.0 / tempo
        beat_subdivisions = np.arange(beats[0], beats[-1], beat_interval / 4)  # 16th note grid
        
        # Count off-beat events
        off_beat_events = 0
        total_events = len(onsets)
        
        for onset in onsets:
            # Find closest subdivision
            closest_subdivision = beat_subdivisions[np.argmin(np.abs(beat_subdivisions - onset))]
            distance_to_subdivision = abs(onset - closest_subdivision)
            
            # If onset is not close to a subdivision, it's syncopated
            if distance_to_subdivision > beat_interval / 16:  # Tolerance
                off_beat_events += 1
        
        off_beat_ratio = off_beat_events / total_events if total_events > 0 else 0
        
        # Classify syncopation level
        if off_beat_ratio < 0.1:
            syncopation_level = 'low'
        elif off_beat_ratio < 0.3:
            syncopation_level = 'moderate'
        else:
            syncopation_level = 'high'
        
        return {
            'syncopation_level': syncopation_level,
            'off_beat_ratio': float(off_beat_ratio),
            'syncopated_events': off_beat_events,
            'total_events': total_events
        }
    
    def _analyze_rhythmic_variations(self, audio: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze rhythmic variations and fills"""
        # Divide track into sections (4-bar sections)
        bars_per_section = 4
        beats_per_bar = 4  # Assume 4/4 time
        beats_per_section = bars_per_section * beats_per_bar
        
        sections = []
        section_features = []
        
        for i in range(0, len(beats) - beats_per_section, beats_per_section):
            section_beats = beats[i:i + beats_per_section]
            if len(section_beats) >= beats_per_section:
                section_start = section_beats[0]
                section_end = section_beats[-1]
                
                # Extract section audio
                start_sample = int(section_start * self.sample_rate)
                end_sample = int(section_end * self.sample_rate)
                section_audio = audio[start_sample:end_sample]
                
                # Extract features for this section
                features = self._extract_section_features(section_audio)
                section_features.append(features)
                sections.append((section_start, section_end))
        
        # Analyze variation between sections
        variation_analysis = self._calculate_section_variations(section_features)
        
        # Detect fills
        fills = self._detect_fills(section_features, sections)
        
        return {
            'section_count': len(sections),
            'variation_score': variation_analysis['score'],
            'fills_detected': fills,
            'repetitiveness': variation_analysis['repetitiveness'],
            'dynamic_changes': variation_analysis['dynamic_changes']
        }
    
    def _detect_groove_characteristics(self, onset_strength: np.ndarray, times: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Detect groove characteristics"""
        # Simplified groove detection
        groove_types = ['straight', 'swing', 'shuffle', 'latin', 'funk']
        
        # For now, return basic groove analysis
        # This would be expanded with machine learning models trained on different groove types
        
        return {
            'type': 'straight',  # Placeholder
            'strength': 0.7,     # Placeholder
            'micro_timing': 'moderate'
        }
    
    def _analyze_swing(self, onset_strength: np.ndarray, times: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze swing characteristics"""
        # Simplified swing analysis
        # Real implementation would analyze timing relationships between beats
        
        return {
            'swing_ratio': 0.5,  # 50% = straight, >50% = swing
            'swing_type': 'straight',
            'swing_consistency': 0.8
        }
    
    def _analyze_pocket(self, onset_strength: np.ndarray, beats: np.ndarray, tempo: float) -> Dict[str, any]:
        """Analyze rhythmic pocket and feel"""
        # Simplified pocket analysis
        return {
            'pocket_quality': 'good',
            'tightness': 0.8,
            'feel': 'on_top'  # 'on_top', 'in_pocket', 'behind'
        }
    
    def _extract_beat_features(self, audio: np.ndarray, beats: np.ndarray) -> np.ndarray:
        """Extract features synchronized to beats"""
        # Create beat-synchronous features
        beat_frames = librosa.time_to_frames(beats, sr=self.sample_rate)
        
        # Extract spectral features at beat locations
        stft = librosa.stft(audio)
        beat_features = []
        
        for frame in beat_frames:
            if frame < stft.shape[1]:
                # Extract frequency bins for this beat
                beat_spectrum = np.abs(stft[:, frame])
                beat_features.append(beat_spectrum)
        
        return np.array(beat_features) if beat_features else np.array([])
    
    def _recognize_patterns(self, beat_features: np.ndarray, tempo: float) -> List[str]:
        """Recognize rhythm patterns"""
        # Simplified pattern recognition
        patterns = []
        
        # Classify based on tempo and basic features
        if 60 <= tempo <= 80:
            patterns.append('ballad')
        elif 80 <= tempo <= 100:
            patterns.append('hip_hop')
        elif 100 <= tempo <= 130:
            patterns.append('house')
        elif 130 <= tempo <= 150:
            patterns.append('techno')
        else:
            patterns.append('electronic')
        
        return patterns
    
    def _analyze_pattern_complexity(self, beat_features: np.ndarray) -> Dict[str, any]:
        """Analyze rhythm pattern complexity"""
        if len(beat_features) == 0:
            return {'complexity_score': 0.0, 'complexity_level': 'unknown'}
        
        # Calculate variation in beat features
        if len(beat_features) > 1:
            variation = np.std(beat_features, axis=0)
            avg_variation = np.mean(variation)
        else:
            avg_variation = 0.0
        
        # Classify complexity
        if avg_variation < 0.1:
            complexity_level = 'simple'
        elif avg_variation < 0.3:
            complexity_level = 'moderate'
        else:
            complexity_level = 'complex'
        
        return {
            'complexity_score': float(avg_variation),
            'complexity_level': complexity_level
        }
    
    def _extract_section_features(self, section_audio: np.ndarray) -> Dict[str, any]:
        """Extract features for a section of audio"""
        if len(section_audio) == 0:
            return {'rms': 0.0, 'spectral_centroid': 0.0, 'zero_crossing_rate': 0.0}
        
        # Calculate basic features
        rms = np.sqrt(np.mean(section_audio**2))
        
        # Spectral centroid
        stft = librosa.stft(section_audio)
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))
        avg_centroid = np.mean(spectral_centroid)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(section_audio)
        avg_zcr = np.mean(zcr)
        
        return {
            'rms': float(rms),
            'spectral_centroid': float(avg_centroid),
            'zero_crossing_rate': float(avg_zcr)
        }
    
    def _calculate_section_variations(self, section_features: List[Dict]) -> Dict[str, any]:
        """Calculate variations between sections"""
        if len(section_features) < 2:
            return {'score': 0.0, 'repetitiveness': 'high', 'dynamic_changes': 0}
        
        # Calculate RMS variations
        rms_values = [section['rms'] for section in section_features]
        rms_variation = np.std(rms_values) / (np.mean(rms_values) + 1e-10)
        
        # Calculate spectral variations
        centroid_values = [section['spectral_centroid'] for section in section_features]
        centroid_variation = np.std(centroid_values) / (np.mean(centroid_values) + 1e-10)
        
        # Overall variation score
        variation_score = (rms_variation + centroid_variation) / 2
        
        # Classify repetitiveness
        if variation_score < 0.1:
            repetitiveness = 'high'
        elif variation_score < 0.3:
            repetitiveness = 'moderate'
        else:
            repetitiveness = 'low'
        
        # Count significant dynamic changes
        dynamic_changes = 0
        for i in range(1, len(rms_values)):
            if abs(rms_values[i] - rms_values[i-1]) / rms_values[i-1] > 0.2:
                dynamic_changes += 1
        
        return {
            'score': float(variation_score),
            'repetitiveness': repetitiveness,
            'dynamic_changes': dynamic_changes
        }
    
    def _detect_fills(self, section_features: List[Dict], sections: List[Tuple]) -> List[Dict]:
        """Detect drum fills and variations"""
        fills = []
        
        # Look for sections with significantly higher activity
        if len(section_features) > 1:
            avg_zcr = np.mean([section['zero_crossing_rate'] for section in section_features])
            
            for i, section in enumerate(section_features):
                if section['zero_crossing_rate'] > avg_zcr * 1.5:
                    fills.append({
                        'section_index': i,
                        'start_time': sections[i][0],
                        'end_time': sections[i][1],
                        'intensity': section['zero_crossing_rate'] / avg_zcr,
                        'type': 'high_activity_fill'
                    })
        
        return fills
    
    def _assess_overall_rhythm(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Assess overall rhythmic quality"""
        beat_quality = analysis_results.get('beat_grid', {}).get('quality', 'poor')
        groove_strength = analysis_results.get('groove', {}).get('groove_strength', 0.0)
        variation_score = analysis_results.get('variations', {}).get('variation_score', 0.0)
        
        # Calculate overall rhythm score
        quality_scores = {
            'excellent': 4,
            'good': 3,
            'fair': 2,
            'poor': 1
        }
        
        beat_score = quality_scores.get(beat_quality, 1)
        groove_score = groove_strength * 4  # Scale to 0-4
        variation_score_scaled = min(variation_score * 4, 4)  # Scale to 0-4
        
        overall_score = (beat_score + groove_score + variation_score_scaled) / 3
        
        # Determine overall quality
        if overall_score >= 3.5:
            overall_quality = 'excellent'
        elif overall_score >= 2.5:
            overall_quality = 'good'
        elif overall_score >= 1.5:
            overall_quality = 'fair'
        else:
            overall_quality = 'poor'
        
        return {
            'overall_quality': overall_quality,
            'rhythm_score': float(overall_score),
            'strengths': self._identify_rhythm_strengths(analysis_results),
            'weaknesses': self._identify_rhythm_weaknesses(analysis_results)
        }
    
    def _identify_rhythm_strengths(self, analysis_results: Dict[str, any]) -> List[str]:
        """Identify rhythmic strengths"""
        strengths = []
        
        beat_quality = analysis_results.get('beat_grid', {}).get('quality', 'poor')
        if beat_quality in ['excellent', 'good']:
            strengths.append('Tight timing and beat consistency')
        
        syncopation = analysis_results.get('syncopation', {}).get('syncopation_level', 'low')
        if syncopation in ['moderate', 'high']:
            strengths.append('Good use of syncopation and off-beat elements')
        
        fills = analysis_results.get('variations', {}).get('fills_detected', [])
        if len(fills) > 0:
            strengths.append('Good use of fills and variations')
        
        return strengths
    
    def _identify_rhythm_weaknesses(self, analysis_results: Dict[str, any]) -> List[str]:
        """Identify rhythmic weaknesses"""
        weaknesses = []
        
        beat_quality = analysis_results.get('beat_grid', {}).get('quality', 'poor')
        if beat_quality == 'poor':
            weaknesses.append('Timing inconsistencies affecting groove')
        
        repetitiveness = analysis_results.get('variations', {}).get('repetitiveness', 'high')
        if repetitiveness == 'high':
            weaknesses.append('Track is too repetitive, needs more rhythmic variation')
        
        complexity = analysis_results.get('patterns', {}).get('pattern_complexity', {}).get('complexity_level', 'simple')
        if complexity == 'simple':
            weaknesses.append('Rhythm patterns could be more interesting')
        
        return weaknesses
    
    def _generate_rhythm_recommendations(self, analysis_results: Dict[str, any], tempo: float) -> Dict[str, any]:
        """Generate specific rhythm improvement recommendations"""
        recommendations = []
        beginner_tips = []
        advanced_techniques = []
        
        # Analyze issues and generate recommendations
        beat_quality = analysis_results.get('beat_grid', {}).get('quality', 'poor')
        
        if beat_quality == 'poor':
            recommendations.append({
                'priority': 'high',
                'area': 'timing',
                'issue': 'Beat timing inconsistencies',
                'solution': 'Use a metronome or quantize your drums',
                'beginner_tip': 'Record to a click track or use your DAW\'s metronome'
            })
        
        repetitiveness = analysis_results.get('variations', {}).get('repetitiveness', 'high')
        if repetitiveness == 'high':
            recommendations.append({
                'priority': 'medium',
                'area': 'variation',
                'issue': 'Track lacks rhythmic variation',
                'solution': 'Add fills, breaks, and rhythm changes',
                'beginner_tip': 'Try adding a simple fill every 8 or 16 bars'
            })
        
        syncopation_level = analysis_results.get('syncopation', {}).get('syncopation_level', 'low')
        if syncopation_level == 'low':
            recommendations.append({
                'priority': 'low',
                'area': 'groove',
                'issue': 'Rhythm could be more interesting',
                'solution': 'Add some off-beat elements and syncopation',
                'beginner_tip': 'Try placing hi-hats or percussion slightly off the beat'
            })
        
        # Generate beginner tips
        beginner_tips = [
            f"For {tempo:.0f} BPM, try a {self._suggest_rhythm_pattern(tempo)} pattern",
            "Start with kick on 1 and 3, snare on 2 and 4 (classic rock/pop pattern)",
            "Add hi-hats on the off-beats for energy",
            "Use ghost notes on the snare for groove",
            "Try the 'less is more' approach - simple can be powerful"
        ]
        
        # Advanced techniques
        advanced_techniques = [
            "Experiment with micro-timing adjustments",
            "Layer different percussion elements",
            "Use sidechain compression for groove",
            "Try polyrhythmic elements",
            "Experiment with odd time signatures"
        ]
        
        return {
            'recommendations': recommendations,
            'beginner_tips': beginner_tips,
            'advanced_techniques': advanced_techniques,
            'suggested_pattern': self._suggest_rhythm_pattern(tempo),
            'next_steps': self._generate_rhythm_next_steps(recommendations)
        }
    
    def _suggest_rhythm_pattern(self, tempo: float) -> str:
        """Suggest appropriate rhythm pattern based on tempo"""
        if tempo < 80:
            return "ballad pattern (simple kick-snare)"
        elif tempo < 100:
            return "hip-hop pattern (emphasize beat 1 and 3)"
        elif tempo < 130:
            return "four-on-the-floor (kick on every beat)"
        elif tempo < 150:
            return "techno pattern (steady kick, syncopated hats)"
        else:
            return "breakbeat or drum & bass pattern"
    
    def _generate_rhythm_next_steps(self, recommendations: List[Dict]) -> List[str]:
        """Generate actionable next steps for rhythm improvement"""
        next_steps = []
        
        high_priority = [rec for rec in recommendations if rec.get('priority') == 'high']
        if high_priority:
            next_steps.append(f"1. Fix timing issues: {high_priority[0]['solution']}")
        
        medium_priority = [rec for rec in recommendations if rec.get('priority') == 'medium']
        if medium_priority:
            next_steps.append(f"2. Add variation: {medium_priority[0]['solution']}")
        
        next_steps.extend([
            "3. Reference professional tracks in your genre",
            "4. Practice with a metronome",
            "5. Experiment with different groove templates"
        ])
        
        return next_steps[:5]
    
    def _load_rhythm_patterns(self) -> Dict[str, any]:
        """Load rhythm pattern templates"""
        # Placeholder for rhythm pattern database
        return {
            'four_on_floor': {},
            'hip_hop': {},
            'breakbeat': {},
            'latin': {},
            'funk': {}
        }