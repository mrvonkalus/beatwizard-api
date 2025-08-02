"""
Harmonic Analyzer - Advanced harmonic content and progression analysis
Analyzes chord progressions, harmonic movement, and provides harmony improvement suggestions
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class HarmonicAnalyzer:
    """
    Analyze harmonic content, chord progressions, and harmonic movement
    Provides specific feedback for improving harmonic content and progressions
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """Initialize the harmonic analyzer"""
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        
        # Musical theory knowledge
        self.chord_templates = self._load_chord_templates()
        self.progression_templates = self._load_progression_templates()
        
        logger.debug("HarmonicAnalyzer initialized")
    
    def analyze_harmony(self, audio: np.ndarray, key: Optional[str] = None) -> Dict[str, any]:
        """
        Comprehensive harmonic analysis
        
        Args:
            audio: Audio data
            key: Detected key (if available)
            
        Returns:
            Dictionary with harmonic analysis results
        """
        logger.debug("Starting harmonic analysis")
        
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Extract harmonic content
        harmonic_content = self._extract_harmonic_content(audio)
        
        # Analyze different harmonic aspects
        analysis_results = {}
        
        # Chord detection and analysis
        analysis_results['chord_analysis'] = self._analyze_chords(harmonic_content, key)
        
        # Chord progression analysis
        analysis_results['progression_analysis'] = self._analyze_chord_progressions(analysis_results['chord_analysis'])
        
        # Harmonic rhythm analysis
        analysis_results['harmonic_rhythm'] = self._analyze_harmonic_rhythm(analysis_results['chord_analysis'])
        
        # Voice leading analysis
        analysis_results['voice_leading'] = self._analyze_voice_leading(analysis_results['chord_analysis'])
        
        # Harmonic complexity analysis
        analysis_results['complexity'] = self._analyze_harmonic_complexity(harmonic_content, analysis_results['chord_analysis'])
        
        # Tension and resolution analysis
        analysis_results['tension_resolution'] = self._analyze_tension_resolution(analysis_results['chord_analysis'])
        
        # Overall harmonic assessment
        analysis_results['overall_harmony'] = self._assess_overall_harmony(analysis_results)
        
        # Generate harmony recommendations
        analysis_results['recommendations'] = self._generate_harmony_recommendations(analysis_results, key)
        
        return analysis_results
    
    def _extract_harmonic_content(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract harmonic content from audio"""
        # Separate harmonic and percussive content
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Extract chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=self.sample_rate, hop_length=self.hop_length)
        
        # Extract tonnetz (harmonic network features)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sample_rate)
        
        # Extract spectral features for harmonic analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=harmonic, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=harmonic, sr=self.sample_rate)
        
        return {
            'harmonic_audio': harmonic,
            'chroma': chroma,
            'tonnetz': tonnetz,
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff
        }
    
    def _analyze_chords(self, harmonic_content: Dict[str, np.ndarray], key: Optional[str] = None) -> Dict[str, any]:
        """Analyze chord content and detect chord progressions"""
        chroma = harmonic_content['chroma']
        
        if chroma.shape[1] == 0:
            return {
                'detected_chords': [],
                'chord_quality': 'poor',
                'chord_changes': 0,
                'issues': ['No harmonic content detected']
            }
        
        # Simple chord detection based on chroma peaks
        detected_chords = []
        chord_confidences = []
        
        # Analyze each time frame
        for frame_idx in range(chroma.shape[1]):
            frame_chroma = chroma[:, frame_idx]
            
            # Find the most prominent chord for this frame
            chord, confidence = self._detect_chord_in_frame(frame_chroma, key)
            detected_chords.append(chord)
            chord_confidences.append(confidence)
        
        # Smooth chord detections (remove rapid changes)
        smoothed_chords = self._smooth_chord_sequence(detected_chords, chord_confidences)
        
        # Count unique chords and changes
        unique_chords = list(set(smoothed_chords))
        chord_changes = self._count_chord_changes(smoothed_chords)
        
        # Assess chord quality
        chord_quality = self._assess_chord_quality(smoothed_chords, chord_confidences)
        
        # Identify issues
        issues = self._identify_chord_issues(smoothed_chords, chord_confidences, key)
        
        return {
            'detected_chords': smoothed_chords,
            'unique_chords': unique_chords,
            'chord_changes': chord_changes,
            'chord_quality': chord_quality,
            'avg_confidence': float(np.mean(chord_confidences)),
            'issues': issues
        }
    
    def _detect_chord_in_frame(self, chroma_frame: np.ndarray, key: Optional[str] = None) -> Tuple[str, float]:
        """Detect the most likely chord in a single frame"""
        # Normalize chroma
        if np.sum(chroma_frame) > 0:
            normalized_chroma = chroma_frame / np.sum(chroma_frame)
        else:
            return 'No Chord', 0.0
        
        # Find the strongest pitch classes
        strong_notes = np.where(normalized_chroma > 0.15)[0]  # Threshold for significant presence
        
        if len(strong_notes) < 2:
            return 'No Chord', 0.0
        
        # Simple chord detection based on intervals
        # This is a simplified version - a full implementation would use more sophisticated algorithms
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Find root note (strongest note)
        root_idx = np.argmax(normalized_chroma)
        root_note = note_names[root_idx]
        
        # Check for major/minor third
        major_third = (root_idx + 4) % 12
        minor_third = (root_idx + 3) % 12
        
        has_major_third = normalized_chroma[major_third] > 0.1
        has_minor_third = normalized_chroma[minor_third] > 0.1
        
        # Check for fifth
        fifth = (root_idx + 7) % 12
        has_fifth = normalized_chroma[fifth] > 0.1
        
        # Determine chord type
        if has_major_third and has_fifth:
            chord = f"{root_note} major"
            confidence = (normalized_chroma[root_idx] + normalized_chroma[major_third] + normalized_chroma[fifth]) / 3
        elif has_minor_third and has_fifth:
            chord = f"{root_note} minor"
            confidence = (normalized_chroma[root_idx] + normalized_chroma[minor_third] + normalized_chroma[fifth]) / 3
        elif has_fifth:
            chord = f"{root_note} power"
            confidence = (normalized_chroma[root_idx] + normalized_chroma[fifth]) / 2
        else:
            chord = f"{root_note} unclear"
            confidence = normalized_chroma[root_idx]
        
        return chord, float(confidence)
    
    def _smooth_chord_sequence(self, chords: List[str], confidences: List[float]) -> List[str]:
        """Smooth chord sequence to remove rapid changes"""
        if len(chords) < 3:
            return chords
        
        smoothed = chords.copy()
        
        # Simple smoothing: if a chord appears for only one frame between two identical chords, remove it
        for i in range(1, len(chords) - 1):
            if chords[i-1] == chords[i+1] and chords[i] != chords[i-1]:
                if confidences[i] < (confidences[i-1] + confidences[i+1]) / 2:
                    smoothed[i] = chords[i-1]
        
        return smoothed
    
    def _count_chord_changes(self, chords: List[str]) -> int:
        """Count the number of chord changes"""
        if len(chords) < 2:
            return 0
        
        changes = 0
        for i in range(1, len(chords)):
            if chords[i] != chords[i-1]:
                changes += 1
        
        return changes
    
    def _assess_chord_quality(self, chords: List[str], confidences: List[float]) -> str:
        """Assess overall chord detection quality"""
        if not chords:
            return 'poor'
        
        # Count 'No Chord' and 'unclear' chords
        weak_chords = sum(1 for chord in chords if 'No Chord' in chord or 'unclear' in chord)
        weak_ratio = weak_chords / len(chords)
        
        avg_confidence = np.mean(confidences)
        
        if weak_ratio < 0.2 and avg_confidence > 0.3:
            return 'good'
        elif weak_ratio < 0.4 and avg_confidence > 0.2:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_chord_issues(self, chords: List[str], confidences: List[float], key: Optional[str] = None) -> List[str]:
        """Identify issues with chord content"""
        issues = []
        
        if not chords:
            issues.append("No harmonic content detected")
            return issues
        
        # Check for lack of chord changes
        unique_chords = len(set(chords))
        if unique_chords < 2:
            issues.append("Track uses only one chord - needs more harmonic movement")
        elif unique_chords < 3:
            issues.append("Limited chord vocabulary - consider adding more chords")
        
        # Check for weak chord detection
        weak_chords = sum(1 for chord in chords if 'No Chord' in chord or 'unclear' in chord)
        if weak_chords > len(chords) * 0.5:
            issues.append("Weak harmonic content - chords are not clearly defined")
        
        # Check average confidence
        avg_confidence = np.mean(confidences)
        if avg_confidence < 0.2:
            issues.append("Low chord detection confidence - harmonic content may be too sparse")
        
        return issues
    
    def _analyze_chord_progressions(self, chord_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze chord progressions and movement"""
        chords = chord_analysis.get('detected_chords', [])
        
        if len(chords) < 4:
            return {
                'progression_quality': 'poor',
                'common_progressions': [],
                'issues': ['Too few chords for progression analysis']
            }
        
        # Extract chord progressions (sequences of 4 chords)
        progressions = []
        for i in range(len(chords) - 3):
            progression = tuple(chords[i:i+4])
            progressions.append(progression)
        
        # Find common progressions
        common_progressions = self._find_common_progressions(progressions)
        
        # Analyze progression quality
        progression_quality = self._assess_progression_quality(progressions, common_progressions)
        
        # Check for specific progression types
        progression_types = self._classify_progressions(progressions)
        
        return {
            'progressions': progressions[:10],  # Limit for storage
            'common_progressions': common_progressions,
            'progression_quality': progression_quality,
            'progression_types': progression_types,
            'issues': self._identify_progression_issues(progressions)
        }
    
    def _find_common_progressions(self, progressions: List[Tuple]) -> List[Dict]:
        """Find and rank common chord progressions"""
        from collections import Counter
        
        # Count progression frequency
        progression_counts = Counter(progressions)
        
        # Return most common progressions
        common = []
        for progression, count in progression_counts.most_common(5):
            common.append({
                'progression': list(progression),
                'frequency': count,
                'percentage': (count / len(progressions)) * 100 if progressions else 0
            })
        
        return common
    
    def _assess_progression_quality(self, progressions: List[Tuple], common_progressions: List[Dict]) -> str:
        """Assess chord progression quality"""
        if not progressions:
            return 'poor'
        
        # Check for variety in progressions
        unique_progressions = len(set(progressions))
        progression_variety = unique_progressions / len(progressions)
        
        # Check for recognizable patterns
        has_recognizable_patterns = any(prog['frequency'] > 1 for prog in common_progressions)
        
        if progression_variety > 0.7:
            return 'good' if has_recognizable_patterns else 'chaotic'
        elif progression_variety > 0.3:
            return 'fair'
        else:
            return 'repetitive'
    
    def _classify_progressions(self, progressions: List[Tuple]) -> List[str]:
        """Classify progression types"""
        # Simplified classification
        types = []
        
        for progression in progressions[:5]:  # Analyze first 5
            # Look for specific patterns
            if any('major' in chord for chord in progression):
                if any('minor' in chord for chord in progression):
                    types.append('mixed_major_minor')
                else:
                    types.append('predominantly_major')
            elif any('minor' in chord for chord in progression):
                types.append('predominantly_minor')
            else:
                types.append('unclear')
        
        return list(set(types))
    
    def _identify_progression_issues(self, progressions: List[Tuple]) -> List[str]:
        """Identify issues with chord progressions"""
        issues = []
        
        if not progressions:
            return ['No chord progressions detected']
        
        # Check for excessive repetition
        from collections import Counter
        progression_counts = Counter(progressions)
        most_common_freq = progression_counts.most_common(1)[0][1] if progression_counts else 0
        
        if most_common_freq > len(progressions) * 0.8:
            issues.append("Chord progression is too repetitive")
        
        # Check for harmonic movement
        static_chords = sum(1 for prog in progressions if len(set(prog)) == 1)
        if static_chords > len(progressions) * 0.5:
            issues.append("Too much static harmony - needs more chord movement")
        
        return issues
    
    def _analyze_harmonic_rhythm(self, chord_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze harmonic rhythm (rate of chord changes)"""
        chords = chord_analysis.get('detected_chords', [])
        
        if len(chords) < 2:
            return {
                'rhythm_quality': 'poor',
                'avg_chord_duration': 0.0,
                'rhythm_pattern': 'static'
            }
        
        # Calculate chord durations
        chord_durations = []
        current_chord = chords[0]
        duration = 1
        
        for chord in chords[1:]:
            if chord == current_chord:
                duration += 1
            else:
                chord_durations.append(duration)
                current_chord = chord
                duration = 1
        chord_durations.append(duration)  # Add last chord duration
        
        # Analyze rhythm patterns
        avg_duration = np.mean(chord_durations)
        duration_variance = np.var(chord_durations)
        
        # Classify rhythm pattern
        if duration_variance < 1:
            rhythm_pattern = 'regular'
        elif duration_variance < 4:
            rhythm_pattern = 'moderate_variation'
        else:
            rhythm_pattern = 'highly_varied'
        
        # Assess quality
        if 2 <= avg_duration <= 8:  # Good range for harmonic rhythm
            rhythm_quality = 'good'
        elif avg_duration < 2:
            rhythm_quality = 'too_fast'
        else:
            rhythm_quality = 'too_slow'
        
        return {
            'avg_chord_duration': float(avg_duration),
            'duration_variance': float(duration_variance),
            'rhythm_pattern': rhythm_pattern,
            'rhythm_quality': rhythm_quality,
            'chord_durations': chord_durations[:10]  # Limit for storage
        }
    
    def _analyze_voice_leading(self, chord_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze voice leading between chords"""
        chords = chord_analysis.get('detected_chords', [])
        
        if len(chords) < 2:
            return {
                'voice_leading_quality': 'unknown',
                'smooth_transitions': 0,
                'rough_transitions': 0
            }
        
        # Simplified voice leading analysis
        smooth_transitions = 0
        rough_transitions = 0
        
        # Note: This is a simplified analysis
        # Full voice leading analysis would require note-by-note tracking
        
        for i in range(len(chords) - 1):
            current_chord = chords[i]
            next_chord = chords[i + 1]
            
            if current_chord != next_chord:
                # Simple heuristic: chords with shared notes = smooth
                # This would be more sophisticated in a full implementation
                if self._chords_share_notes(current_chord, next_chord):
                    smooth_transitions += 1
                else:
                    rough_transitions += 1
        
        total_transitions = smooth_transitions + rough_transitions
        
        if total_transitions > 0:
            smooth_ratio = smooth_transitions / total_transitions
            if smooth_ratio > 0.7:
                voice_leading_quality = 'smooth'
            elif smooth_ratio > 0.4:
                voice_leading_quality = 'moderate'
            else:
                voice_leading_quality = 'rough'
        else:
            voice_leading_quality = 'static'
        
        return {
            'voice_leading_quality': voice_leading_quality,
            'smooth_transitions': smooth_transitions,
            'rough_transitions': rough_transitions,
            'smooth_ratio': smooth_ratio if total_transitions > 0 else 0.0
        }
    
    def _chords_share_notes(self, chord1: str, chord2: str) -> bool:
        """Check if two chords share common notes (simplified)"""
        # Extract root notes
        root1 = chord1.split()[0] if ' ' in chord1 else chord1
        root2 = chord2.split()[0] if ' ' in chord2 else chord2
        
        # Simplified: chords with same root or related roots share notes
        note_circle = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']
        
        try:
            idx1 = note_circle.index(root1)
            idx2 = note_circle.index(root2)
            distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
            return distance <= 2  # Adjacent notes in circle of fifths
        except ValueError:
            return False  # If notes not found, assume no relationship
    
    def _analyze_harmonic_complexity(self, harmonic_content: Dict[str, np.ndarray], chord_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze harmonic complexity"""
        chroma = harmonic_content['chroma']
        chords = chord_analysis.get('detected_chords', [])
        
        # Calculate harmonic complexity metrics
        if chroma.shape[1] > 0:
            # Spectral complexity (how many different pitches are active)
            avg_active_notes = np.mean(np.sum(chroma > 0.1, axis=0))
            
            # Harmonic entropy (how evenly distributed the pitch content is)
            frame_entropies = []
            for frame in range(chroma.shape[1]):
                frame_chroma = chroma[:, frame]
                if np.sum(frame_chroma) > 0:
                    normalized = frame_chroma / np.sum(frame_chroma)
                    entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
                    frame_entropies.append(entropy)
            
            avg_entropy = np.mean(frame_entropies) if frame_entropies else 0.0
        else:
            avg_active_notes = 0.0
            avg_entropy = 0.0
        
        # Chord vocabulary complexity
        unique_chords = len(set(chords)) if chords else 0
        
        # Overall complexity assessment
        if avg_active_notes > 4 and unique_chords > 4:
            complexity_level = 'high'
        elif avg_active_notes > 2 and unique_chords > 2:
            complexity_level = 'moderate'
        else:
            complexity_level = 'low'
        
        return {
            'complexity_level': complexity_level,
            'avg_active_notes': float(avg_active_notes),
            'harmonic_entropy': float(avg_entropy),
            'chord_vocabulary_size': unique_chords
        }
    
    def _analyze_tension_resolution(self, chord_analysis: Dict[str, any]) -> Dict[str, any]:
        """Analyze tension and resolution in chord progressions"""
        chords = chord_analysis.get('detected_chords', [])
        
        if len(chords) < 4:
            return {
                'tension_resolution_quality': 'insufficient_data',
                'tension_points': 0,
                'resolution_points': 0
            }
        
        # Simplified tension/resolution analysis
        # In a full implementation, this would use music theory rules
        
        tension_points = 0
        resolution_points = 0
        
        # Look for patterns that suggest tension and resolution
        for i in range(len(chords) - 1):
            current_chord = chords[i]
            next_chord = chords[i + 1]
            
            # Simple heuristic: minor chords create tension, major resolve
            if 'minor' in current_chord and 'major' in next_chord:
                resolution_points += 1
            elif 'major' in current_chord and 'minor' in next_chord:
                tension_points += 1
        
        total_points = tension_points + resolution_points
        
        if total_points > 0:
            if tension_points > 0 and resolution_points > 0:
                quality = 'good_balance'
            elif resolution_points > tension_points:
                quality = 'resolution_heavy'
            else:
                quality = 'tension_heavy'
        else:
            quality = 'static'
        
        return {
            'tension_resolution_quality': quality,
            'tension_points': tension_points,
            'resolution_points': resolution_points,
            'balance_ratio': resolution_points / max(tension_points, 1)
        }
    
    def _assess_overall_harmony(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Assess overall harmonic quality"""
        chord_quality = analysis_results.get('chord_analysis', {}).get('chord_quality', 'poor')
        progression_quality = analysis_results.get('progression_analysis', {}).get('progression_quality', 'poor')
        complexity_level = analysis_results.get('complexity', {}).get('complexity_level', 'low')
        
        # Calculate overall harmony score
        quality_scores = {
            'excellent': 4,
            'good': 3,
            'fair': 2,
            'poor': 1
        }
        
        chord_score = quality_scores.get(chord_quality, 1)
        progression_score = quality_scores.get(progression_quality, 1)
        
        complexity_scores = {'high': 3, 'moderate': 2, 'low': 1}
        complexity_score = complexity_scores.get(complexity_level, 1)
        
        overall_score = (chord_score + progression_score + complexity_score) / 3
        
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
            'harmony_score': float(overall_score),
            'strengths': self._identify_harmony_strengths(analysis_results),
            'weaknesses': self._identify_harmony_weaknesses(analysis_results)
        }
    
    def _identify_harmony_strengths(self, analysis_results: Dict[str, any]) -> List[str]:
        """Identify harmonic strengths"""
        strengths = []
        
        chord_quality = analysis_results.get('chord_analysis', {}).get('chord_quality', 'poor')
        if chord_quality in ['excellent', 'good']:
            strengths.append('Clear and well-defined chord progressions')
        
        complexity_level = analysis_results.get('complexity', {}).get('complexity_level', 'low')
        if complexity_level in ['moderate', 'high']:
            strengths.append('Good harmonic complexity and sophistication')
        
        voice_leading = analysis_results.get('voice_leading', {}).get('voice_leading_quality', 'rough')
        if voice_leading == 'smooth':
            strengths.append('Smooth voice leading between chords')
        
        return strengths
    
    def _identify_harmony_weaknesses(self, analysis_results: Dict[str, any]) -> List[str]:
        """Identify harmonic weaknesses"""
        weaknesses = []
        
        chord_issues = analysis_results.get('chord_analysis', {}).get('issues', [])
        weaknesses.extend(chord_issues)
        
        progression_issues = analysis_results.get('progression_analysis', {}).get('issues', [])
        weaknesses.extend(progression_issues)
        
        complexity_level = analysis_results.get('complexity', {}).get('complexity_level', 'low')
        if complexity_level == 'low':
            weaknesses.append('Harmonic content could be more sophisticated')
        
        return weaknesses
    
    def _generate_harmony_recommendations(self, analysis_results: Dict[str, any], key: Optional[str] = None) -> Dict[str, any]:
        """Generate specific harmony improvement recommendations"""
        recommendations = []
        beginner_tips = []
        chord_suggestions = []
        
        # Analyze issues and generate recommendations
        chord_analysis = analysis_results.get('chord_analysis', {})
        chord_issues = chord_analysis.get('issues', [])
        
        for issue in chord_issues:
            if 'one chord' in issue:
                recommendations.append({
                    'priority': 'high',
                    'area': 'chord_variety',
                    'issue': issue,
                    'solution': 'Add more chords to create harmonic movement',
                    'beginner_tip': 'Try the I-V-vi-IV progression in your key'
                })
            elif 'weak harmonic content' in issue:
                recommendations.append({
                    'priority': 'high',
                    'area': 'harmonic_strength',
                    'issue': issue,
                    'solution': 'Use stronger, more defined chord sounds',
                    'beginner_tip': 'Layer multiple instruments playing the same chords'
                })
        
        # Generate chord suggestions based on key
        if key:
            chord_suggestions = self._suggest_chords_for_key(key)
        else:
            chord_suggestions = ['C major', 'F major', 'G major', 'A minor']  # Default suggestions
        
        # Generate beginner tips
        beginner_tips = [
            "Start with simple triads (three-note chords)",
            "Learn the I-V-vi-IV progression - it works in any key",
            "Use sus chords (sus2, sus4) to add interest",
            "Try inversions to create smoother bass lines",
            "Layer different instruments playing the same chords for fullness"
        ]
        
        # Advanced techniques
        advanced_techniques = [
            "Experiment with extended chords (7ths, 9ths, 11ths)",
            "Use modal interchange (borrowing chords from parallel modes)",
            "Try secondary dominants for temporary key centers",
            "Use pedal tones to create harmonic tension",
            "Experiment with non-functional harmony"
        ]
        
        return {
            'recommendations': recommendations,
            'beginner_tips': beginner_tips,
            'chord_suggestions': chord_suggestions,
            'advanced_techniques': advanced_techniques,
            'next_steps': self._generate_harmony_next_steps(recommendations)
        }
    
    def _suggest_chords_for_key(self, key: str) -> List[str]:
        """Suggest appropriate chords for a given key"""
        if not key or 'unknown' in key.lower():
            return ['C major', 'F major', 'G major', 'A minor']
        
        try:
            root, mode = key.split()
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            root_index = note_names.index(root)
            
            if mode.lower() == 'major':
                # Major key chord suggestions (I, ii, iii, IV, V, vi)
                chord_indices = [0, 2, 4, 5, 7, 9]  # Scale degrees
                chord_qualities = ['major', 'minor', 'minor', 'major', 'major', 'minor']
            else:  # minor key
                # Minor key chord suggestions (i, ii°, III, iv, v, VI)
                chord_indices = [0, 2, 3, 5, 7, 8]
                chord_qualities = ['minor', 'diminished', 'major', 'minor', 'minor', 'major']
            
            suggestions = []
            for i, quality in zip(chord_indices, chord_qualities):
                chord_root = note_names[(root_index + i) % 12]
                suggestions.append(f"{chord_root} {quality}")
            
            return suggestions[:4]  # Return top 4 suggestions
            
        except (ValueError, IndexError):
            return ['C major', 'F major', 'G major', 'A minor']
    
    def _generate_harmony_next_steps(self, recommendations: List[Dict]) -> List[str]:
        """Generate actionable next steps for harmony improvement"""
        next_steps = []
        
        high_priority = [rec for rec in recommendations if rec.get('priority') == 'high']
        if high_priority:
            next_steps.append(f"1. {high_priority[0]['solution']}")
        
        next_steps.extend([
            "2. Learn basic chord progressions (I-V-vi-IV, vi-IV-I-V)",
            "3. Practice chord inversions for smoother voice leading",
            "4. Study songs in your genre to understand common progressions",
            "5. Experiment with one new chord type per week"
        ])
        
        return next_steps[:5]
    
    def _load_chord_templates(self) -> Dict[str, any]:
        """Load chord template database"""
        # Placeholder for chord template database
        return {
            'major': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Root, major third, fifth
            'minor': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Root, minor third, fifth
            'dominant7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # Root, major third, fifth, minor seventh
        }
    
    def _load_progression_templates(self) -> Dict[str, any]:
        """Load common progression templates"""
        # Placeholder for progression template database
        return {
            'pop_progression': ['I', 'V', 'vi', 'IV'],
            'circle_of_fifths': ['I', 'vi', 'ii', 'V'],
            'blues_progression': ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V']
        }