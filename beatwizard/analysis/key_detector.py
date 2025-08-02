"""
Enhanced Key Detection - Professional musical key analysis
Advanced key detection using Krumhansl-Schmuckler algorithm and chroma analysis
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class KeyDetector:
    """
    Professional key detection using multiple algorithms
    Implements Krumhansl-Schmuckler algorithm and advanced chroma analysis
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """
        Initialize the key detector
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        
        # Musical notes and keys
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.keys = []
        
        # Generate all major and minor keys
        for note in self.notes:
            self.keys.extend([f"{note} major", f"{note} minor"])
        
        # Krumhansl-Schmuckler profiles
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        self.major_profile = self.major_profile / np.sum(self.major_profile)
        self.minor_profile = self.minor_profile / np.sum(self.minor_profile)
        
        logger.debug("KeyDetector initialized with Krumhansl-Schmuckler profiles")
    
    def detect_key(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive key detection with multiple methods
        
        Args:
            audio: Audio data (mono)
            
        Returns:
            Dictionary with key analysis results
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        logger.debug("Starting comprehensive key analysis")
        
        # Extract chroma features
        chroma_features = self._extract_chroma_features(audio)
        
        # Method 1: Krumhansl-Schmuckler algorithm
        ks_result = self._krumhansl_schmuckler_analysis(chroma_features)
        
        # Method 2: Template matching
        template_result = self._template_matching_analysis(chroma_features)
        
        # Method 3: HPCP (Harmonic Pitch Class Profile) analysis
        hpcp_result = self._hpcp_analysis(audio)
        
        # Method 4: Chord progression analysis
        chord_result = self._chord_progression_analysis(chroma_features)
        
        # Combine results and calculate confidence
        combined_result = self._combine_key_results([ks_result, template_result, hpcp_result])
        
        # Key stability analysis
        stability_analysis = self._analyze_key_stability(audio)
        
        # Musical context analysis
        context_analysis = self._analyze_musical_context(chroma_features, combined_result)
        
        result = {
            'primary_key': combined_result['key'],
            'confidence': combined_result['confidence'],
            'key_methods': {
                'krumhansl_schmuckler': ks_result,
                'template_matching': template_result,
                'hpcp': hpcp_result,
                'chord_progression': chord_result
            },
            'stability_analysis': stability_analysis,
            'context_analysis': context_analysis,
            'chroma_profile': chroma_features['mean_chroma'].tolist(),
            'alternative_keys': combined_result.get('alternatives', [])
        }
        
        logger.info(f"Key detected: {combined_result['key']} (confidence: {combined_result['confidence']:.2f})")
        
        return result
    
    def _extract_chroma_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive chroma features"""
        try:
            # Standard chroma
            chroma_stft = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # CQT-based chroma (higher frequency resolution)
            chroma_cqt = librosa.feature.chroma_cqt(
                y=audio,
                sr=self.sample_rate,                
                hop_length=self.hop_length
            )
            
            # CENS chroma (robust to timbre)
            chroma_cens = librosa.feature.chroma_cens(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Calculate mean chroma profile
            mean_chroma = np.mean(chroma_cqt, axis=1)
            
            # Normalize
            if np.sum(mean_chroma) > 0:
                mean_chroma = mean_chroma / np.sum(mean_chroma)
            
            return {
                'chroma_stft': chroma_stft,
                'chroma_cqt': chroma_cqt,
                'chroma_cens': chroma_cens,
                'mean_chroma': mean_chroma
            }
            
        except Exception as e:
            logger.error(f"Chroma extraction failed: {e}")
            # Return empty result
            empty_chroma = np.zeros(12)
            return {
                'chroma_stft': np.zeros((12, 1)),
                'chroma_cqt': np.zeros((12, 1)),
                'chroma_cens': np.zeros((12, 1)),
                'mean_chroma': empty_chroma
            }
    
    def _krumhansl_schmuckler_analysis(self, chroma_features: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Krumhansl-Schmuckler key finding algorithm"""
        chroma_profile = chroma_features['mean_chroma']
        
        if np.sum(chroma_profile) == 0:
            return {'key': 'Unknown', 'confidence': 0.0, 'correlations': []}
        
        correlations = []
        
        # Test all 24 keys (12 major + 12 minor)
        for i in range(12):
            # Major key correlation
            major_template = np.roll(self.major_profile, i)
            major_corr = np.corrcoef(chroma_profile, major_template)[0, 1]
            if not np.isnan(major_corr):
                correlations.append({
                    'key': f"{self.notes[i]} major",
                    'correlation': major_corr,
                    'type': 'major'
                })
            
            # Minor key correlation
            minor_template = np.roll(self.minor_profile, i)
            minor_corr = np.corrcoef(chroma_profile, minor_template)[0, 1]
            if not np.isnan(minor_corr):
                correlations.append({
                    'key': f"{self.notes[i]} minor",
                    'correlation': minor_corr,
                    'type': 'minor'
                })
        
        # Sort by correlation
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        if correlations:
            best_key = correlations[0]['key']
            confidence = correlations[0]['correlation']
            # Normalize confidence to 0-1 range
            confidence = max(0.0, confidence)
        else:
            best_key = 'Unknown'
            confidence = 0.0
        
        return {
            'key': best_key,
            'confidence': confidence,
            'correlations': correlations[:5]  # Top 5 candidates
        }
    
    def _template_matching_analysis(self, chroma_features: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Template matching key detection"""
        chroma_profile = chroma_features['mean_chroma']
        
        if np.sum(chroma_profile) == 0:
            return {'key': 'Unknown', 'confidence': 0.0}
        
        # Simplified template matching using tonic emphasis
        key_scores = []
        
        for i in range(12):
            # Major key: emphasize tonic, dominant, and mediant
            major_weights = np.zeros(12)
            major_weights[i] = 3.0  # Tonic
            major_weights[(i + 4) % 12] = 2.0  # Mediant
            major_weights[(i + 7) % 12] = 2.5  # Dominant
            major_weights[(i + 9) % 12] = 1.0  # Submediant
            
            major_score = np.dot(chroma_profile, major_weights)
            key_scores.append({
                'key': f"{self.notes[i]} major",
                'score': major_score,
                'type': 'major'
            })
            
            # Minor key: emphasize tonic, mediant, and dominant
            minor_weights = np.zeros(12)
            minor_weights[i] = 3.0  # Tonic
            minor_weights[(i + 3) % 12] = 2.0  # Minor mediant
            minor_weights[(i + 7) % 12] = 2.5  # Dominant
            minor_weights[(i + 8) % 12] = 1.0  # Minor submediant
            
            minor_score = np.dot(chroma_profile, minor_weights)
            key_scores.append({
                'key': f"{self.notes[i]} minor",
                'score': minor_score,
                'type': 'minor'
            })
        
        # Sort by score
        key_scores.sort(key=lambda x: x['score'], reverse=True)
        
        if key_scores:
            best_key = key_scores[0]['key']
            # Normalize confidence
            max_score = key_scores[0]['score']
            confidence = min(1.0, max_score / np.sum(chroma_profile)) if np.sum(chroma_profile) > 0 else 0.0
        else:
            best_key = 'Unknown'
            confidence = 0.0
        
        return {
            'key': best_key,
            'confidence': confidence,
            'top_candidates': key_scores[:3]
        }
    
    def _hpcp_analysis(self, audio: np.ndarray) -> Dict[str, any]:
        """Harmonic Pitch Class Profile analysis"""
        try:
            # Use CQT for better harmonic resolution
            C = np.abs(librosa.cqt(
                audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=84,  # 7 octaves
                bins_per_octave=12
            ))
            
            # Sum across octaves to get pitch class profile
            hpcp = np.zeros(12)
            for i in range(C.shape[0]):
                hpcp[i % 12] += np.mean(C[i, :])
            
            # Normalize
            if np.sum(hpcp) > 0:
                hpcp = hpcp / np.sum(hpcp)
            
            # Find key using template matching
            correlations = []
            for i in range(12):
                # Major
                major_template = np.roll(self.major_profile, i)
                major_corr = np.corrcoef(hpcp, major_template)[0, 1]
                if not np.isnan(major_corr):
                    correlations.append((f"{self.notes[i]} major", major_corr))
                
                # Minor
                minor_template = np.roll(self.minor_profile, i)
                minor_corr = np.corrcoef(hpcp, minor_template)[0, 1]
                if not np.isnan(minor_corr):
                    correlations.append((f"{self.notes[i]} minor", minor_corr))
            
            if correlations:
                correlations.sort(key=lambda x: x[1], reverse=True)
                best_key = correlations[0][0]
                confidence = max(0.0, correlations[0][1])
            else:
                best_key = 'Unknown'
                confidence = 0.0
            
            return {
                'key': best_key,
                'confidence': confidence,
                'hpcp_profile': hpcp.tolist()
            }
            
        except Exception as e:
            logger.warning(f"HPCP analysis failed: {e}")
            return {'key': 'Unknown', 'confidence': 0.0, 'hpcp_profile': []}
    
    def _chord_progression_analysis(self, chroma_features: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Analyze chord progressions to infer key"""
        try:
            chroma = chroma_features['chroma_cqt']
            
            # Simple chord detection based on chroma peaks
            chord_sequence = []
            
            for frame in range(chroma.shape[1]):
                frame_chroma = chroma[:, frame]
                if np.max(frame_chroma) > 0.1:  # Threshold for significant activity
                    # Find dominant pitch classes
                    peaks = frame_chroma > (np.max(frame_chroma) * 0.6)
                    chord_sequence.append(peaks)
            
            if not chord_sequence:
                return {'key': 'Unknown', 'confidence': 0.0}
            
            # Analyze chord transitions (simplified)
            # Look for common progressions that suggest keys
            key_votes = {}
            
            for chord in chord_sequence:
                chord_notes = [i for i, active in enumerate(chord) if active]
                
                # Vote for keys based on chord content
                for i in range(12):
                    # Major key chord analysis
                    major_chords = [i, (i + 2) % 12, (i + 4) % 12, (i + 5) % 12, (i + 7) % 12, (i + 9) % 12, (i + 11) % 12]
                    major_votes = sum(1 for note in chord_notes if note in major_chords)
                    
                    key_name = f"{self.notes[i]} major"
                    if key_name not in key_votes:
                        key_votes[key_name] = 0
                    key_votes[key_name] += major_votes
                    
                    # Minor key chord analysis  
                    minor_chords = [i, (i + 2) % 12, (i + 3) % 12, (i + 5) % 12, (i + 7) % 12, (i + 8) % 12, (i + 10) % 12]
                    minor_votes = sum(1 for note in chord_notes if note in minor_chords)
                    
                    key_name = f"{self.notes[i]} minor"
                    if key_name not in key_votes:
                        key_votes[key_name] = 0
                    key_votes[key_name] += minor_votes
            
            if key_votes:
                best_key = max(key_votes, key=key_votes.get)
                max_votes = key_votes[best_key]
                total_votes = sum(key_votes.values())
                confidence = max_votes / total_votes if total_votes > 0 else 0.0
            else:
                best_key = 'Unknown'
                confidence = 0.0
            
            return {
                'key': best_key,
                'confidence': confidence,
                'chord_analysis': dict(sorted(key_votes.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            logger.warning(f"Chord progression analysis failed: {e}")
            return {'key': 'Unknown', 'confidence': 0.0}
    
    def _combine_key_results(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Combine results from multiple key detection methods"""
        # Filter out failed results
        valid_results = [r for r in results if r['key'] != 'Unknown' and r['confidence'] > 0]
        
        if not valid_results:
            return {'key': 'Unknown', 'confidence': 0.0, 'alternatives': []}
        
        # Weighted voting system
        key_scores = {}
        
        for i, result in enumerate(valid_results):
            key = result['key']
            confidence = result['confidence']
            
            # Weight by method reliability (KS algorithm gets highest weight)
            weights = [1.0, 0.8, 0.9, 0.7]  # KS, Template, HPCP, Chord
            weight = weights[i] if i < len(weights) else 0.5
            
            score = confidence * weight
            
            if key not in key_scores:
                key_scores[key] = 0
            key_scores[key] += score
        
        # Normalize scores
        total_score = sum(key_scores.values())
        if total_score > 0:
            key_scores = {k: v/total_score for k, v in key_scores.items()}
        
        # Sort by score
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_keys:
            primary_key = sorted_keys[0][0]
            confidence = sorted_keys[0][1]
            alternatives = [{'key': k, 'confidence': c} for k, c in sorted_keys[1:4]]
        else:
            primary_key = 'Unknown'
            confidence = 0.0
            alternatives = []
        
        return {
            'key': primary_key,
            'confidence': confidence,
            'alternatives': alternatives
        }
    
    def _analyze_key_stability(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze key stability throughout the track"""
        try:
            segment_length = 15.0  # 15 seconds per segment
            segment_samples = int(segment_length * self.sample_rate)
            
            if len(audio) < segment_samples:
                return {'is_stable': True, 'key_changes': [], 'stability_score': 1.0}
            
            segment_keys = []
            
            for start in range(0, len(audio) - segment_samples, segment_samples // 2):
                segment = audio[start:start + segment_samples]
                
                # Quick key detection for segment
                chroma_features = self._extract_chroma_features(segment)
                ks_result = self._krumhansl_schmuckler_analysis(chroma_features)
                
                if ks_result['confidence'] > 0.3:  # Only consider confident detections
                    segment_keys.append(ks_result['key'])
            
            if len(segment_keys) < 2:
                return {'is_stable': True, 'key_changes': [], 'stability_score': 1.0}
            
            # Find key changes
            key_changes = []
            current_key = segment_keys[0]
            
            for i, key in enumerate(segment_keys[1:], 1):
                if key != current_key:
                    key_changes.append({
                        'from_key': current_key,
                        'to_key': key,
                        'time_position': i * segment_length / 2  # Approximate time
                    })
                    current_key = key
            
            # Calculate stability score
            unique_keys = len(set(segment_keys))
            stability_score = 1.0 / unique_keys if unique_keys > 0 else 0.0
            
            is_stable = len(key_changes) <= 1  # Allow one modulation
            
            return {
                'is_stable': is_stable,
                'key_changes': key_changes,
                'stability_score': stability_score,
                'segment_keys': segment_keys,
                'dominant_key': max(set(segment_keys), key=segment_keys.count) if segment_keys else 'Unknown'
            }
            
        except Exception as e:
            logger.warning(f"Key stability analysis failed: {e}")
            return {'is_stable': True, 'key_changes': [], 'stability_score': 1.0}
    
    def _analyze_musical_context(self, chroma_features: Dict[str, np.ndarray], key_result: Dict[str, any]) -> Dict[str, any]:
        """Analyze musical context and characteristics"""
        try:
            chroma_profile = chroma_features['mean_chroma']
            
            # Calculate tonal clarity (how clearly defined the key is)
            if np.sum(chroma_profile) > 0:
                normalized_chroma = chroma_profile / np.sum(chroma_profile)
                entropy = -np.sum(normalized_chroma * np.log2(normalized_chroma + 1e-10))
                tonal_clarity = 1.0 - (entropy / np.log2(12))  # Normalize to 0-1
            else:
                tonal_clarity = 0.0
            
            # Detect modality (major/minor tendency)
            major_weight = np.sum(chroma_profile[[0, 2, 4, 5, 7, 9, 11]])  # Major scale notes
            minor_weight = np.sum(chroma_profile[[0, 2, 3, 5, 7, 8, 10]])  # Minor scale notes
            
            if major_weight + minor_weight > 0:
                modality_score = (major_weight - minor_weight) / (major_weight + minor_weight)
            else:
                modality_score = 0.0
            
            # Determine modality
            if modality_score > 0.1:
                modality = 'major'
            elif modality_score < -0.1:
                modality = 'minor'
            else:
                modality = 'ambiguous'
            
            # Calculate consonance (how harmonious the overall sound is)
            consonant_intervals = [0, 3, 4, 7, 8, 9]  # Perfect unison, major/minor thirds, fifth, etc.
            consonance_score = np.sum(chroma_profile[consonant_intervals]) / np.sum(chroma_profile) if np.sum(chroma_profile) > 0 else 0.0
            
            return {
                'tonal_clarity': float(tonal_clarity),
                'modality': modality,
                'modality_score': float(modality_score),
                'consonance_score': float(consonance_score),
                'key_strength': float(key_result.get('confidence', 0.0)),
                'harmonic_complexity': float(entropy) if 'entropy' in locals() else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Musical context analysis failed: {e}")
            return {
                'tonal_clarity': 0.0,
                'modality': 'unknown',
                'modality_score': 0.0,
                'consonance_score': 0.0,
                'key_strength': 0.0,
                'harmonic_complexity': 0.0
            }
    
    def suggest_key_adjustments(self, key_result: Dict[str, any]) -> Dict[str, any]:
        """Suggest key adjustments for better mix compatibility"""
        primary_key = key_result.get('primary_key', 'Unknown')
        confidence = key_result.get('confidence', 0.0)
        
        if primary_key == 'Unknown':
            return {'suggestions': [], 'reasoning': 'No valid key detected'}
        
        suggestions = []
        reasoning = []
        
        # Parse key
        try:
            note, mode = primary_key.split(' ')
            note_index = self.notes.index(note)
        except (ValueError, IndexError):
            return {'suggestions': [], 'reasoning': 'Invalid key format'}
        
        # DJ-friendly keys (easier to mix)
        dj_friendly_keys = ['C major', 'G major', 'D major', 'A major', 'E major', 'F major',
                           'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor', 'D minor']
        
        if primary_key not in dj_friendly_keys:
            # Find relative or parallel keys that are DJ-friendly
            if mode == 'major':
                relative_minor = f"{self.notes[(note_index + 9) % 12]} minor"
                if relative_minor in dj_friendly_keys:
                    suggestions.append({
                        'type': 'relative_key',
                        'target_key': relative_minor,
                        'relationship': 'relative_minor'
                    })
            else:  # minor
                relative_major = f"{self.notes[(note_index + 3) % 12]} major"
                if relative_major in dj_friendly_keys:
                    suggestions.append({
                        'type': 'relative_key',
                        'target_key': relative_major,
                        'relationship': 'relative_major'
                    })
        
        # Confidence-based suggestions
        if confidence < 0.6:
            suggestions.append({
                'type': 'confidence_improvement',
                'suggestion': 'Consider harmonic analysis - low key detection confidence',
                'current_confidence': confidence
            })
            reasoning.append("Low key detection confidence suggests further harmonic analysis needed")
        
        # Stability suggestions
        stability = key_result.get('stability_analysis', {})
        if not stability.get('is_stable', True):
            suggestions.append({
                'type': 'key_stability',
                'suggestion': 'Track contains key changes - consider section-based analysis',
                'key_changes': len(stability.get('key_changes', []))
            })
            reasoning.append("Key changes detected - may need different treatment per section")
        
        return {
            'suggestions': suggestions,
            'reasoning': reasoning,
            'current_key': primary_key,
            'confidence': confidence
        }