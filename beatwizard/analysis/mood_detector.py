"""
Enhanced Mood/Emotion Detection - AI-powered musical emotion analysis
Advanced emotion detection using spectral features and machine learning
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from loguru import logger
from scipy import stats
# Optional scikit-learn for advanced features
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None

from config.settings import audio_settings


class MoodDetector:
    """
    Professional mood and emotion detection for music
    Uses spectral features, tempo, and musical characteristics to classify emotional content
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """
        Initialize the mood detector
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        
        # Emotion categories based on Russell's Circumplex Model
        self.emotion_categories = {
            'valence': ['negative', 'neutral', 'positive'],
            'arousal': ['calm', 'moderate', 'energetic'],
            'dominance': ['submissive', 'balanced', 'dominant']
        }
        
        # Mood labels based on valence-arousal combinations
        self.mood_labels = {
            ('positive', 'energetic'): 'excited',
            ('positive', 'moderate'): 'happy',
            ('positive', 'calm'): 'peaceful',
            ('neutral', 'energetic'): 'intense',
            ('neutral', 'moderate'): 'neutral',
            ('neutral', 'calm'): 'relaxed',
            ('negative', 'energetic'): 'angry',
            ('negative', 'moderate'): 'sad',
            ('negative', 'calm'): 'melancholic'
        }
        
        # Musical features for emotion detection
        self.feature_extractors = {
            'spectral_centroid': self._extract_spectral_centroid,
            'spectral_rolloff': self._extract_spectral_rolloff,
            'spectral_bandwidth': self._extract_spectral_bandwidth,
            'zero_crossing_rate': self._extract_zero_crossing_rate,
            'mfcc': self._extract_mfcc,
            'chroma': self._extract_chroma,
            'tonnetz': self._extract_tonnetz,
            'tempo_features': self._extract_tempo_features,
            'rhythmic_features': self._extract_rhythmic_features,
            'harmonic_features': self._extract_harmonic_features
        }
        
        # Check for optional dependencies
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - using simplified mood detection")
        
        logger.debug("MoodDetector initialized")
    
    def detect_mood(self, audio: np.ndarray, tempo: Optional[float] = None, key: Optional[str] = None) -> Dict[str, any]:
        """
        Comprehensive mood and emotion detection
        
        Args:
            audio: Audio data (mono)
            tempo: Optional tempo information
            key: Optional key information
            
        Returns:
            Dictionary with mood analysis results
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        logger.debug("Starting comprehensive mood analysis")
        
        # Extract comprehensive features
        features = self._extract_all_features(audio)
        
        # Analyze emotional dimensions
        valence_analysis = self._analyze_valence(features, key)
        arousal_analysis = self._analyze_arousal(features, tempo)
        dominance_analysis = self._analyze_dominance(features)
        
        # Determine primary mood
        primary_mood = self._determine_primary_mood(valence_analysis, arousal_analysis)
        
        # Generate mood confidence scores
        mood_confidence = self._calculate_mood_confidence(features, primary_mood)
        
        # Analyze mood dynamics over time
        mood_dynamics = self._analyze_mood_dynamics(audio)
        
        # Generate musical context for mood
        musical_context = self._generate_musical_context(features, tempo, key)
        
        # Create mood recommendations
        mood_recommendations = self._generate_mood_recommendations(primary_mood, features)
        
        result = {
            'primary_mood': primary_mood,
            'mood_confidence': mood_confidence,
            'emotional_dimensions': {
                'valence': valence_analysis,
                'arousal': arousal_analysis,
                'dominance': dominance_analysis
            },
            'mood_dynamics': mood_dynamics,
            'musical_context': musical_context,
            'features': features,
            'recommendations': mood_recommendations,
            'mood_description': self._generate_mood_description(primary_mood, valence_analysis, arousal_analysis)
        }
        
        logger.info(f"Mood detected: {primary_mood} (confidence: {mood_confidence:.2f})")
        
        return result
    
    def _extract_all_features(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract comprehensive features for mood analysis"""
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(audio)
            except Exception as e:
                logger.warning(f"Failed to extract {feature_name}: {e}")
                features[feature_name] = None
        
        return features
    
    def _extract_spectral_centroid(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral centroid features (brightness)"""
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        return {
            'mean': np.mean(centroid),
            'std': np.std(centroid),
            'median': np.median(centroid),
            'range': np.max(centroid) - np.min(centroid)
        }
    
    def _extract_spectral_rolloff(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral rolloff features (energy distribution)"""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        return {
            'mean': np.mean(rolloff),
            'std': np.std(rolloff),
            'median': np.median(rolloff)
        }
    
    def _extract_spectral_bandwidth(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral bandwidth features"""
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        return {
            'mean': np.mean(bandwidth),
            'std': np.std(bandwidth),
            'median': np.median(bandwidth)
        }
    
    def _extract_zero_crossing_rate(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract zero crossing rate (roughness/smoothness)"""
        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        return {
            'mean': np.mean(zcr),
            'std': np.std(zcr),
            'median': np.median(zcr)
        }
    
    def _extract_mfcc(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract MFCC features (timbral characteristics)"""
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )
        
        # Statistical measures for each MFCC coefficient
        mfcc_stats = {}
        for i in range(mfccs.shape[0]):
            mfcc_stats[f'mfcc_{i+1}'] = {
                'mean': np.mean(mfccs[i]),
                'std': np.std(mfccs[i]),
                'median': np.median(mfccs[i])
            }
        
        return {
            'coefficients': mfcc_stats,
            'delta_mfcc': self._calculate_delta_features(mfccs),
            'overall_timbre': self._analyze_overall_timbre(mfccs)
        }
    
    def _extract_chroma(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract chroma features (harmonic content)"""
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'mean_chroma': np.mean(chroma, axis=1).tolist(),
            'std_chroma': np.std(chroma, axis=1).tolist(),
            'chroma_variance': np.var(chroma),
            'harmonic_stability': self._calculate_harmonic_stability(chroma)
        }
    
    def _extract_tonnetz(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract tonnetz features (tonal space)"""
        tonnetz = librosa.feature.tonnetz(
            y=audio, sr=self.sample_rate
        )
        
        return {
            'mean_tonnetz': np.mean(tonnetz, axis=1).tolist(),
            'std_tonnetz': np.std(tonnetz, axis=1).tolist(),
            'tonal_complexity': np.var(tonnetz)
        }
    
    def _extract_tempo_features(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract tempo-related features for emotion"""
        tempo, beats = librosa.beat.beat_track(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Beat consistency
        beat_intervals = np.diff(beats)
        beat_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)) if len(beat_intervals) > 0 else 0.0
        
        return {
            'tempo': tempo,
            'beat_consistency': beat_consistency,
            'tempo_category': self._categorize_tempo(tempo),
            'rhythmic_energy': self._calculate_rhythmic_energy(audio, beats)
        }
    
    def _extract_rhythmic_features(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract rhythmic complexity features"""
        onset_strength = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'onset_density': len(librosa.onset.onset_detect(onset_strength=onset_strength)) / (len(audio) / self.sample_rate),
            'rhythmic_complexity': np.std(onset_strength),
            'pulse_clarity': self._calculate_pulse_clarity(onset_strength)
        }
    
    def _extract_harmonic_features(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract harmonic features for emotion"""
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Harmonic energy ratio
        harmonic_energy = np.sum(harmonic ** 2)
        total_energy = np.sum(audio ** 2)
        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0
        
        return {
            'harmonic_ratio': harmonic_ratio,
            'harmonic_complexity': self._calculate_harmonic_complexity(harmonic),
            'consonance_score': self._calculate_consonance_score(audio)
        }
    
    def _analyze_valence(self, features: Dict[str, any], key: Optional[str] = None) -> Dict[str, any]:
        """Analyze valence (positive/negative emotion)"""
        valence_score = 0.0
        valence_factors = {}
        
        # Spectral brightness (higher = more positive)
        if features.get('spectral_centroid'):
            brightness = features['spectral_centroid']['mean']
            brightness_normalized = min(brightness / 3000, 1.0)  # Normalize to 0-1
            valence_score += brightness_normalized * 0.3
            valence_factors['brightness'] = brightness_normalized
        
        # Harmonic content (major keys tend to be more positive)
        if key and features.get('harmonic_features'):
            key_valence = self._get_key_valence(key)
            valence_score += key_valence * 0.2
            valence_factors['key_valence'] = key_valence
        
        # Consonance (more consonant = more positive)
        if features.get('harmonic_features'):
            consonance = features['harmonic_features'].get('consonance_score', 0.5)
            valence_score += consonance * 0.2
            valence_factors['consonance'] = consonance
        
        # MFCC characteristics (timbral warmth)
        if features.get('mfcc'):
            timbre_warmth = self._calculate_timbre_warmth(features['mfcc'])
            valence_score += timbre_warmth * 0.3
            valence_factors['timbre_warmth'] = timbre_warmth
        
        # Normalize to 0-1
        valence_score = np.clip(valence_score, 0.0, 1.0)
        
        # Categorize valence
        if valence_score > 0.6:
            valence_category = 'positive'
        elif valence_score > 0.4:
            valence_category = 'neutral'
        else:
            valence_category = 'negative'
        
        return {
            'score': valence_score,
            'category': valence_category,
            'confidence': self._calculate_dimension_confidence(valence_score),
            'factors': valence_factors
        }
    
    def _analyze_arousal(self, features: Dict[str, any], tempo: Optional[float] = None) -> Dict[str, any]:
        """Analyze arousal (energy/activation level)"""
        arousal_score = 0.0
        arousal_factors = {}
        
        # Tempo-based arousal
        if tempo or features.get('tempo_features'):
            track_tempo = tempo or features['tempo_features'].get('tempo', 120)
            tempo_arousal = self._tempo_to_arousal(track_tempo)
            arousal_score += tempo_arousal * 0.4
            arousal_factors['tempo'] = tempo_arousal
        
        # Spectral energy and brightness
        if features.get('spectral_centroid') and features.get('spectral_rolloff'):
            energy_arousal = min(features['spectral_rolloff']['mean'] / 8000, 1.0)
            arousal_score += energy_arousal * 0.3
            arousal_factors['spectral_energy'] = energy_arousal
        
        # Rhythmic complexity and onset density
        if features.get('rhythmic_features'):
            rhythmic_arousal = min(features['rhythmic_features'].get('onset_density', 0) / 10, 1.0)
            arousal_score += rhythmic_arousal * 0.3
            arousal_factors['rhythmic_activity'] = rhythmic_arousal
        
        # Normalize to 0-1
        arousal_score = np.clip(arousal_score, 0.0, 1.0)
        
        # Categorize arousal
        if arousal_score > 0.7:
            arousal_category = 'energetic'
        elif arousal_score > 0.4:
            arousal_category = 'moderate'
        else:
            arousal_category = 'calm'
        
        return {
            'score': arousal_score,
            'category': arousal_category,
            'confidence': self._calculate_dimension_confidence(arousal_score),
            'factors': arousal_factors
        }
    
    def _analyze_dominance(self, features: Dict[str, any]) -> Dict[str, any]:
        """Analyze dominance (control/power in the music)"""
        dominance_score = 0.5  # Default neutral
        dominance_factors = {}
        
        # Dynamic range (wider range = more dominant)
        if features.get('spectral_centroid'):
            dynamic_range = features['spectral_centroid']['range']
            range_dominance = min(dynamic_range / 4000, 1.0)
            dominance_score += (range_dominance - 0.5) * 0.4
            dominance_factors['dynamic_range'] = range_dominance
        
        # Harmonic stability (less stable = more dominant/complex)
        if features.get('chroma'):
            stability = features['chroma'].get('harmonic_stability', 0.5)
            dominance_score += (1.0 - stability) * 0.3
            dominance_factors['harmonic_complexity'] = 1.0 - stability
        
        # Rhythmic strength
        if features.get('tempo_features'):
            rhythm_strength = features['tempo_features'].get('beat_consistency', 0.5)
            dominance_score += rhythm_strength * 0.3
            dominance_factors['rhythmic_strength'] = rhythm_strength
        
        # Normalize to 0-1
        dominance_score = np.clip(dominance_score, 0.0, 1.0)
        
        # Categorize dominance
        if dominance_score > 0.6:
            dominance_category = 'dominant'
        elif dominance_score > 0.4:
            dominance_category = 'balanced'
        else:
            dominance_category = 'submissive'
        
        return {
            'score': dominance_score,
            'category': dominance_category,
            'confidence': self._calculate_dimension_confidence(dominance_score),
            'factors': dominance_factors
        }
    
    def _determine_primary_mood(self, valence: Dict[str, any], arousal: Dict[str, any]) -> str:
        """Determine primary mood from valence and arousal"""
        valence_cat = valence['category']
        arousal_cat = arousal['category']
        
        return self.mood_labels.get((valence_cat, arousal_cat), 'neutral')
    
    def _calculate_mood_confidence(self, features: Dict[str, any], mood: str) -> float:
        """Calculate confidence in mood prediction"""
        # Base confidence on feature extraction success
        feature_success_rate = sum(1 for f in features.values() if f is not None) / len(features)
        
        # Adjust based on clear emotional indicators
        confidence_boost = 0.0
        
        # Strong tempo indicators
        if features.get('tempo_features'):
            tempo = features['tempo_features'].get('tempo', 120)
            if tempo > 140 or tempo < 80:  # Clear fast or slow
                confidence_boost += 0.1
        
        # Clear spectral characteristics
        if features.get('spectral_centroid'):
            brightness = features['spectral_centroid']['mean']
            if brightness > 3000 or brightness < 1000:  # Very bright or dark
                confidence_boost += 0.1
        
        return min(feature_success_rate + confidence_boost, 1.0)
    
    def _analyze_mood_dynamics(self, audio: np.ndarray) -> Dict[str, any]:
        """Analyze how mood changes over time"""
        # Split audio into segments
        segment_length = int(self.sample_rate * 5)  # 5-second segments
        segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]
        
        if len(segments) < 2:
            return {'dynamics': 'stable', 'variation': 0.0, 'segments': 1}
        
        # Analyze each segment
        segment_moods = []
        for segment in segments:
            if len(segment) > self.sample_rate:  # At least 1 second
                features = self._extract_all_features(segment)
                valence = self._analyze_valence(features)
                arousal = self._analyze_arousal(features)
                mood = self._determine_primary_mood(valence, arousal)
                segment_moods.append(mood)
        
        # Calculate variation
        unique_moods = len(set(segment_moods))
        variation = unique_moods / len(segment_moods) if segment_moods else 0.0
        
        # Determine dynamics type
        if variation > 0.6:
            dynamics = 'highly_variable'
        elif variation > 0.3:
            dynamics = 'moderately_variable'
        else:
            dynamics = 'stable'
        
        return {
            'dynamics': dynamics,
            'variation': variation,
            'segments': len(segment_moods),
            'mood_progression': segment_moods
        }
    
    def _generate_musical_context(self, features: Dict[str, any], tempo: Optional[float], key: Optional[str]) -> Dict[str, any]:
        """Generate musical context for the detected mood"""
        context = {}
        
        # Tempo context
        if tempo or features.get('tempo_features'):
            track_tempo = tempo or features['tempo_features'].get('tempo', 120)
            context['tempo_mood_match'] = self._get_tempo_mood_context(track_tempo)
        
        # Key context
        if key:
            context['key_emotional_context'] = self._get_key_emotional_context(key)
        
        # Instrumentation suggestions based on mood
        context['suggested_instruments'] = self._get_mood_appropriate_instruments(features)
        
        # Production style suggestions
        context['production_style'] = self._get_mood_production_style(features)
        
        return context
    
    def _generate_mood_recommendations(self, mood: str, features: Dict[str, any]) -> Dict[str, List[str]]:
        """Generate recommendations based on detected mood"""
        recommendations = {
            'mixing': [],
            'arrangement': [],
            'creative': []
        }
        
        # Mood-specific mixing recommendations
        if mood in ['excited', 'intense', 'angry']:
            recommendations['mixing'].extend([
                "Use compression to control dynamic peaks",
                "Add brightness with high-frequency enhancement",
                "Consider parallel compression for energy"
            ])
        elif mood in ['peaceful', 'relaxed', 'melancholic']:
            recommendations['mixing'].extend([
                "Preserve dynamics with gentle compression",
                "Use reverb for spatial depth",
                "Focus on low-mid warmth"
            ])
        
        # Arrangement recommendations
        if mood in ['happy', 'excited']:
            recommendations['arrangement'].extend([
                "Layer instruments for richness",
                "Use major scales and bright intervals",
                "Build energy through the track"
            ])
        elif mood in ['sad', 'melancholic']:
            recommendations['arrangement'].extend([
                "Use space and silence effectively",
                "Consider minor keys and darker tones",
                "Focus on emotional instrumental solos"
            ])
        
        # Creative suggestions
        recommendations['creative'].extend([
            f"Enhance the '{mood}' feeling with complementary elements",
            "Consider the listener's emotional journey",
            "Use the mood as a guide for creative decisions"
        ])
        
        return recommendations
    
    def _generate_mood_description(self, mood: str, valence: Dict[str, any], arousal: Dict[str, any]) -> str:
        """Generate a human-readable mood description"""
        descriptions = {
            'excited': "High-energy and positive - perfect for uplifting moments",
            'happy': "Positive and moderately energetic - feel-good music",
            'peaceful': "Positive and calm - soothing and relaxing",
            'intense': "High-energy but emotionally neutral - driving music",
            'neutral': "Balanced emotion and energy - versatile for many contexts",
            'relaxed': "Calm and emotionally neutral - background-friendly",
            'angry': "High-energy and negative - intense and aggressive",
            'sad': "Negative emotion with moderate energy - melancholic",
            'melancholic': "Negative and calm - deeply introspective"
        }
        
        base_description = descriptions.get(mood, "Unique emotional character")
        
        # Add confidence context
        val_conf = valence['confidence']
        ar_conf = arousal['confidence']
        avg_conf = (val_conf + ar_conf) / 2
        
        if avg_conf > 0.8:
            confidence_text = "Very clear emotional signature"
        elif avg_conf > 0.6:
            confidence_text = "Clear emotional characteristics"
        else:
            confidence_text = "Subtle emotional nuances"
        
        return f"{base_description}. {confidence_text}."
    
    # Helper methods
    def _calculate_delta_features(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate delta (change) features"""
        delta = librosa.feature.delta(features)
        return {
            'mean_delta': np.mean(np.abs(delta)),
            'delta_variance': np.var(delta)
        }
    
    def _analyze_overall_timbre(self, mfccs: np.ndarray) -> Dict[str, float]:
        """Analyze overall timbral characteristics"""
        return {
            'brightness': np.mean(mfccs[1:4]),  # Higher MFCCs relate to brightness
            'warmth': -np.mean(mfccs[1]),  # Lower values = warmer
            'roughness': np.std(mfccs[6:9])  # Mid-range MFCC variation
        }
    
    def _calculate_harmonic_stability(self, chroma: np.ndarray) -> float:
        """Calculate harmonic stability from chroma features"""
        # Higher stability = less variation in chroma over time
        return 1.0 - np.mean(np.std(chroma, axis=1))
    
    def _categorize_tempo(self, tempo: float) -> str:
        """Categorize tempo for emotion analysis"""
        if tempo < 80:
            return 'very_slow'
        elif tempo < 100:
            return 'slow'
        elif tempo < 120:
            return 'moderate'
        elif tempo < 140:
            return 'fast'
        else:
            return 'very_fast'
    
    def _calculate_rhythmic_energy(self, audio: np.ndarray, beats: np.ndarray) -> float:
        """Calculate rhythmic energy at beat locations"""
        if len(beats) == 0:
            return 0.0
        
        beat_frames = librosa.frames_to_samples(beats, hop_length=self.hop_length)
        beat_energies = []
        
        for beat_frame in beat_frames:
            if beat_frame < len(audio):
                # Energy in a small window around the beat
                start = max(0, beat_frame - 512)
                end = min(len(audio), beat_frame + 512)
                energy = np.sum(audio[start:end] ** 2)
                beat_energies.append(energy)
        
        return np.mean(beat_energies) if beat_energies else 0.0
    
    def _calculate_pulse_clarity(self, onset_strength: np.ndarray) -> float:
        """Calculate how clear the pulse/beat is"""
        # Use autocorrelation to find periodic patterns
        autocorr = np.correlate(onset_strength, onset_strength, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find the strength of the main periodic component
        if len(autocorr) > 1:
            return np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0.0
        return 0.0
    
    def _calculate_harmonic_complexity(self, harmonic: np.ndarray) -> float:
        """Calculate harmonic complexity"""
        # Use spectral centroid of harmonic component
        if len(harmonic) > 0:
            return np.std(harmonic) / (np.mean(np.abs(harmonic)) + 1e-10)
        return 0.0
    
    def _calculate_consonance_score(self, audio: np.ndarray) -> float:
        """Calculate consonance/dissonance score"""
        # Simplified consonance based on spectral regularity
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Calculate spectral regularity (more regular = more consonant)
        regularity = 1.0 - np.mean(np.std(magnitude, axis=1)) / (np.mean(magnitude) + 1e-10)
        return np.clip(regularity, 0.0, 1.0)
    
    def _get_key_valence(self, key: str) -> float:
        """Get valence score based on musical key"""
        # Simplified major/minor classification
        if 'major' in key.lower() or 'maj' in key.lower():
            return 0.7  # Major keys tend to be more positive
        elif 'minor' in key.lower() or 'min' in key.lower():
            return 0.3  # Minor keys tend to be more negative
        else:
            return 0.5  # Neutral if unclear
    
    def _calculate_timbre_warmth(self, mfcc_features: Dict[str, any]) -> float:
        """Calculate timbral warmth from MFCC features"""
        if mfcc_features.get('overall_timbre'):
            warmth = mfcc_features['overall_timbre'].get('warmth', 0.0)
            # Normalize to 0-1 range
            return np.clip((warmth + 5) / 10, 0.0, 1.0)
        return 0.5
    
    def _calculate_dimension_confidence(self, score: float) -> float:
        """Calculate confidence based on how far from neutral the score is"""
        distance_from_neutral = abs(score - 0.5)
        return min(distance_from_neutral * 2, 1.0)
    
    def _tempo_to_arousal(self, tempo: float) -> float:
        """Convert tempo to arousal score"""
        # Normalize tempo to arousal (very slow = low arousal, very fast = high arousal)
        if tempo < 60:
            return 0.1
        elif tempo < 90:
            return 0.3
        elif tempo < 110:
            return 0.5
        elif tempo < 130:
            return 0.7
        elif tempo < 150:
            return 0.85
        else:
            return 0.95
    
    def _get_tempo_mood_context(self, tempo: float) -> str:
        """Get tempo-mood context description"""
        if tempo < 80:
            return "Slow tempo supports introspective, calm moods"
        elif tempo < 120:
            return "Moderate tempo suitable for balanced emotional expression"
        elif tempo < 140:
            return "Upbeat tempo enhances energetic, positive moods"
        else:
            return "Fast tempo drives high-energy, intense emotions"
    
    def _get_key_emotional_context(self, key: str) -> str:
        """Get key-emotion context description"""
        if 'major' in key.lower():
            return "Major key naturally supports positive, uplifting emotions"
        elif 'minor' in key.lower():
            return "Minor key naturally supports melancholic, introspective emotions"
        else:
            return "Modal or atonal key allows for complex emotional expression"
    
    def _get_mood_appropriate_instruments(self, features: Dict[str, any]) -> List[str]:
        """Suggest instruments based on detected mood characteristics"""
        instruments = []
        
        # Based on spectral characteristics
        if features.get('spectral_centroid'):
            brightness = features['spectral_centroid']['mean']
            if brightness > 2500:
                instruments.extend(['acoustic guitar', 'piano', 'strings', 'bright synths'])
            else:
                instruments.extend(['bass guitar', 'cello', 'warm pads', 'saxophone'])
        
        # Based on harmonic content
        if features.get('harmonic_features'):
            harmonic_ratio = features['harmonic_features'].get('harmonic_ratio', 0.5)
            if harmonic_ratio > 0.7:
                instruments.extend(['piano', 'strings', 'choir', 'sustained pads'])
            else:
                instruments.extend(['drums', 'percussion', 'bass', 'rhythmic elements'])
        
        return list(set(instruments))  # Remove duplicates
    
    def _get_mood_production_style(self, features: Dict[str, any]) -> str:
        """Suggest production style based on mood characteristics"""
        # This is a simplified example
        if features.get('spectral_centroid'):
            brightness = features['spectral_centroid']['mean']
            if brightness > 3000:
                return "Bright, modern production with emphasis on clarity"
            elif brightness > 1500:
                return "Balanced production with warm mid-range focus"
            else:
                return "Warm, intimate production with rich low frequencies"
        
        return "Balanced production approach recommended"
