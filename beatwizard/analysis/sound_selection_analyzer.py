"""
Sound Selection Analyzer - Advanced sound quality and selection feedback
Analyzes individual elements and provides specific sound selection recommendations
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import audio_settings


class SoundSelectionAnalyzer:
    """
    Analyze individual sound elements and provide specific sound selection feedback
    Perfect for helping producers choose better sounds and samples
    """
    
    def __init__(self, sample_rate: int = None, hop_length: int = None):
        """Initialize the sound selection analyzer"""
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        
        # Load sound reference database
        self.sound_references = self._load_sound_references()
        
        logger.debug("SoundSelectionAnalyzer initialized")
    
    def analyze_sound_selection(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Comprehensive sound selection analysis
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary with sound selection analysis and recommendations
        """
        logger.debug("Starting sound selection analysis")
        
        # Separate audio into different frequency-based elements
        elements = self._separate_audio_elements(audio)
        
        # Analyze each element
        analysis_results = {}
        
        # Kick analysis
        analysis_results['kick_analysis'] = self._analyze_kick_quality(elements['kick'])
        
        # Snare analysis
        analysis_results['snare_analysis'] = self._analyze_snare_quality(elements['snare'])
        
        # Hi-hat analysis
        analysis_results['hihat_analysis'] = self._analyze_hihat_quality(elements['hihats'])
        
        # Bass analysis
        analysis_results['bass_analysis'] = self._analyze_bass_quality(elements['bass'])
        
        # Lead/melody analysis
        analysis_results['melody_analysis'] = self._analyze_melody_quality(elements['melody'])
        
        # Pad/harmony analysis
        analysis_results['harmony_analysis'] = self._analyze_harmony_quality(elements['harmony'])
        
        # Overall sound selection assessment
        analysis_results['overall_sound_selection'] = self._assess_overall_sound_selection(analysis_results)
        
        # Generate specific recommendations
        analysis_results['recommendations'] = self._generate_sound_recommendations(analysis_results)
        
        return analysis_results
    
    def _separate_audio_elements(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Separate audio into different instrument/element groups"""
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Use harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Further separate percussive elements by frequency
        # Low frequencies (kick)
        kick_filter = self._create_bandpass_filter(20, 120)
        kick = signal.filtfilt(kick_filter[0], kick_filter[1], percussive)
        
        # Mid frequencies (snare)
        snare_filter = self._create_bandpass_filter(150, 500)
        snare = signal.filtfilt(snare_filter[0], snare_filter[1], percussive)
        
        # High frequencies (hi-hats)
        hihat_filter = self._create_highpass_filter(8000)
        hihats = signal.filtfilt(hihat_filter[0], hihat_filter[1], percussive)
        
        # Bass (low harmonic content)
        bass_filter = self._create_bandpass_filter(40, 300)
        bass = signal.filtfilt(bass_filter[0], bass_filter[1], harmonic)
        
        # Melody (mid-high harmonic content)
        melody_filter = self._create_bandpass_filter(200, 4000)
        melody = signal.filtfilt(melody_filter[0], melody_filter[1], harmonic)
        
        # Harmony/pads (broader harmonic content)
        harmony = harmonic - melody  # Remaining harmonic content
        
        return {
            'kick': kick,
            'snare': snare,
            'hihats': hihats,
            'bass': bass,
            'melody': melody,
            'harmony': harmony,
            'full_percussive': percussive,
            'full_harmonic': harmonic
        }
    
    def _analyze_kick_quality(self, kick_audio: np.ndarray) -> Dict[str, any]:
        """Analyze kick drum quality and characteristics"""
        if len(kick_audio) == 0 or np.max(np.abs(kick_audio)) < 0.01:
            return {
                'quality': 'weak',
                'issues': ['Kick barely audible or missing'],
                'recommendations': ['Add a stronger kick drum'],
                'splice_suggestions': ['Trap kicks', 'House kicks', '808 kicks']
            }
        
        # Calculate kick characteristics
        rms = np.sqrt(np.mean(kick_audio**2))
        peak = np.max(np.abs(kick_audio))
        
        # Frequency analysis
        freqs, psd = signal.welch(kick_audio, fs=self.sample_rate, nperseg=2048)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Analyze frequency distribution
        sub_bass_energy = np.sum(psd[(freqs >= 20) & (freqs <= 60)])
        kick_freq_energy = np.sum(psd[(freqs >= 60) & (freqs <= 120)])
        punch_energy = np.sum(psd[(freqs >= 120) & (freqs <= 300)])
        
        total_energy = sub_bass_energy + kick_freq_energy + punch_energy
        
        if total_energy > 0:
            sub_ratio = sub_bass_energy / total_energy
            kick_ratio = kick_freq_energy / total_energy
            punch_ratio = punch_energy / total_energy
        else:
            sub_ratio = kick_ratio = punch_ratio = 0
        
        # Quality assessment
        quality_score = 0
        issues = []
        recommendations = []
        
        # Check for sufficient low-end
        if kick_ratio < 0.3:
            issues.append("Kick lacks fundamental frequency content")
            recommendations.append("Choose a kick with stronger 60-120Hz content")
            quality_score -= 2
        
        # Check for punch
        if punch_ratio < 0.2:
            issues.append("Kick lacks punch and attack")
            recommendations.append("Add a kick with more 120-300Hz content for punch")
            quality_score -= 1
        
        # Check for excessive sub-bass
        if sub_ratio > 0.6:
            issues.append("Kick has too much sub-bass, may sound muddy")
            recommendations.append("High-pass the kick around 30-40Hz")
            quality_score -= 1
        
        # Check overall level
        if rms < 0.1:
            issues.append("Kick is too quiet in the mix")
            recommendations.append("Increase kick level or use compression")
            quality_score -= 1
        
        # Determine quality
        if quality_score >= 0:
            quality = 'good'
        elif quality_score >= -2:
            quality = 'fair'
        else:
            quality = 'poor'
        
        # Generate specific sample suggestions
        splice_suggestions = self._get_kick_sample_suggestions(dominant_freq, sub_ratio, kick_ratio, punch_ratio)
        
        return {
            'quality': quality,
            'dominant_frequency': float(dominant_freq),
            'frequency_distribution': {
                'sub_bass_ratio': float(sub_ratio),
                'fundamental_ratio': float(kick_ratio),
                'punch_ratio': float(punch_ratio)
            },
            'level_analysis': {
                'rms': float(rms),
                'peak': float(peak),
                'dynamic_range': float(20 * np.log10(peak / (rms + 1e-10)))
            },
            'issues': issues,
            'recommendations': recommendations,
            'splice_suggestions': splice_suggestions
        }
    
    def _analyze_snare_quality(self, snare_audio: np.ndarray) -> Dict[str, any]:
        """Analyze snare quality and characteristics"""
        if len(snare_audio) == 0 or np.max(np.abs(snare_audio)) < 0.005:
            return {
                'quality': 'weak',
                'issues': ['Snare barely audible or missing'],
                'recommendations': ['Add a more prominent snare'],
                'splice_suggestions': ['Trap snares', 'Acoustic snares', 'Claps']
            }
        
        rms = np.sqrt(np.mean(snare_audio**2))
        
        # Frequency analysis
        freqs, psd = signal.welch(snare_audio, fs=self.sample_rate, nperseg=2048)
        
        # Snare frequency regions
        body_energy = np.sum(psd[(freqs >= 150) & (freqs <= 300)])
        crack_energy = np.sum(psd[(freqs >= 1000) & (freqs <= 5000)])
        sizzle_energy = np.sum(psd[(freqs >= 5000) & (freqs <= 12000)])
        
        total_energy = body_energy + crack_energy + sizzle_energy
        
        if total_energy > 0:
            body_ratio = body_energy / total_energy
            crack_ratio = crack_energy / total_energy
            sizzle_ratio = sizzle_energy / total_energy
        else:
            body_ratio = crack_ratio = sizzle_ratio = 0
        
        quality_score = 0
        issues = []
        recommendations = []
        
        # Check for body
        if body_ratio < 0.2:
            issues.append("Snare lacks body and thickness")
            recommendations.append("Choose a snare with more 150-300Hz content")
            quality_score -= 1
        
        # Check for crack/snap
        if crack_ratio < 0.3:
            issues.append("Snare lacks crack and presence")
            recommendations.append("Add a snare with more midrange punch (1-5kHz)")
            quality_score -= 2
        
        # Check for high-end sizzle
        if sizzle_ratio < 0.1:
            issues.append("Snare sounds dull, lacks high-end sizzle")
            recommendations.append("Layer with a snare that has more high-frequency content")
            quality_score -= 1
        
        # Check level
        if rms < 0.05:
            issues.append("Snare is too quiet in the mix")
            recommendations.append("Increase snare level or add compression")
            quality_score -= 1
        
        # Determine quality
        if quality_score >= 0:
            quality = 'good'
        elif quality_score >= -2:
            quality = 'fair'
        else:
            quality = 'poor'
        
        splice_suggestions = self._get_snare_sample_suggestions(body_ratio, crack_ratio, sizzle_ratio)
        
        return {
            'quality': quality,
            'frequency_distribution': {
                'body_ratio': float(body_ratio),
                'crack_ratio': float(crack_ratio),
                'sizzle_ratio': float(sizzle_ratio)
            },
            'level_analysis': {'rms': float(rms)},
            'issues': issues,
            'recommendations': recommendations,
            'splice_suggestions': splice_suggestions
        }
    
    def _analyze_bass_quality(self, bass_audio: np.ndarray) -> Dict[str, any]:
        """Analyze bass quality and characteristics"""
        if len(bass_audio) == 0 or np.max(np.abs(bass_audio)) < 0.01:
            return {
                'quality': 'weak',
                'issues': ['Bass barely audible or missing'],
                'recommendations': ['Add a stronger bass line'],
                'splice_suggestions': ['808 bass', 'Sub bass', 'Reese bass']
            }
        
        rms = np.sqrt(np.mean(bass_audio**2))
        
        # Frequency analysis
        freqs, psd = signal.welch(bass_audio, fs=self.sample_rate, nperseg=2048)
        
        # Bass frequency regions
        sub_energy = np.sum(psd[(freqs >= 20) & (freqs <= 60)])
        fundamental_energy = np.sum(psd[(freqs >= 60) & (freqs <= 150)])
        harmonics_energy = np.sum(psd[(freqs >= 150) & (freqs <= 400)])
        
        total_energy = sub_energy + fundamental_energy + harmonics_energy
        
        if total_energy > 0:
            sub_ratio = sub_energy / total_energy
            fundamental_ratio = fundamental_energy / total_energy
            harmonics_ratio = harmonics_energy / total_energy
        else:
            sub_ratio = fundamental_ratio = harmonics_ratio = 0
        
        quality_score = 0
        issues = []
        recommendations = []
        
        # Check for fundamental
        if fundamental_ratio < 0.4:
            issues.append("Bass lacks fundamental frequency content")
            recommendations.append("Choose a bass with stronger 60-150Hz presence")
            quality_score -= 2
        
        # Check for harmonics (musical content)
        if harmonics_ratio < 0.2:
            issues.append("Bass sounds flat, lacks harmonic content")
            recommendations.append("Add a bass with more harmonic richness or overdrive")
            quality_score -= 1
        
        # Check for excessive sub
        if sub_ratio > 0.7:
            issues.append("Bass has too much sub content, may sound boomy")
            recommendations.append("High-pass the bass around 30-40Hz")
            quality_score -= 1
        
        # Check level
        if rms < 0.08:
            issues.append("Bass is too quiet in the mix")
            recommendations.append("Increase bass level or add compression")
            quality_score -= 1
        
        # Determine quality
        if quality_score >= 0:
            quality = 'good'
        elif quality_score >= -2:
            quality = 'fair'
        else:
            quality = 'poor'
        
        splice_suggestions = self._get_bass_sample_suggestions(sub_ratio, fundamental_ratio, harmonics_ratio)
        
        return {
            'quality': quality,
            'frequency_distribution': {
                'sub_ratio': float(sub_ratio),
                'fundamental_ratio': float(fundamental_ratio),
                'harmonics_ratio': float(harmonics_ratio)
            },
            'level_analysis': {'rms': float(rms)},
            'issues': issues,
            'recommendations': recommendations,
            'splice_suggestions': splice_suggestions
        }
    
    def _analyze_melody_quality(self, melody_audio: np.ndarray) -> Dict[str, any]:
        """Analyze melody and lead sound quality"""
        if len(melody_audio) == 0 or np.max(np.abs(melody_audio)) < 0.005:
            return {
                'quality': 'weak',
                'issues': ['Melody/lead barely audible or missing'],
                'recommendations': ['Add a stronger melodic element'],
                'splice_suggestions': ['Synth leads', 'Plucks', 'Arps']
            }
        
        # Extract pitch content
        pitches, magnitudes = librosa.piptrack(y=melody_audio, sr=self.sample_rate)
        
        # Calculate note activity
        note_activity = np.sum(magnitudes > 0.1, axis=1)
        active_frames = np.sum(note_activity > 0)
        total_frames = len(note_activity)
        
        activity_ratio = active_frames / total_frames if total_frames > 0 else 0
        
        # Frequency analysis
        freqs, psd = signal.welch(melody_audio, fs=self.sample_rate, nperseg=2048)
        
        # Find dominant frequency regions
        low_mid_energy = np.sum(psd[(freqs >= 200) & (freqs <= 800)])
        mid_energy = np.sum(psd[(freqs >= 800) & (freqs <= 2000)])
        high_mid_energy = np.sum(psd[(freqs >= 2000) & (freqs <= 6000)])
        
        total_energy = low_mid_energy + mid_energy + high_mid_energy
        
        if total_energy > 0:
            low_mid_ratio = low_mid_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_mid_ratio = high_mid_energy / total_energy
        else:
            low_mid_ratio = mid_ratio = high_mid_ratio = 0
        
        quality_score = 0
        issues = []
        recommendations = []
        
        # Check for presence
        if activity_ratio < 0.3:
            issues.append("Melody lacks presence and activity")
            recommendations.append("Add more melodic content or increase melody level")
            quality_score -= 2
        
        # Check frequency distribution
        if mid_ratio < 0.3:
            issues.append("Melody lacks midrange presence")
            recommendations.append("Choose a lead sound with more 800Hz-2kHz content")
            quality_score -= 1
        
        if high_mid_ratio < 0.2:
            issues.append("Melody sounds dull, lacks brilliance")
            recommendations.append("Add brightness with EQ or choose a brighter lead sound")
            quality_score -= 1
        
        # Determine quality
        if quality_score >= 0:
            quality = 'good'
        elif quality_score >= -2:
            quality = 'fair'
        else:
            quality = 'poor'
        
        splice_suggestions = self._get_melody_sample_suggestions(low_mid_ratio, mid_ratio, high_mid_ratio)
        
        return {
            'quality': quality,
            'activity_ratio': float(activity_ratio),
            'frequency_distribution': {
                'low_mid_ratio': float(low_mid_ratio),
                'mid_ratio': float(mid_ratio),
                'high_mid_ratio': float(high_mid_ratio)
            },
            'issues': issues,
            'recommendations': recommendations,
            'splice_suggestions': splice_suggestions
        }
    
    def _analyze_harmony_quality(self, harmony_audio: np.ndarray) -> Dict[str, any]:
        """Analyze harmony and pad quality"""
        # Similar structure to melody analysis but focused on harmonic content
        rms = np.sqrt(np.mean(harmony_audio**2))
        
        issues = []
        recommendations = []
        splice_suggestions = ['Pads', 'Strings', 'Chords', 'Ambient textures']
        
        if rms < 0.02:
            issues.append("Harmony/pads are too quiet or missing")
            recommendations.append("Add harmonic content with pads or chords")
        
        quality = 'fair' if not issues else 'weak'
        
        return {
            'quality': quality,
            'level_analysis': {'rms': float(rms)},
            'issues': issues,
            'recommendations': recommendations,
            'splice_suggestions': splice_suggestions
        }
    
    def _analyze_hihat_quality(self, hihat_audio: np.ndarray) -> Dict[str, any]:
        """Analyze hi-hat quality"""
        rms = np.sqrt(np.mean(hihat_audio**2))
        
        issues = []
        recommendations = []
        splice_suggestions = ['Open hats', 'Closed hats', 'Hi-hat loops', 'Shakers']
        
        if rms < 0.01:
            issues.append("Hi-hats are too quiet or missing")
            recommendations.append("Add hi-hat elements for rhythm and energy")
        
        quality = 'fair' if not issues else 'weak'
        
        return {
            'quality': quality,
            'level_analysis': {'rms': float(rms)},
            'issues': issues,
            'recommendations': recommendations,
            'splice_suggestions': splice_suggestions
        }
    
    def _get_kick_sample_suggestions(self, dominant_freq: float, sub_ratio: float, kick_ratio: float, punch_ratio: float) -> List[str]:
        """Get specific kick sample suggestions based on analysis"""
        suggestions = []
        
        if sub_ratio > 0.6:
            suggestions.extend(['808 kicks', 'Sub-heavy kicks', 'Trap 808s'])
        elif kick_ratio > 0.5:
            suggestions.extend(['House kicks', 'Techno kicks', 'Four-on-floor kicks'])
        elif punch_ratio > 0.4:
            suggestions.extend(['Punchy kicks', 'Rock kicks', 'Compressed kicks'])
        else:
            suggestions.extend(['Balanced kicks', 'All-purpose kicks', 'Studio kicks'])
        
        # Add genre-specific suggestions
        if dominant_freq < 70:
            suggestions.append('Deep house kicks')
        elif dominant_freq > 100:
            suggestions.append('Breakbeat kicks')
        
        return suggestions[:5]  # Limit to top 5
    
    def _get_snare_sample_suggestions(self, body_ratio: float, crack_ratio: float, sizzle_ratio: float) -> List[str]:
        """Get specific snare sample suggestions"""
        suggestions = []
        
        if crack_ratio > 0.5:
            suggestions.extend(['Snappy snares', 'Trap snares', 'Crispy snares'])
        elif body_ratio > 0.4:
            suggestions.extend(['Fat snares', 'Thick snares', 'Punchy snares'])
        elif sizzle_ratio > 0.3:
            suggestions.extend(['Bright snares', 'Sizzling snares', 'Hi-freq snares'])
        else:
            suggestions.extend(['Balanced snares', 'Classic snares', 'Acoustic snares'])
        
        # Add alternatives
        suggestions.extend(['Claps', 'Rim shots', 'Layered snares'])
        
        return suggestions[:5]
    
    def _get_bass_sample_suggestions(self, sub_ratio: float, fundamental_ratio: float, harmonics_ratio: float) -> List[str]:
        """Get specific bass sample suggestions"""
        suggestions = []
        
        if sub_ratio > 0.6:
            suggestions.extend(['808 bass', 'Sub bass', 'Deep bass'])
        elif fundamental_ratio > 0.5:
            suggestions.extend(['Reese bass', 'Saw bass', 'Wobble bass'])
        elif harmonics_ratio > 0.4:
            suggestions.extend(['Distorted bass', 'Acid bass', 'Growl bass'])
        else:
            suggestions.extend(['Balanced bass', 'Clean bass', 'Synth bass'])
        
        return suggestions[:5]
    
    def _get_melody_sample_suggestions(self, low_mid_ratio: float, mid_ratio: float, high_mid_ratio: float) -> List[str]:
        """Get specific melody sample suggestions"""
        suggestions = []
        
        if high_mid_ratio > 0.4:
            suggestions.extend(['Bright leads', 'Supersaw leads', 'Bell leads'])
        elif mid_ratio > 0.5:
            suggestions.extend(['Pluck leads', 'Classic leads', 'Balanced leads'])
        elif low_mid_ratio > 0.4:
            suggestions.extend(['Warm leads', 'Deep leads', 'Analog leads'])
        else:
            suggestions.extend(['Arps', 'Sequences', 'Melodic loops'])
        
        return suggestions[:5]
    
    def _assess_overall_sound_selection(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Assess overall sound selection quality"""
        element_qualities = []
        
        for element in ['kick_analysis', 'snare_analysis', 'bass_analysis', 'melody_analysis']:
            if element in analysis_results:
                quality = analysis_results[element].get('quality', 'weak')
                if quality == 'good':
                    element_qualities.append(3)
                elif quality == 'fair':
                    element_qualities.append(2)
                else:
                    element_qualities.append(1)
        
        if element_qualities:
            avg_quality = sum(element_qualities) / len(element_qualities)
            
            if avg_quality >= 2.5:
                overall_quality = 'good'
                overall_feedback = "Sound selection is generally good with minor improvements needed"
            elif avg_quality >= 2.0:
                overall_quality = 'fair'
                overall_feedback = "Sound selection needs some work to reach professional standards"
            else:
                overall_quality = 'poor'
                overall_feedback = "Sound selection needs significant improvement - focus on higher quality samples"
        else:
            overall_quality = 'unknown'
            overall_feedback = "Unable to assess sound selection"
        
        return {
            'overall_quality': overall_quality,
            'quality_score': avg_quality if element_qualities else 0,
            'feedback': overall_feedback
        }
    
    def _generate_sound_recommendations(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Generate specific sound recommendations"""
        priority_recommendations = []
        splice_pack_suggestions = []
        beginner_tips = []
        
        # Analyze each element and prioritize fixes
        for element_name, element_data in analysis_results.items():
            if element_name.endswith('_analysis') and 'quality' in element_data:
                element_type = element_name.replace('_analysis', '')
                quality = element_data['quality']
                issues = element_data.get('issues', [])
                recommendations = element_data.get('recommendations', [])
                splice_suggestions = element_data.get('splice_suggestions', [])
                
                if quality in ['poor', 'weak']:
                    priority_recommendations.append({
                        'element': element_type,
                        'priority': 'high',
                        'issues': issues,
                        'recommendations': recommendations,
                        'beginner_tip': self._get_beginner_tip(element_type, quality)
                    })
                    
                    splice_pack_suggestions.extend([
                        f"{element_type.title()}: {suggestion}" for suggestion in splice_suggestions[:3]
                    ])
        
        # Generate beginner-friendly tips
        beginner_tips = [
            "Start with the kick - it's the foundation of your track",
            "Layer your snare with a clap for more impact",
            "Use reference tracks to compare your sound selection",
            "Don't be afraid to replace sounds that aren't working",
            "Quality samples save time and sound more professional"
        ]
        
        return {
            'priority_recommendations': priority_recommendations,
            'splice_pack_suggestions': splice_pack_suggestions[:10],
            'beginner_tips': beginner_tips,
            'next_steps': self._generate_next_steps(priority_recommendations)
        }
    
    def _get_beginner_tip(self, element_type: str, quality: str) -> str:
        """Get beginner-friendly tips for specific elements"""
        tips = {
            'kick': "Your kick should be felt, not just heard. Try layering a punchy kick with an 808 for dance music.",
            'snare': "A good snare cuts through the mix. Try layering different snare samples for thickness and snap.",
            'bass': "Bass should complement the kick, not fight with it. High-pass around 30Hz to clean up the low end.",
            'melody': "Melody should sit well in the mix. Use EQ to carve out space and avoid frequency conflicts.",
            'hihat': "Hi-hats add energy and groove. Layer different hi-hat samples to create interesting rhythms.",
            'harmony': "Pads and chords create atmosphere. Use them to fill space and add emotional depth."
        }
        
        return tips.get(element_type, "Focus on choosing high-quality samples that fit your genre.")
    
    def _generate_next_steps(self, priority_recommendations: List[Dict]) -> List[str]:
        """Generate actionable next steps"""
        next_steps = []
        
        if len(priority_recommendations) > 0:
            worst_element = priority_recommendations[0]['element']
            next_steps.append(f"1. Replace your {worst_element} - it's holding back your track the most")
        
        if len(priority_recommendations) > 1:
            second_element = priority_recommendations[1]['element']
            next_steps.append(f"2. Work on your {second_element} after fixing the {worst_element}")
        
        next_steps.extend([
            "3. A/B test your new samples against the originals",
            "4. Use reference tracks to guide your sound selection",
            "5. Focus on one element at a time for best results"
        ])
        
        return next_steps[:5]
    
    def _create_bandpass_filter(self, low_freq: float, high_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create bandpass filter coefficients"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return b, a
    
    def _create_highpass_filter(self, freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create highpass filter coefficients"""
        nyquist = self.sample_rate / 2
        normalized_freq = freq / nyquist
        b, a = signal.butter(4, normalized_freq, btype='high')
        return b, a
    
    def _load_sound_references(self) -> Dict[str, any]:
        """Load sound reference database (placeholder for future implementation)"""
        return {
            'kick_references': {},
            'snare_references': {},
            'bass_references': {},
            'melody_references': {}
        }