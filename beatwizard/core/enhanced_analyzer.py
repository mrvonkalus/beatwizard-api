"""
Enhanced Audio Analyzer - Main analysis orchestrator
Combines all individual analyzers for comprehensive professional audio analysis
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import time
import json

from .audio_processor import AudioProcessor
from ..analysis.tempo_detector import TempoDetector
from ..analysis.key_detector import KeyDetector
from ..analysis.frequency_analyzer import FrequencyAnalyzer
from ..analysis.loudness_analyzer import LoudnessAnalyzer
from ..analysis.stereo_analyzer import StereoAnalyzer
from ..analysis.sound_selection_analyzer import SoundSelectionAnalyzer
from ..analysis.rhythmic_analyzer import RhythmicAnalyzer
from ..analysis.harmonic_analyzer import HarmonicAnalyzer
from ..ai_integration.intelligent_feedback import IntelligentFeedbackGenerator
from config.settings import audio_settings, performance_settings, project_settings


class EnhancedAudioAnalyzer:
    """
    Professional comprehensive audio analysis system
    Orchestrates multiple specialized analyzers for complete track analysis
    """
    
    def __init__(self, 
                 sample_rate: int = None, 
                 hop_length: int = None,
                 enable_caching: bool = True,
                 enable_multiprocessing: bool = None):
        """
        Initialize the enhanced audio analyzer
        
        Args:
            sample_rate: Target sample rate for analysis
            hop_length: Hop length for time-based analysis
            enable_caching: Whether to enable result caching
            enable_multiprocessing: Whether to use multiprocessing (defaults to config)
        """
        self.sample_rate = sample_rate or audio_settings.DEFAULT_SAMPLE_RATE
        self.hop_length = hop_length or audio_settings.DEFAULT_HOP_LENGTH
        self.enable_caching = enable_caching and performance_settings.ENABLE_DISK_CACHE
        self.enable_multiprocessing = enable_multiprocessing if enable_multiprocessing is not None else performance_settings.USE_MULTIPROCESSING
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(self.sample_rate, self.hop_length)
        
        # Initialize specialized analyzers
        self.tempo_detector = TempoDetector(self.sample_rate, self.hop_length)
        self.key_detector = KeyDetector(self.sample_rate, self.hop_length)
        self.frequency_analyzer = FrequencyAnalyzer(self.sample_rate, self.hop_length)
        self.loudness_analyzer = LoudnessAnalyzer(self.sample_rate)
        self.stereo_analyzer = StereoAnalyzer(self.sample_rate, self.hop_length)
        
        # Initialize advanced analyzers
        self.sound_selection_analyzer = SoundSelectionAnalyzer(self.sample_rate, self.hop_length)
        self.rhythmic_analyzer = RhythmicAnalyzer(self.sample_rate, self.hop_length)
        self.harmonic_analyzer = HarmonicAnalyzer(self.sample_rate, self.hop_length)
        
        # Initialize AI feedback generator
        self.feedback_generator = IntelligentFeedbackGenerator()
        
        # Analysis metadata
        self.analysis_version = "1.0.0"
        self.last_analysis_time = None
        
        logger.info(f"EnhancedAudioAnalyzer initialized - SR: {self.sample_rate}, Caching: {self.enable_caching}")
    
    def analyze_track(self, 
                     file_path: Union[str, Path], 
                     analysis_types: Optional[List[str]] = None,
                     normalize_audio: bool = True,
                     trim_silence: bool = False,
                     force_reanalysis: bool = False) -> Dict[str, any]:
        """
        Comprehensive track analysis
        
        Args:
            file_path: Path to audio file
            analysis_types: List of analysis types to perform (None = all)
            normalize_audio: Whether to normalize audio before analysis
            trim_silence: Whether to trim silence from audio
            force_reanalysis: Force reanalysis even if cached result exists
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        logger.info(f"Starting enhanced analysis of: {file_path.name}")
        
        # Check for cached results
        if self.enable_caching and not force_reanalysis:
            cached_result = self._load_cached_analysis(file_path)
            if cached_result:
                logger.info(f"Using cached analysis for: {file_path.name}")
                return cached_result
        
        try:
            # Load and preprocess audio
            audio_data, metadata = self._load_and_preprocess_audio(
                file_path, normalize_audio, trim_silence
            )
            
            # Determine analysis types to perform
            if analysis_types is None:
                analysis_types = ['tempo', 'key', 'frequency', 'loudness', 'stereo', 'sound_selection', 'rhythm', 'harmony']
            
            # Perform comprehensive analysis
            analysis_results = self._perform_analysis(audio_data, metadata, analysis_types)
            
            # Add metadata and timing
            analysis_results['analysis_metadata'] = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'analysis_version': self.analysis_version,
                'analysis_timestamp': time.time(),
                'analysis_duration': time.time() - start_time,
                'analysis_types': analysis_types,
                'audio_metadata': metadata,
                'preprocessing': {
                    'normalized': normalize_audio,
                    'silence_trimmed': trim_silence
                }
            }
            
            # Generate overall assessment
            analysis_results['overall_assessment'] = self._generate_overall_assessment(analysis_results)
            
            # Generate professional insights
            analysis_results['professional_insights'] = self._generate_professional_insights(analysis_results)
            
            # Cache results if enabled
            if self.enable_caching:
                self._cache_analysis_results(file_path, analysis_results)
            
            self.last_analysis_time = time.time() - start_time
            
            logger.info(f"Enhanced analysis completed in {self.last_analysis_time:.2f}s: {file_path.name}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed for {file_path}: {str(e)}")
            raise
    
    def _load_and_preprocess_audio(self, 
                                  file_path: Path, 
                                  normalize: bool, 
                                  trim_silence: bool) -> Tuple[np.ndarray, Dict[str, any]]:
        """Load and preprocess audio for analysis"""
        # Load audio
        audio, metadata = self.audio_processor.load_audio(str(file_path), normalize=normalize)
        
        # Validate audio quality
        quality_analysis = self.audio_processor.validate_audio_quality(audio, metadata)
        metadata['quality_analysis'] = quality_analysis
        
        # Trim silence if requested
        if trim_silence:
            audio, trim_info = self.audio_processor.trim_silence(audio)
            metadata['silence_trimmed'] = True
            metadata['trim_info'] = trim_info
            logger.debug(f"Trimmed silence: {trim_info}")
        else:
            metadata['silence_trimmed'] = False
        
        # Apply high-pass filter for cleaner analysis
        if len(audio.shape) == 2:
            audio[0] = self.audio_processor.apply_high_pass_filter(audio[0])
            audio[1] = self.audio_processor.apply_high_pass_filter(audio[1])
        else:
            audio = self.audio_processor.apply_high_pass_filter(audio)
        
        logger.debug(f"Audio preprocessing completed for: {file_path.name}")
        
        return audio, metadata
    
    def _perform_analysis(self, 
                         audio: np.ndarray, 
                         metadata: Dict[str, any], 
                         analysis_types: List[str]) -> Dict[str, any]:
        """Perform the requested analysis types"""
        results = {}
        
        # Get mono version for mono-specific analyses
        mono_audio = self.audio_processor.get_mono(audio)
        
        # Tempo Analysis
        if 'tempo' in analysis_types:
            logger.debug("Performing tempo analysis")
            try:
                results['tempo_analysis'] = self.tempo_detector.detect_tempo(mono_audio)
            except Exception as e:
                logger.error(f"Tempo analysis failed: {e}")
                results['tempo_analysis'] = {'error': str(e)}
        
        # Key Analysis
        if 'key' in analysis_types:
            logger.debug("Performing key analysis")
            try:
                results['key_analysis'] = self.key_detector.detect_key(mono_audio)
            except Exception as e:
                logger.error(f"Key analysis failed: {e}")
                results['key_analysis'] = {'error': str(e)}
        
        # Frequency Analysis
        if 'frequency' in analysis_types:
            logger.debug("Performing frequency analysis")
            try:
                results['frequency_analysis'] = self.frequency_analyzer.analyze_frequency_spectrum(mono_audio)
            except Exception as e:
                logger.error(f"Frequency analysis failed: {e}")
                results['frequency_analysis'] = {'error': str(e)}
        
        # Loudness Analysis
        if 'loudness' in analysis_types:
            logger.debug("Performing loudness analysis")
            try:
                results['loudness_analysis'] = self.loudness_analyzer.analyze_loudness(audio)
            except Exception as e:
                logger.error(f"Loudness analysis failed: {e}")
                results['loudness_analysis'] = {'error': str(e)}
        
        # Stereo Analysis (only for stereo audio)
        if 'stereo' in analysis_types:
            logger.debug("Performing stereo analysis")
            try:
                results['stereo_analysis'] = self.stereo_analyzer.analyze_stereo_image(audio)
            except Exception as e:
                logger.error(f"Stereo analysis failed: {e}")
                results['stereo_analysis'] = {'error': str(e)}
        
        # Sound Selection Analysis
        if 'sound_selection' in analysis_types:
            logger.debug("Performing sound selection analysis")
            try:
                results['sound_selection_analysis'] = self.sound_selection_analyzer.analyze_sound_selection(mono_audio)
            except Exception as e:
                logger.error(f"Sound selection analysis failed: {e}")
                results['sound_selection_analysis'] = {'error': str(e)}
        
        # Rhythm Analysis
        if 'rhythm' in analysis_types:
            logger.debug("Performing rhythmic analysis")
            try:
                tempo = results.get('tempo_analysis', {}).get('primary_tempo')
                results['rhythm_analysis'] = self.rhythmic_analyzer.analyze_rhythm(mono_audio, tempo)
            except Exception as e:
                logger.error(f"Rhythmic analysis failed: {e}")
                results['rhythm_analysis'] = {'error': str(e)}
        
        # Harmony Analysis
        if 'harmony' in analysis_types:
            logger.debug("Performing harmonic analysis")
            try:
                key = results.get('key_analysis', {}).get('primary_key')
                results['harmony_analysis'] = self.harmonic_analyzer.analyze_harmony(mono_audio, key)
            except Exception as e:
                logger.error(f"Harmonic analysis failed: {e}")
                results['harmony_analysis'] = {'error': str(e)}
        
        return results
    
    def generate_intelligent_feedback(self, 
                                    analysis_results: Dict[str, any],
                                    skill_level: str = 'beginner',
                                    genre: Optional[str] = None,
                                    goals: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Generate intelligent, contextual feedback for producers
        
        Args:
            analysis_results: Complete analysis results
            skill_level: Producer skill level ('beginner', 'intermediate', 'advanced')
            genre: Target genre
            goals: Production goals (e.g., ['streaming', 'club', 'radio'])
            
        Returns:
            Dictionary with comprehensive feedback
        """
        logger.debug(f"Generating intelligent feedback for {skill_level} producer")
        
        try:
            feedback = self.feedback_generator.generate_comprehensive_feedback(
                analysis_results, skill_level, genre, goals
            )
            return feedback
        except Exception as e:
            logger.error(f"Intelligent feedback generation failed: {e}")
            return {
                'error': str(e),
                'fallback_message': 'Basic analysis completed - see individual analysis sections for details'
            }
    
    def _generate_overall_assessment(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Generate overall track assessment based on all analyses"""
        try:
            assessment = {
                'overall_quality': 'unknown',
                'technical_quality': 'unknown',
                'mix_quality': 'unknown',
                'mastering_readiness': False,
                'commercial_readiness': False,
                'issue_count': 0,
                'strengths': [],
                'weaknesses': [],
                'priority_issues': []
            }
            
            # Collect individual quality scores
            quality_scores = []
            issues = []
            strengths = []
            
            # Tempo Analysis Assessment
            tempo_analysis = analysis_results.get('tempo_analysis', {})
            if 'error' not in tempo_analysis:
                tempo_confidence = tempo_analysis.get('confidence', 0.0)
                if tempo_confidence > 0.8:
                    strengths.append("Excellent tempo detection confidence")
                    quality_scores.append(5)
                elif tempo_confidence > 0.6:
                    quality_scores.append(4)
                elif tempo_confidence > 0.4:
                    quality_scores.append(3)
                    issues.append("Tempo detection uncertainty")
                else:
                    quality_scores.append(2)
                    issues.append("Poor tempo detection - may need manual verification")
            
            # Key Analysis Assessment
            key_analysis = analysis_results.get('key_analysis', {})
            if 'error' not in key_analysis:
                key_confidence = key_analysis.get('confidence', 0.0)
                if key_confidence > 0.7:
                    strengths.append("Clear musical key detected")
                    quality_scores.append(5)
                elif key_confidence > 0.5:
                    quality_scores.append(4)
                elif key_confidence > 0.3:
                    quality_scores.append(3)
                    issues.append("Key detection uncertainty")
                else:
                    quality_scores.append(2)
                    issues.append("Unclear musical key")
            
            # Frequency Analysis Assessment
            frequency_analysis = analysis_results.get('frequency_analysis', {})
            if 'error' not in frequency_analysis:
                freq_assessment = frequency_analysis.get('overall_assessment', {})
                freq_quality = freq_assessment.get('quality_rating', 'unknown')
                
                if freq_quality == 'excellent':
                    strengths.append("Excellent frequency balance")
                    quality_scores.append(5)
                elif freq_quality == 'good':
                    strengths.append("Good frequency balance")
                    quality_scores.append(4)
                elif freq_quality == 'fair':
                    quality_scores.append(3)
                    issues.append("Frequency balance could be improved")
                else:
                    quality_scores.append(2)
                    issues.append("Frequency balance needs significant work")
                
                # Check for ready for mastering
                if freq_assessment.get('ready_for_mastering', False):
                    strengths.append("Ready for mastering (frequency perspective)")
            
            # Loudness Analysis Assessment
            loudness_analysis = analysis_results.get('loudness_analysis', {})
            if 'error' not in loudness_analysis:
                loudness_assessment = loudness_analysis.get('overall_assessment', {})
                loudness_quality = loudness_assessment.get('overall_quality', 'unknown')
                
                if loudness_quality == 'excellent':
                    strengths.append("Professional loudness standards")
                    quality_scores.append(5)
                elif loudness_quality == 'good':
                    strengths.append("Good loudness characteristics")
                    quality_scores.append(4)
                elif loudness_quality == 'acceptable':
                    quality_scores.append(3)
                else:
                    quality_scores.append(2)
                    issues.append("Loudness standards need attention")
                
                # Check platform compliance
                compliance_count = loudness_assessment.get('compliance_count', 0)
                if compliance_count >= 3:
                    strengths.append("Compliant with multiple streaming platforms")
                elif compliance_count >= 1:
                    strengths.append("Compliant with at least one streaming platform")
                else:
                    issues.append("Not compliant with streaming platform standards")
                
                # Check for distribution readiness
                if loudness_assessment.get('ready_for_distribution', False):
                    strengths.append("Ready for distribution (loudness perspective)")
            
            # Stereo Analysis Assessment
            stereo_analysis = analysis_results.get('stereo_analysis', {})
            if 'error' not in stereo_analysis:
                stereo_assessment = stereo_analysis.get('overall_assessment', {})
                stereo_quality = stereo_assessment.get('overall_quality', 'unknown')
                
                if stereo_quality == 'excellent':
                    strengths.append("Excellent stereo imaging")
                    quality_scores.append(5)
                elif stereo_quality == 'good':
                    strengths.append("Good stereo imaging")
                    quality_scores.append(4)
                elif stereo_quality == 'acceptable':
                    quality_scores.append(3)
                elif stereo_quality == 'mono':
                    # Mono is not necessarily bad
                    quality_scores.append(3)
                    strengths.append("Mono audio (intentional choice)")
                else:
                    quality_scores.append(2)
                    issues.append("Stereo imaging needs attention")
                
                # Check for mastering readiness
                if stereo_assessment.get('ready_for_mastering', False):
                    strengths.append("Ready for mastering (stereo perspective)")
            
            # Calculate overall scores
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                
                if avg_quality >= 4.5:
                    assessment['overall_quality'] = 'excellent'
                    assessment['technical_quality'] = 'professional'
                    assessment['mix_quality'] = 'excellent'
                elif avg_quality >= 3.5:
                    assessment['overall_quality'] = 'good'
                    assessment['technical_quality'] = 'good'
                    assessment['mix_quality'] = 'good'
                elif avg_quality >= 2.5:
                    assessment['overall_quality'] = 'acceptable'
                    assessment['technical_quality'] = 'acceptable'
                    assessment['mix_quality'] = 'needs_improvement'
                else:
                    assessment['overall_quality'] = 'poor'
                    assessment['technical_quality'] = 'poor'
                    assessment['mix_quality'] = 'needs_significant_work'
            
            # Determine readiness
            readiness_indicators = []
            if frequency_analysis.get('overall_assessment', {}).get('ready_for_mastering', False):
                readiness_indicators.append('frequency')
            if loudness_analysis.get('overall_assessment', {}).get('ready_for_distribution', False):
                readiness_indicators.append('loudness')
            if stereo_analysis.get('overall_assessment', {}).get('ready_for_mastering', False):
                readiness_indicators.append('stereo')
            
            assessment['mastering_readiness'] = len(readiness_indicators) >= 2
            assessment['commercial_readiness'] = len(readiness_indicators) >= 2 and len(issues) <= 2
            
            # Finalize assessment
            assessment['issue_count'] = len(issues)
            assessment['strengths'] = strengths[:10]  # Limit for readability
            assessment['weaknesses'] = issues[:10]
            assessment['priority_issues'] = [issue for issue in issues if any(keyword in issue.lower() for keyword in ['significant', 'poor', 'not compliant'])]
            
            return assessment
            
        except Exception as e:
            logger.error(f"Overall assessment generation failed: {e}")
            return {
                'overall_quality': 'unknown',
                'error': str(e)
            }
    
    def _generate_professional_insights(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Generate professional mixing and mastering insights"""
        try:
            insights = {
                'mixing_suggestions': [],
                'mastering_suggestions': [],
                'eq_recommendations': [],
                'dynamics_recommendations': [],
                'stereo_recommendations': [],
                'platform_optimization': [],
                'workflow_suggestions': []
            }
            
            # Frequency-based insights
            frequency_analysis = analysis_results.get('frequency_analysis', {})
            if 'error' not in frequency_analysis:
                mixing_insights = frequency_analysis.get('mixing_insights', {})
                insights['eq_recommendations'].extend(mixing_insights.get('eq_suggestions', []))
                insights['mixing_suggestions'].extend(mixing_insights.get('mixing_notes', []))
            
            # Loudness-based insights
            loudness_analysis = analysis_results.get('loudness_analysis', {})
            if 'error' not in loudness_analysis:
                # Get loudness suggestions
                try:
                    loudness_suggestions = self.loudness_analyzer.suggest_loudness_adjustments(loudness_analysis)
                    for suggestion in loudness_suggestions.get('suggestions', []):
                        if suggestion.get('type') == 'gain_adjustment':
                            insights['mastering_suggestions'].append(
                                f"Adjust gain by {suggestion.get('adjustment_db', 0):+.1f} dB for {suggestion.get('platform', 'streaming')} compliance"
                            )
                        elif suggestion.get('type') == 'peak_limiting':
                            insights['mastering_suggestions'].append(
                                f"Apply peak limiting for {suggestion.get('platform', 'streaming')} compliance"
                            )
                except Exception as e:
                    logger.warning(f"Loudness suggestions generation failed: {e}")
                
                # Dynamic range insights
                dr_analysis = loudness_analysis.get('dynamic_range_analysis', {})
                dr_quality = dr_analysis.get('dynamic_range_quality', 'unknown')
                
                if dr_quality == 'over_compressed':
                    insights['dynamics_recommendations'].append("Reduce compression to improve dynamic range")
                elif dr_quality == 'compressed':
                    insights['dynamics_recommendations'].append("Consider lighter compression for more dynamics")
                elif dr_quality == 'excellent':
                    insights['dynamics_recommendations'].append("Excellent dynamic range - preserve in mastering")
            
            # Stereo-based insights
            stereo_analysis = analysis_results.get('stereo_analysis', {})
            if 'error' not in stereo_analysis:
                stereo_insights = stereo_analysis.get('mixing_insights', {})
                insights['stereo_recommendations'].extend(stereo_insights.get('recommendations', []))
                
                # Phase correlation warnings
                phase_issues = stereo_analysis.get('phase_analysis', {}).get('phase_issues', [])
                for issue in phase_issues:
                    if issue.get('type') == 'out_of_phase':
                        insights['mixing_suggestions'].append("Phase cancellation detected - check stereo alignment")
            
            # Tempo and key insights
            tempo_analysis = analysis_results.get('tempo_analysis', {})
            if 'error' not in tempo_analysis:
                try:
                    tempo_suggestions = self.tempo_detector.suggest_tempo_adjustments(tempo_analysis)
                    for suggestion in tempo_suggestions.get('suggestions', []):
                        if suggestion.get('type') == 'dj_friendly':
                            insights['workflow_suggestions'].append(
                                f"Consider tempo adjustment to {suggestion.get('target_tempo')} BPM for DJ compatibility"
                            )
                except Exception as e:
                    logger.warning(f"Tempo suggestions generation failed: {e}")
            
            key_analysis = analysis_results.get('key_analysis', {})
            if 'error' not in key_analysis:
                try:
                    key_suggestions = self.key_detector.suggest_key_adjustments(key_analysis)
                    insights['workflow_suggestions'].extend(key_suggestions.get('reasoning', []))
                except Exception as e:
                    logger.warning(f"Key suggestions generation failed: {e}")
            
            # Platform optimization insights
            compliance_analysis = loudness_analysis.get('compliance_analysis', {})
            platform_compliance = compliance_analysis.get('platform_compliance', {})
            
            compliant_platforms = [p for p, data in platform_compliance.items() if data.get('overall_compliant', False)]
            non_compliant_platforms = [p for p, data in platform_compliance.items() if not data.get('overall_compliant', False)]
            
            if compliant_platforms:
                insights['platform_optimization'].append(f"Currently compliant with: {', '.join(compliant_platforms)}")
            
            if non_compliant_platforms:
                insights['platform_optimization'].append(f"Needs adjustment for: {', '.join(non_compliant_platforms)}")
            
            # Remove empty sections
            insights = {k: v for k, v in insights.items() if v}
            
            return insights
            
        except Exception as e:
            logger.error(f"Professional insights generation failed: {e}")
            return {'error': str(e)}
    
    def _load_cached_analysis(self, file_path: Path) -> Optional[Dict[str, any]]:
        """Load cached analysis results if available"""
        try:
            cache_file = self._get_cache_file_path(file_path)
            
            if cache_file.exists():
                # Check if cache is still valid
                file_mtime = file_path.stat().st_mtime
                cache_mtime = cache_file.stat().st_mtime
                
                # Cache is valid if it's newer than the audio file
                if cache_mtime > file_mtime:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Verify cache version
                    if cached_data.get('analysis_metadata', {}).get('analysis_version') == self.analysis_version:
                        logger.debug(f"Loaded cached analysis: {file_path.name}")
                        return cached_data
                    else:
                        logger.debug(f"Cache version mismatch, ignoring: {file_path.name}")
                else:
                    logger.debug(f"Cache outdated, ignoring: {file_path.name}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load cached analysis: {e}")
            return None
    
    def _cache_analysis_results(self, file_path: Path, results: Dict[str, any]) -> None:
        """Cache analysis results to disk"""
        try:
            cache_file = self._get_cache_file_path(file_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.debug(f"Cached analysis results: {file_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to cache analysis results: {e}")
    
    def _get_cache_file_path(self, file_path: Path) -> Path:
        """Get cache file path for audio file"""
        cache_dir = project_settings.ANALYSIS_CACHE_DIR
        cache_name = f"{file_path.stem}_{hash(str(file_path))}.json"
        return cache_dir / cache_name
    
    def _make_json_serializable(self, obj) -> any:
        """Make object JSON serializable by converting numpy arrays"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def batch_analyze(self, 
                     file_paths: List[Union[str, Path]], 
                     analysis_types: Optional[List[str]] = None,
                     **kwargs) -> Dict[str, Dict[str, any]]:
        """
        Batch analyze multiple audio files
        
        Args:
            file_paths: List of audio file paths
            analysis_types: List of analysis types to perform
            **kwargs: Additional arguments for analyze_track
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        results = {}
        total_files = len(file_paths)
        
        logger.info(f"Starting batch analysis of {total_files} files")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Processing file {i}/{total_files}: {Path(file_path).name}")
                results[str(file_path)] = self.analyze_track(
                    file_path, 
                    analysis_types=analysis_types, 
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                results[str(file_path)] = {'error': str(e)}
        
        logger.info(f"Batch analysis completed: {len(results)} files processed")
        
        return results
    
    def compare_with_reference(self, 
                              track_path: Union[str, Path], 
                              reference_path: Union[str, Path],
                              comparison_aspects: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Compare track with reference track
        
        Args:
            track_path: Path to track to analyze
            reference_path: Path to reference track
            comparison_aspects: Aspects to compare (None = all)
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {Path(track_path).name} with reference {Path(reference_path).name}")
        
        # Analyze both tracks
        track_analysis = self.analyze_track(track_path)
        reference_analysis = self.analyze_track(reference_path)
        
        # Perform comparison
        comparison = self._compare_analyses(track_analysis, reference_analysis, comparison_aspects)
        
        return {
            'track_analysis': track_analysis,
            'reference_analysis': reference_analysis,
            'comparison': comparison
        }
    
    def _compare_analyses(self, 
                         track_analysis: Dict[str, any], 
                         reference_analysis: Dict[str, any],
                         aspects: Optional[List[str]] = None) -> Dict[str, any]:
        """Compare two analysis results"""
        if aspects is None:
            aspects = ['tempo', 'key', 'frequency', 'loudness', 'stereo']
        
        comparison = {}
        
        # Tempo comparison
        if 'tempo' in aspects:
            track_tempo = track_analysis.get('tempo_analysis', {}).get('primary_tempo')
            ref_tempo = reference_analysis.get('tempo_analysis', {}).get('primary_tempo')
            
            if track_tempo and ref_tempo:
                tempo_diff = track_tempo - ref_tempo
                comparison['tempo'] = {
                    'track_tempo': track_tempo,
                    'reference_tempo': ref_tempo,
                    'difference_bpm': tempo_diff,
                    'percentage_difference': (tempo_diff / ref_tempo) * 100
                }
        
        # Loudness comparison
        if 'loudness' in aspects:
            track_lufs = track_analysis.get('loudness_analysis', {}).get('integrated_loudness')
            ref_lufs = reference_analysis.get('loudness_analysis', {}).get('integrated_loudness')
            
            if track_lufs and ref_lufs and not np.isneginf(track_lufs) and not np.isneginf(ref_lufs):
                lufs_diff = track_lufs - ref_lufs
                comparison['loudness'] = {
                    'track_lufs': track_lufs,
                    'reference_lufs': ref_lufs,
                    'difference_lufs': lufs_diff,
                    'recommendation': 'increase level' if lufs_diff < -2 else 'decrease level' if lufs_diff > 2 else 'level acceptable'
                }
        
        # Add more comparisons as needed...
        
        return comparison
    
    def get_analysis_summary(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """Generate a concise summary of analysis results"""
        try:
            summary = {
                'file_name': analysis_results.get('analysis_metadata', {}).get('file_name', 'Unknown'),
                'overall_quality': analysis_results.get('overall_assessment', {}).get('overall_quality', 'Unknown'),
                'key_metrics': {},
                'main_issues': [],
                'recommendations': []
            }
            
            # Extract key metrics
            if 'tempo_analysis' in analysis_results:
                tempo = analysis_results['tempo_analysis'].get('primary_tempo')
                if tempo:
                    summary['key_metrics']['tempo'] = f"{tempo:.1f} BPM"
            
            if 'key_analysis' in analysis_results:
                key = analysis_results['key_analysis'].get('primary_key')
                if key:
                    summary['key_metrics']['key'] = key
            
            if 'loudness_analysis' in analysis_results:
                lufs = analysis_results['loudness_analysis'].get('integrated_loudness')
                if lufs and not np.isneginf(lufs):
                    summary['key_metrics']['loudness'] = f"{lufs:.1f} LUFS"
            
            # Extract main issues
            overall_assessment = analysis_results.get('overall_assessment', {})
            summary['main_issues'] = overall_assessment.get('priority_issues', [])[:3]
            
            # Extract top recommendations
            professional_insights = analysis_results.get('professional_insights', {})
            all_recommendations = []
            
            for category in ['mixing_suggestions', 'mastering_suggestions', 'eq_recommendations']:
                all_recommendations.extend(professional_insights.get(category, []))
            
            summary['recommendations'] = all_recommendations[:3]
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {'error': str(e)}
    
    def export_analysis_report(self, 
                              analysis_results: Dict[str, any], 
                              output_path: Union[str, Path],
                              format: str = 'json') -> None:
        """
        Export analysis results to file
        
        Args:
            analysis_results: Analysis results to export
            output_path: Output file path
            format: Export format ('json', 'txt', 'csv')
        """
        output_path = Path(output_path)
        
        try:
            if format.lower() == 'json':
                serializable_results = self._make_json_serializable(analysis_results)
                with open(output_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            
            elif format.lower() == 'txt':
                self._export_text_report(analysis_results, output_path)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Analysis report exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export analysis report: {e}")
            raise
    
    def _export_text_report(self, analysis_results: Dict[str, any], output_path: Path) -> None:
        """Export analysis results as formatted text report"""
        with open(output_path, 'w') as f:
            f.write("BEATWIZARD ENHANCED AUDIO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # File information
            metadata = analysis_results.get('analysis_metadata', {})
            f.write(f"File: {metadata.get('file_name', 'Unknown')}\n")
            f.write(f"Analysis Date: {time.ctime(metadata.get('analysis_timestamp', 0))}\n")
            f.write(f"Analysis Duration: {metadata.get('analysis_duration', 0):.2f}s\n\n")
            
            # Overall assessment
            overall = analysis_results.get('overall_assessment', {})
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Quality: {overall.get('overall_quality', 'Unknown')}\n")
            f.write(f"Technical Quality: {overall.get('technical_quality', 'Unknown')}\n")
            f.write(f"Mix Quality: {overall.get('mix_quality', 'Unknown')}\n")
            f.write(f"Mastering Ready: {'Yes' if overall.get('mastering_readiness', False) else 'No'}\n")
            f.write(f"Commercial Ready: {'Yes' if overall.get('commercial_readiness', False) else 'No'}\n\n")
            
            # Key metrics
            f.write("KEY METRICS\n")
            f.write("-" * 12 + "\n")
            
            if 'tempo_analysis' in analysis_results:
                tempo = analysis_results['tempo_analysis'].get('primary_tempo')
                if tempo:
                    f.write(f"Tempo: {tempo:.1f} BPM\n")
            
            if 'key_analysis' in analysis_results:
                key = analysis_results['key_analysis'].get('primary_key')
                if key:
                    f.write(f"Key: {key}\n")
            
            if 'loudness_analysis' in analysis_results:
                lufs = analysis_results['loudness_analysis'].get('integrated_loudness')
                if lufs and not np.isneginf(lufs):
                    f.write(f"Loudness: {lufs:.1f} LUFS\n")
            
            f.write("\n")
            
            # Issues and recommendations
            f.write("ISSUES AND RECOMMENDATIONS\n")
            f.write("-" * 25 + "\n")
            
            issues = overall.get('weaknesses', [])
            if issues:
                f.write("Issues Found:\n")
                for i, issue in enumerate(issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            insights = analysis_results.get('professional_insights', {})
            if insights:
                f.write("Recommendations:\n")
                all_recommendations = []
                for category, recs in insights.items():
                    all_recommendations.extend(recs)
                
                for i, rec in enumerate(all_recommendations[:10], 1):
                    f.write(f"{i}. {rec}\n")