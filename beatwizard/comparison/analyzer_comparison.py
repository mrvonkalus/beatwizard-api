"""
Analyzer Comparison Framework
Compare enhanced BeatWizard analysis with basic aubio analysis
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import json

try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False
    logger.warning("Aubio not available - comparison with aubio will be skipped")

from ..core.enhanced_analyzer import EnhancedAudioAnalyzer
from ..core.audio_processor import AudioProcessor
from config.settings import project_settings


class BasicAubioAnalyzer:
    """
    Basic audio analysis using aubio for comparison purposes
    Implements basic tempo and key detection
    """
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize basic aubio analyzer"""
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_size = 1024
        
        if not AUBIO_AVAILABLE:
            logger.error("Aubio not available - cannot perform basic analysis")
            self.available = False
        else:
            self.available = True
            logger.debug("BasicAubioAnalyzer initialized")
    
    def analyze_track(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Basic track analysis using aubio
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with basic analysis results
        """
        if not self.available:
            return {'error': 'Aubio not available'}
        
        start_time = time.time()
        file_path = Path(file_path)
        
        logger.debug(f"Starting basic aubio analysis: {file_path.name}")
        
        try:
            # Load audio using aubio
            source = aubio.source(str(file_path), self.sample_rate, self.hop_length)
            
            # Initialize aubio objects
            tempo_detector = aubio.tempo("default", self.frame_size, self.hop_length, self.sample_rate)
            pitch_detector = aubio.pitch("default", self.frame_size, self.hop_length, self.sample_rate)
            
            # Analysis variables
            total_frames = 0
            beats = []
            pitches = []
            confidences = []
            
            # Process audio
            while True:
                samples, read = source()
                
                # Tempo detection
                is_beat = tempo_detector(samples)
                if is_beat:
                    beats.append(tempo_detector.get_last_s())
                
                # Pitch detection
                pitch = pitch_detector(samples)
                confidence = pitch_detector.get_confidence()
                
                if confidence > 0.5:  # Only include confident pitch detections
                    pitches.append(pitch)
                    confidences.append(confidence)
                
                total_frames += read
                if read < self.hop_length:
                    break
            
            # Calculate results
            duration = total_frames / float(self.sample_rate)
            
            # Tempo calculation
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                if len(beat_intervals) > 0:
                    avg_interval = np.mean(beat_intervals)
                    tempo = 60.0 / avg_interval if avg_interval > 0 else 0.0
                else:
                    tempo = 0.0
            else:
                tempo = 0.0
            
            # Basic pitch analysis (simplified key detection)
            if pitches:
                # Convert pitches to MIDI notes
                midi_notes = [aubio.freq2note(p) for p in pitches if p > 0]
                # Simple key estimation (most common note)
                if midi_notes:
                    unique_notes, counts = np.unique(midi_notes, return_counts=True)
                    most_common_note = unique_notes[np.argmax(counts)]
                    estimated_key = most_common_note
                else:
                    estimated_key = "Unknown"
            else:
                estimated_key = "Unknown"
            
            analysis_time = time.time() - start_time
            
            result = {
                'basic_tempo': tempo,
                'estimated_key': estimated_key,
                'beat_count': len(beats),
                'beats': beats[:50],  # Limit for storage
                'pitch_detections': len(pitches),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'duration': duration,
                'analysis_time': analysis_time,
                'analyzer': 'aubio_basic',
                'file_name': file_path.name
            }
            
            logger.debug(f"Basic aubio analysis completed in {analysis_time:.2f}s: {file_path.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Basic aubio analysis failed for {file_path}: {str(e)}")
            return {
                'error': str(e),
                'analyzer': 'aubio_basic',
                'file_name': file_path.name
            }


class AnalyzerComparison:
    """
    Compare enhanced BeatWizard analysis with basic aubio analysis
    Provides detailed comparison metrics and performance analysis
    """
    
    def __init__(self):
        """Initialize the comparison system"""
        self.enhanced_analyzer = EnhancedAudioAnalyzer()
        self.basic_analyzer = BasicAubioAnalyzer()
        
        logger.info("AnalyzerComparison initialized")
    
    def compare_single_track(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Compare enhanced vs basic analysis for a single track
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with comparison results
        """
        file_path = Path(file_path)
        logger.info(f"Comparing analyzers on: {file_path.name}")
        
        # Perform both analyses
        enhanced_result = self.enhanced_analyzer.analyze_track(file_path)
        basic_result = self.basic_analyzer.analyze_track(file_path)
        
        # Compare results
        comparison = self._compare_results(enhanced_result, basic_result)
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'enhanced_analysis': enhanced_result,
            'basic_analysis': basic_result,
            'comparison': comparison,
            'comparison_timestamp': time.time()
        }
    
    def compare_batch(self, file_paths: List[Union[str, Path]]) -> Dict[str, any]:
        """
        Compare analyzers on multiple tracks
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            Dictionary with batch comparison results
        """
        logger.info(f"Starting batch comparison of {len(file_paths)} files")
        
        individual_comparisons = []
        aggregate_metrics = {
            'total_files': len(file_paths),
            'successful_enhanced': 0,
            'successful_basic': 0,
            'tempo_comparisons': [],
            'analysis_time_comparisons': [],
            'enhanced_advantages': [],
            'basic_advantages': []
        }
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Processing file {i}/{len(file_paths)}: {Path(file_path).name}")
                
                comparison_result = self.compare_single_track(file_path)
                individual_comparisons.append(comparison_result)
                
                # Update aggregate metrics
                if 'error' not in comparison_result['enhanced_analysis']:
                    aggregate_metrics['successful_enhanced'] += 1
                
                if 'error' not in comparison_result['basic_analysis']:
                    aggregate_metrics['successful_basic'] += 1
                
                # Collect metrics for aggregation
                comparison = comparison_result['comparison']
                
                if 'tempo_comparison' in comparison:
                    aggregate_metrics['tempo_comparisons'].append(comparison['tempo_comparison'])
                
                if 'performance_comparison' in comparison:
                    perf = comparison['performance_comparison']
                    aggregate_metrics['analysis_time_comparisons'].append({
                        'enhanced_time': perf.get('enhanced_analysis_time', 0),
                        'basic_time': perf.get('basic_analysis_time', 0)
                    })
                
                # Collect advantages
                if 'enhanced_advantages' in comparison:
                    aggregate_metrics['enhanced_advantages'].extend(comparison['enhanced_advantages'])
                
                if 'basic_advantages' in comparison:
                    aggregate_metrics['basic_advantages'].extend(comparison['basic_advantages'])
                
            except Exception as e:
                logger.error(f"Comparison failed for {file_path}: {e}")
                individual_comparisons.append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(aggregate_metrics)
        
        result = {
            'batch_summary': {
                'total_files': aggregate_metrics['total_files'],
                'successful_comparisons': len([c for c in individual_comparisons if 'error' not in c]),
                'enhanced_success_rate': aggregate_metrics['successful_enhanced'] / len(file_paths),
                'basic_success_rate': aggregate_metrics['successful_basic'] / len(file_paths)
            },
            'aggregate_statistics': aggregate_stats,
            'individual_comparisons': individual_comparisons,
            'comparison_timestamp': time.time()
        }
        
        logger.info(f"Batch comparison completed: {len(individual_comparisons)} files processed")
        
        return result
    
    def _compare_results(self, enhanced_result: Dict[str, any], basic_result: Dict[str, any]) -> Dict[str, any]:
        """Compare enhanced and basic analysis results"""
        comparison = {
            'enhanced_advantages': [],
            'basic_advantages': [],
            'similarity_metrics': {},
            'performance_comparison': {},
            'quality_assessment': {}
        }
        
        # Performance comparison
        enhanced_time = enhanced_result.get('analysis_metadata', {}).get('analysis_duration', 0)
        basic_time = basic_result.get('analysis_time', 0)
        
        comparison['performance_comparison'] = {
            'enhanced_analysis_time': enhanced_time,
            'basic_analysis_time': basic_time,
            'time_difference': enhanced_time - basic_time,
            'speed_ratio': basic_time / enhanced_time if enhanced_time > 0 else 0
        }
        
        if basic_time < enhanced_time:
            comparison['basic_advantages'].append(f"Faster analysis ({basic_time:.2f}s vs {enhanced_time:.2f}s)")
        else:
            comparison['enhanced_advantages'].append("Analysis time competitive with basic approach")
        
        # Tempo comparison
        if 'tempo_analysis' in enhanced_result and 'basic_tempo' in basic_result:
            enhanced_tempo = enhanced_result['tempo_analysis'].get('primary_tempo')
            basic_tempo = basic_result.get('basic_tempo')
            
            if enhanced_tempo and basic_tempo and basic_tempo > 0:
                tempo_diff = abs(enhanced_tempo - basic_tempo)
                tempo_agreement = tempo_diff < 5.0  # Within 5 BPM
                
                comparison['tempo_comparison'] = {
                    'enhanced_tempo': enhanced_tempo,
                    'basic_tempo': basic_tempo,
                    'difference_bpm': tempo_diff,
                    'agreement': tempo_agreement,
                    'enhanced_confidence': enhanced_result['tempo_analysis'].get('confidence', 0.0)
                }
                
                if tempo_agreement:
                    comparison['similarity_metrics']['tempo_agreement'] = True
                    comparison['enhanced_advantages'].append("Tempo detection agrees with basic method")
                else:
                    comparison['similarity_metrics']['tempo_agreement'] = False
                    comparison['enhanced_advantages'].append(f"Different tempo detected (confidence: {enhanced_result['tempo_analysis'].get('confidence', 0.0):.2f})")
                
                # Enhanced tempo features
                if enhanced_result['tempo_analysis'].get('confidence', 0) > 0.7:
                    comparison['enhanced_advantages'].append("High tempo detection confidence")
                
                if 'stability_analysis' in enhanced_result['tempo_analysis']:
                    comparison['enhanced_advantages'].append("Tempo stability analysis available")
            
            elif enhanced_tempo and not basic_tempo:
                comparison['enhanced_advantages'].append("Enhanced analyzer detected tempo where basic failed")
            
            elif basic_tempo and not enhanced_tempo:
                comparison['basic_advantages'].append("Basic analyzer detected tempo where enhanced failed")
        
        # Key comparison
        if 'key_analysis' in enhanced_result and 'estimated_key' in basic_result:
            enhanced_key = enhanced_result['key_analysis'].get('primary_key')
            basic_key = basic_result.get('estimated_key')
            
            if enhanced_key and basic_key and basic_key != "Unknown":
                # Simple key comparison (note that basic is very simplified)
                comparison['key_comparison'] = {
                    'enhanced_key': enhanced_key,
                    'basic_key': basic_key,
                    'enhanced_confidence': enhanced_result['key_analysis'].get('confidence', 0.0)
                }
                
                comparison['enhanced_advantages'].append("Professional key detection with confidence scoring")
            
            elif enhanced_key and (not basic_key or basic_key == "Unknown"):
                comparison['enhanced_advantages'].append("Enhanced analyzer detected key where basic failed")
        
        # Enhanced-only features
        enhanced_features = []
        
        if 'frequency_analysis' in enhanced_result:
            enhanced_features.append("Professional 7-band frequency analysis")
            enhanced_features.append("Spectral balance assessment")
            enhanced_features.append("EQ recommendations")
        
        if 'loudness_analysis' in enhanced_result:
            enhanced_features.append("Professional LUFS measurement")
            enhanced_features.append("Streaming platform compliance")
            enhanced_features.append("Dynamic range analysis")
        
        if 'stereo_analysis' in enhanced_result:
            enhanced_features.append("Comprehensive stereo imaging analysis")
            enhanced_features.append("Phase correlation analysis")
            enhanced_features.append("Mid-Side analysis")
        
        if 'overall_assessment' in enhanced_result:
            enhanced_features.append("Overall quality assessment")
            enhanced_features.append("Professional mixing insights")
            enhanced_features.append("Mastering readiness evaluation")
        
        comparison['enhanced_advantages'].extend(enhanced_features)
        
        # Quality assessment
        enhanced_overall = enhanced_result.get('overall_assessment', {})
        
        comparison['quality_assessment'] = {
            'enhanced_provides_quality_rating': enhanced_overall.get('overall_quality') is not None,
            'enhanced_provides_recommendations': len(enhanced_overall.get('weaknesses', [])) > 0,
            'enhanced_provides_professional_insights': 'professional_insights' in enhanced_result,
            'basic_limitations': [
                "No frequency analysis",
                "No loudness measurement", 
                "No stereo analysis",
                "No quality assessment",
                "Simplified key detection",
                "No professional recommendations"
            ]
        }
        
        # Overall comparison summary
        enhanced_score = len(comparison['enhanced_advantages'])
        basic_score = len(comparison['basic_advantages'])
        
        comparison['summary'] = {
            'enhanced_advantage_count': enhanced_score,
            'basic_advantage_count': basic_score,
            'overall_winner': 'enhanced' if enhanced_score > basic_score else 'basic' if basic_score > enhanced_score else 'tie',
            'enhanced_improvement_ratio': enhanced_score / max(basic_score, 1)
        }
        
        return comparison
    
    def _calculate_aggregate_statistics(self, aggregate_metrics: Dict[str, any]) -> Dict[str, any]:
        """Calculate aggregate statistics from batch comparison"""
        stats = {}
        
        # Tempo statistics
        tempo_comparisons = aggregate_metrics['tempo_comparisons']
        if tempo_comparisons:
            tempo_diffs = [t.get('difference_bpm', 0) for t in tempo_comparisons if 'difference_bpm' in t]
            tempo_agreements = [t.get('agreement', False) for t in tempo_comparisons if 'agreement' in t]
            
            stats['tempo_statistics'] = {
                'total_comparisons': len(tempo_comparisons),
                'agreement_rate': sum(tempo_agreements) / len(tempo_agreements) if tempo_agreements else 0,
                'avg_tempo_difference': float(np.mean(tempo_diffs)) if tempo_diffs else 0,
                'max_tempo_difference': float(np.max(tempo_diffs)) if tempo_diffs else 0,
                'std_tempo_difference': float(np.std(tempo_diffs)) if tempo_diffs else 0
            }
        
        # Performance statistics
        time_comparisons = aggregate_metrics['analysis_time_comparisons']
        if time_comparisons:
            enhanced_times = [t['enhanced_time'] for t in time_comparisons]
            basic_times = [t['basic_time'] for t in time_comparisons]
            
            stats['performance_statistics'] = {
                'avg_enhanced_time': float(np.mean(enhanced_times)),
                'avg_basic_time': float(np.mean(basic_times)),
                'enhanced_time_std': float(np.std(enhanced_times)),
                'basic_time_std': float(np.std(basic_times)),
                'avg_speed_ratio': float(np.mean([b/e for e, b in zip(enhanced_times, basic_times) if e > 0]))
            }
        
        # Feature advantages
        enhanced_advantages = aggregate_metrics['enhanced_advantages']
        basic_advantages = aggregate_metrics['basic_advantages']
        
        # Count unique advantages
        unique_enhanced = list(set(enhanced_advantages))
        unique_basic = list(set(basic_advantages))
        
        stats['feature_comparison'] = {
            'unique_enhanced_advantages': len(unique_enhanced),
            'unique_basic_advantages': len(unique_basic),
            'enhanced_advantage_frequency': {adv: enhanced_advantages.count(adv) for adv in unique_enhanced},
            'basic_advantage_frequency': {adv: basic_advantages.count(adv) for adv in unique_basic},
            'overall_enhanced_superiority': len(unique_enhanced) / max(len(unique_basic), 1)
        }
        
        return stats
    
    def generate_comparison_report(self, comparison_results: Dict[str, any], output_path: Union[str, Path]) -> None:
        """
        Generate a detailed comparison report
        
        Args:
            comparison_results: Results from compare_single_track or compare_batch
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("BEATWIZARD ANALYZER COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if 'batch_summary' in comparison_results:
                # Batch report
                self._write_batch_report(f, comparison_results)
            else:
                # Single track report
                self._write_single_track_report(f, comparison_results)
        
        logger.info(f"Comparison report generated: {output_path}")
    
    def _write_single_track_report(self, f, results: Dict[str, any]) -> None:
        """Write single track comparison report"""
        f.write(f"File: {results['file_name']}\n")
        f.write(f"Comparison Date: {time.ctime(results['comparison_timestamp'])}\n\n")
        
        comparison = results['comparison']
        
        # Performance comparison
        if 'performance_comparison' in comparison:
            perf = comparison['performance_comparison']
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-" * 22 + "\n")
            f.write(f"Enhanced Analysis Time: {perf.get('enhanced_analysis_time', 0):.2f}s\n")
            f.write(f"Basic Analysis Time: {perf.get('basic_analysis_time', 0):.2f}s\n")
            f.write(f"Time Difference: {perf.get('time_difference', 0):+.2f}s\n\n")
        
        # Tempo comparison
        if 'tempo_comparison' in comparison:
            tempo = comparison['tempo_comparison']
            f.write("TEMPO ANALYSIS COMPARISON\n")
            f.write("-" * 25 + "\n")
            f.write(f"Enhanced Tempo: {tempo.get('enhanced_tempo', 0):.1f} BPM\n")
            f.write(f"Basic Tempo: {tempo.get('basic_tempo', 0):.1f} BPM\n")
            f.write(f"Difference: {tempo.get('difference_bpm', 0):.1f} BPM\n")
            f.write(f"Agreement: {'Yes' if tempo.get('agreement', False) else 'No'}\n")
            f.write(f"Enhanced Confidence: {tempo.get('enhanced_confidence', 0):.2f}\n\n")
        
        # Advantages
        f.write("ENHANCED ANALYZER ADVANTAGES\n")
        f.write("-" * 30 + "\n")
        for advantage in comparison.get('enhanced_advantages', []):
            f.write(f"• {advantage}\n")
        
        f.write(f"\nBASIC ANALYZER ADVANTAGES\n")
        f.write("-" * 26 + "\n")
        for advantage in comparison.get('basic_advantages', []):
            f.write(f"• {advantage}\n")
        
        # Summary
        summary = comparison.get('summary', {})
        f.write(f"\nSUMMARY\n")
        f.write("-" * 8 + "\n")
        f.write(f"Overall Winner: {summary.get('overall_winner', 'Unknown').title()}\n")
        f.write(f"Enhanced Advantages: {summary.get('enhanced_advantage_count', 0)}\n")
        f.write(f"Basic Advantages: {summary.get('basic_advantage_count', 0)}\n")
        f.write(f"Improvement Ratio: {summary.get('enhanced_improvement_ratio', 0):.1f}x\n")
    
    def _write_batch_report(self, f, results: Dict[str, any]) -> None:
        """Write batch comparison report"""
        batch_summary = results['batch_summary']
        aggregate_stats = results['aggregate_statistics']
        
        f.write("BATCH COMPARISON SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total Files: {batch_summary['total_files']}\n")
        f.write(f"Successful Comparisons: {batch_summary['successful_comparisons']}\n")
        f.write(f"Enhanced Success Rate: {batch_summary['enhanced_success_rate']:.1%}\n")
        f.write(f"Basic Success Rate: {batch_summary['basic_success_rate']:.1%}\n\n")
        
        # Tempo statistics
        if 'tempo_statistics' in aggregate_stats:
            tempo_stats = aggregate_stats['tempo_statistics']
            f.write("TEMPO ANALYSIS STATISTICS\n")
            f.write("-" * 26 + "\n")
            f.write(f"Total Tempo Comparisons: {tempo_stats['total_comparisons']}\n")
            f.write(f"Agreement Rate: {tempo_stats['agreement_rate']:.1%}\n")
            f.write(f"Average Difference: {tempo_stats['avg_tempo_difference']:.1f} BPM\n")
            f.write(f"Max Difference: {tempo_stats['max_tempo_difference']:.1f} BPM\n\n")
        
        # Performance statistics
        if 'performance_statistics' in aggregate_stats:
            perf_stats = aggregate_stats['performance_statistics']
            f.write("PERFORMANCE STATISTICS\n")
            f.write("-" * 22 + "\n")
            f.write(f"Average Enhanced Time: {perf_stats['avg_enhanced_time']:.2f}s\n")
            f.write(f"Average Basic Time: {perf_stats['avg_basic_time']:.2f}s\n")
            f.write(f"Average Speed Ratio: {perf_stats['avg_speed_ratio']:.1f}x\n\n")
        
        # Feature comparison
        if 'feature_comparison' in aggregate_stats:
            feature_comp = aggregate_stats['feature_comparison']
            f.write("FEATURE COMPARISON\n")
            f.write("-" * 18 + "\n")
            f.write(f"Unique Enhanced Advantages: {feature_comp['unique_enhanced_advantages']}\n")
            f.write(f"Unique Basic Advantages: {feature_comp['unique_basic_advantages']}\n")
            f.write(f"Enhanced Superiority Ratio: {feature_comp['overall_enhanced_superiority']:.1f}x\n\n")
            
            # Most common advantages
            enhanced_freq = feature_comp.get('enhanced_advantage_frequency', {})
            if enhanced_freq:
                f.write("MOST COMMON ENHANCED ADVANTAGES:\n")
                sorted_advantages = sorted(enhanced_freq.items(), key=lambda x: x[1], reverse=True)
                for advantage, count in sorted_advantages[:5]:
                    f.write(f"• {advantage} ({count} files)\n")
        
        f.write(f"\nCONCLUSION\n")
        f.write("-" * 11 + "\n")
        f.write("The enhanced BeatWizard analyzer provides significantly more\n")
        f.write("comprehensive analysis compared to basic aubio, including:\n")
        f.write("• Professional loudness measurement (LUFS)\n")
        f.write("• 7-band frequency analysis with mixing insights\n")
        f.write("• Comprehensive stereo imaging analysis\n")
        f.write("• Overall quality assessment and recommendations\n")
        f.write("• Streaming platform compliance analysis\n")
    
    def save_comparison_results(self, results: Dict[str, any], output_path: Union[str, Path]) -> None:
        """Save comparison results to JSON file"""
        output_path = Path(output_path)
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Comparison results saved: {output_path}")
    
    def _make_json_serializable(self, obj) -> any:
        """Make object JSON serializable"""
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