"""
Validation Framework for BeatWizard Analysis
Test and validate analysis accuracy against known reference tracks
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import time

from ..core.enhanced_analyzer import EnhancedAudioAnalyzer
from ..comparison.analyzer_comparison import AnalyzerComparison
from config.settings import project_settings


class ValidationDataset:
    """
    Manage validation dataset with known ground truth values
    """
    
    def __init__(self, dataset_path: Optional[Union[str, Path]] = None):
        """
        Initialize validation dataset
        
        Args:
            dataset_path: Path to dataset metadata file
        """
        self.dataset_path = Path(dataset_path) if dataset_path else project_settings.VALIDATION_RESULTS_DIR / "dataset.json"
        self.dataset = self._load_dataset()
        
        logger.info(f"ValidationDataset initialized with {len(self.dataset)} entries")
    
    def _load_dataset(self) -> Dict[str, Dict[str, any]]:
        """Load dataset from file or create empty"""
        if self.dataset_path.exists():
            try:
                with open(self.dataset_path, 'r') as f:
                    dataset = json.load(f)
                logger.info(f"Loaded dataset from {self.dataset_path}")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load dataset from {self.dataset_path}: {e}")
        
        # Create empty dataset
        return {}
    
    def add_reference_track(self, 
                           file_path: Union[str, Path],
                           ground_truth: Dict[str, any],
                           metadata: Optional[Dict[str, any]] = None) -> None:
        """
        Add reference track with known ground truth values
        
        Args:
            file_path: Path to audio file
            ground_truth: Known correct values for validation
            metadata: Additional metadata about the track
        """
        file_path = Path(file_path)
        track_key = str(file_path)
        
        self.dataset[track_key] = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'ground_truth': ground_truth,
            'metadata': metadata or {},
            'added_timestamp': time.time()
        }
        
        self._save_dataset()
        logger.info(f"Added reference track: {file_path.name}")
    
    def remove_reference_track(self, file_path: Union[str, Path]) -> None:
        """Remove reference track from dataset"""
        track_key = str(Path(file_path))
        
        if track_key in self.dataset:
            del self.dataset[track_key]
            self._save_dataset()
            logger.info(f"Removed reference track: {Path(file_path).name}")
        else:
            logger.warning(f"Track not found in dataset: {Path(file_path).name}")
    
    def get_reference_tracks(self) -> List[Dict[str, any]]:
        """Get all reference tracks"""
        return list(self.dataset.values())
    
    def get_track_ground_truth(self, file_path: Union[str, Path]) -> Optional[Dict[str, any]]:
        """Get ground truth for specific track"""
        track_key = str(Path(file_path))
        return self.dataset.get(track_key, {}).get('ground_truth')
    
    def _save_dataset(self) -> None:
        """Save dataset to file"""
        try:
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dataset_path, 'w') as f:
                json.dump(self.dataset, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")


class ValidationMetrics:
    """
    Calculate validation metrics for analysis accuracy
    """
    
    @staticmethod
    def calculate_tempo_accuracy(predicted_tempo: float, 
                               ground_truth_tempo: float,
                               tolerance_bpm: float = 2.0) -> Dict[str, any]:
        """
        Calculate tempo detection accuracy
        
        Args:
            predicted_tempo: Detected tempo
            ground_truth_tempo: Known correct tempo
            tolerance_bpm: Tolerance in BPM for "correct" detection
            
        Returns:
            Dictionary with accuracy metrics
        """
        if predicted_tempo is None or ground_truth_tempo is None:
            return {
                'accurate': False,
                'error_bpm': float('inf'),
                'error_percentage': float('inf'),
                'within_tolerance': False
            }
        
        error_bpm = abs(predicted_tempo - ground_truth_tempo)
        error_percentage = (error_bpm / ground_truth_tempo) * 100
        within_tolerance = error_bpm <= tolerance_bpm
        
        return {
            'accurate': within_tolerance,
            'error_bpm': float(error_bpm),
            'error_percentage': float(error_percentage),
            'within_tolerance': within_tolerance,
            'predicted_tempo': float(predicted_tempo),
            'ground_truth_tempo': float(ground_truth_tempo),
            'tolerance_bpm': tolerance_bpm
        }
    
    @staticmethod
    def calculate_key_accuracy(predicted_key: str, 
                              ground_truth_key: str) -> Dict[str, any]:
        """
        Calculate key detection accuracy
        
        Args:
            predicted_key: Detected key
            ground_truth_key: Known correct key
            
        Returns:
            Dictionary with accuracy metrics
        """
        if predicted_key is None or ground_truth_key is None:
            return {
                'accurate': False,
                'exact_match': False,
                'enharmonic_match': False
            }
        
        # Exact match
        exact_match = predicted_key.lower() == ground_truth_key.lower()
        
        # Enharmonic equivalents (simplified)
        enharmonic_map = {
            'c# major': 'db major',
            'db major': 'c# major',
            'd# major': 'eb major',
            'eb major': 'd# major',
            'f# major': 'gb major',
            'gb major': 'f# major',
            'g# major': 'ab major',
            'ab major': 'g# major',
            'a# major': 'bb major',
            'bb major': 'a# major',
            'c# minor': 'db minor',
            'db minor': 'c# minor',
            'd# minor': 'eb minor',
            'eb minor': 'd# minor',
            'f# minor': 'gb minor',
            'gb minor': 'f# minor',
            'g# minor': 'ab minor',
            'ab minor': 'g# minor',
            'a# minor': 'bb minor',
            'bb minor': 'a# minor'
        }
        
        enharmonic_equivalent = enharmonic_map.get(predicted_key.lower())
        enharmonic_match = (enharmonic_equivalent and 
                           enharmonic_equivalent == ground_truth_key.lower())
        
        accurate = exact_match or enharmonic_match
        
        return {
            'accurate': accurate,
            'exact_match': exact_match,
            'enharmonic_match': enharmonic_match,
            'predicted_key': predicted_key,
            'ground_truth_key': ground_truth_key
        }
    
    @staticmethod
    def calculate_loudness_accuracy(predicted_lufs: float,
                                  ground_truth_lufs: float,
                                  tolerance_lufs: float = 1.0) -> Dict[str, any]:
        """
        Calculate loudness measurement accuracy
        
        Args:
            predicted_lufs: Detected LUFS value
            ground_truth_lufs: Known correct LUFS value
            tolerance_lufs: Tolerance in LUFS for "correct" measurement
            
        Returns:
            Dictionary with accuracy metrics
        """
        if (predicted_lufs is None or ground_truth_lufs is None or
            np.isneginf(predicted_lufs) or np.isneginf(ground_truth_lufs)):
            return {
                'accurate': False,
                'error_lufs': float('inf'),
                'within_tolerance': False
            }
        
        error_lufs = abs(predicted_lufs - ground_truth_lufs)
        within_tolerance = error_lufs <= tolerance_lufs
        
        return {
            'accurate': within_tolerance,
            'error_lufs': float(error_lufs),
            'within_tolerance': within_tolerance,
            'predicted_lufs': float(predicted_lufs),
            'ground_truth_lufs': float(ground_truth_lufs),
            'tolerance_lufs': tolerance_lufs
        }


class ValidationFramework:
    """
    Complete validation framework for BeatWizard analysis system
    """
    
    def __init__(self, dataset_path: Optional[Union[str, Path]] = None):
        """
        Initialize validation framework
        
        Args:
            dataset_path: Path to validation dataset
        """
        self.dataset = ValidationDataset(dataset_path)
        self.analyzer = EnhancedAudioAnalyzer()
        self.comparison = AnalyzerComparison()
        
        logger.info("ValidationFramework initialized")
    
    def validate_single_track(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Validate analysis accuracy for a single track
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        # Get ground truth
        ground_truth = self.dataset.get_track_ground_truth(file_path)
        if not ground_truth:
            logger.warning(f"No ground truth available for: {file_path.name}")
            return {'error': 'No ground truth available'}
        
        logger.info(f"Validating analysis for: {file_path.name}")
        
        # Perform analysis
        analysis_result = self.analyzer.analyze_track(file_path)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(analysis_result, ground_truth)
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'ground_truth': ground_truth,
            'analysis_result': analysis_result,
            'validation_metrics': validation_metrics,
            'validation_timestamp': time.time()
        }
    
    def validate_dataset(self) -> Dict[str, any]:
        """
        Validate analysis accuracy across entire dataset
        
        Returns:
            Dictionary with comprehensive validation results
        """
        reference_tracks = self.dataset.get_reference_tracks()
        
        if not reference_tracks:
            logger.warning("No reference tracks in dataset")
            return {'error': 'No reference tracks available'}
        
        logger.info(f"Validating dataset with {len(reference_tracks)} tracks")
        
        individual_validations = []
        aggregate_metrics = {
            'tempo_accuracies': [],
            'key_accuracies': [],
            'loudness_accuracies': [],
            'total_tracks': len(reference_tracks),
            'successful_validations': 0
        }
        
        for track_info in reference_tracks:
            try:
                file_path = track_info['file_path']
                validation_result = self.validate_single_track(file_path)
                
                if 'error' not in validation_result:
                    individual_validations.append(validation_result)
                    aggregate_metrics['successful_validations'] += 1
                    
                    # Collect metrics for aggregation
                    metrics = validation_result['validation_metrics']
                    
                    if 'tempo_accuracy' in metrics:
                        aggregate_metrics['tempo_accuracies'].append(metrics['tempo_accuracy'])
                    
                    if 'key_accuracy' in metrics:
                        aggregate_metrics['key_accuracies'].append(metrics['key_accuracy'])
                    
                    if 'loudness_accuracy' in metrics:
                        aggregate_metrics['loudness_accuracies'].append(metrics['loudness_accuracy'])
                
            except Exception as e:
                logger.error(f"Validation failed for {track_info.get('file_name', 'unknown')}: {e}")
                individual_validations.append({
                    'file_path': track_info['file_path'],
                    'error': str(e)
                })
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_validation_stats(aggregate_metrics)
        
        result = {
            'dataset_validation_summary': {
                'total_tracks': aggregate_metrics['total_tracks'],
                'successful_validations': aggregate_metrics['successful_validations'],
                'validation_success_rate': aggregate_metrics['successful_validations'] / aggregate_metrics['total_tracks']
            },
            'aggregate_statistics': aggregate_stats,
            'individual_validations': individual_validations,
            'validation_timestamp': time.time()
        }
        
        logger.info(f"Dataset validation completed: {aggregate_metrics['successful_validations']}/{aggregate_metrics['total_tracks']} successful")
        
        return result
    
    def _calculate_validation_metrics(self, 
                                    analysis_result: Dict[str, any], 
                                    ground_truth: Dict[str, any]) -> Dict[str, any]:
        """Calculate validation metrics comparing analysis to ground truth"""
        metrics = {}
        
        # Tempo validation
        if 'tempo' in ground_truth and 'tempo_analysis' in analysis_result:
            predicted_tempo = analysis_result['tempo_analysis'].get('primary_tempo')
            ground_truth_tempo = ground_truth['tempo']
            
            metrics['tempo_accuracy'] = ValidationMetrics.calculate_tempo_accuracy(
                predicted_tempo, ground_truth_tempo
            )
        
        # Key validation
        if 'key' in ground_truth and 'key_analysis' in analysis_result:
            predicted_key = analysis_result['key_analysis'].get('primary_key')
            ground_truth_key = ground_truth['key']
            
            metrics['key_accuracy'] = ValidationMetrics.calculate_key_accuracy(
                predicted_key, ground_truth_key
            )
        
        # Loudness validation
        if 'lufs' in ground_truth and 'loudness_analysis' in analysis_result:
            predicted_lufs = analysis_result['loudness_analysis'].get('integrated_loudness')
            ground_truth_lufs = ground_truth['lufs']
            
            metrics['loudness_accuracy'] = ValidationMetrics.calculate_loudness_accuracy(
                predicted_lufs, ground_truth_lufs
            )
        
        # Overall accuracy score
        accuracy_scores = []
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'accurate' in metric_data:
                accuracy_scores.append(1.0 if metric_data['accurate'] else 0.0)
        
        if accuracy_scores:
            metrics['overall_accuracy_score'] = sum(accuracy_scores) / len(accuracy_scores)
        else:
            metrics['overall_accuracy_score'] = 0.0
        
        return metrics
    
    def _calculate_aggregate_validation_stats(self, aggregate_metrics: Dict[str, any]) -> Dict[str, any]:
        """Calculate aggregate validation statistics"""
        stats = {}
        
        # Tempo statistics
        tempo_accuracies = aggregate_metrics['tempo_accuracies']
        if tempo_accuracies:
            tempo_accurate_count = sum(1 for acc in tempo_accuracies if acc['accurate'])
            tempo_errors = [acc['error_bpm'] for acc in tempo_accuracies if 'error_bpm' in acc]
            
            stats['tempo_statistics'] = {
                'total_comparisons': len(tempo_accuracies),
                'accuracy_rate': tempo_accurate_count / len(tempo_accuracies),
                'mean_error_bpm': float(np.mean(tempo_errors)) if tempo_errors else 0,
                'std_error_bpm': float(np.std(tempo_errors)) if tempo_errors else 0,
                'max_error_bpm': float(np.max(tempo_errors)) if tempo_errors else 0
            }
        
        # Key statistics
        key_accuracies = aggregate_metrics['key_accuracies']
        if key_accuracies:
            key_accurate_count = sum(1 for acc in key_accuracies if acc['accurate'])
            exact_match_count = sum(1 for acc in key_accuracies if acc['exact_match'])
            
            stats['key_statistics'] = {
                'total_comparisons': len(key_accuracies),
                'accuracy_rate': key_accurate_count / len(key_accuracies),
                'exact_match_rate': exact_match_count / len(key_accuracies)
            }
        
        # Loudness statistics
        loudness_accuracies = aggregate_metrics['loudness_accuracies']
        if loudness_accuracies:
            loudness_accurate_count = sum(1 for acc in loudness_accuracies if acc['accurate'])
            loudness_errors = [acc['error_lufs'] for acc in loudness_accuracies if 'error_lufs' in acc]
            
            stats['loudness_statistics'] = {
                'total_comparisons': len(loudness_accuracies),
                'accuracy_rate': loudness_accurate_count / len(loudness_accuracies),
                'mean_error_lufs': float(np.mean(loudness_errors)) if loudness_errors else 0,
                'std_error_lufs': float(np.std(loudness_errors)) if loudness_errors else 0,
                'max_error_lufs': float(np.max(loudness_errors)) if loudness_errors else 0
            }
        
        # Overall statistics
        all_accuracies = tempo_accuracies + key_accuracies + loudness_accuracies
        if all_accuracies:
            overall_accurate_count = sum(1 for acc in all_accuracies if acc['accurate'])
            stats['overall_statistics'] = {
                'total_measurements': len(all_accuracies),
                'overall_accuracy_rate': overall_accurate_count / len(all_accuracies)
            }
        
        return stats
    
    def create_sample_dataset(self) -> None:
        """Create a sample validation dataset with common test cases"""
        sample_tracks = [
            {
                'filename': 'test_120bpm_cmajor.wav',
                'ground_truth': {
                    'tempo': 120.0,
                    'key': 'C major',
                    'lufs': -14.0,
                    'description': 'Standard test track at 120 BPM in C major'
                }
            },
            {
                'filename': 'test_140bpm_aminor.wav', 
                'ground_truth': {
                    'tempo': 140.0,
                    'key': 'A minor',
                    'lufs': -16.0,
                    'description': 'Electronic track at 140 BPM in A minor'
                }
            },
            {
                'filename': 'test_95bpm_gmajor.wav',
                'ground_truth': {
                    'tempo': 95.0,
                    'key': 'G major',
                    'lufs': -12.0,
                    'description': 'Slower track at 95 BPM in G major'
                }
            }
        ]
        
        logger.info("Creating sample validation dataset")
        
        for track in sample_tracks:
            # Check if files exist in test directory
            test_file_path = project_settings.TEST_TRACKS_DIR / track['filename']
            
            if test_file_path.exists():
                self.dataset.add_reference_track(
                    test_file_path,
                    track['ground_truth'],
                    {'description': track['ground_truth']['description']}
                )
                logger.info(f"Added sample track: {track['filename']}")
            else:
                logger.warning(f"Sample track not found: {track['filename']}")
        
        logger.info("Sample dataset creation completed")
    
    def generate_validation_report(self, 
                                 validation_results: Dict[str, any],
                                 output_path: Union[str, Path]) -> None:
        """
        Generate detailed validation report
        
        Args:
            validation_results: Results from validate_dataset
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("BEATWIZARD VALIDATION REPORT\n")
            f.write("=" * 35 + "\n\n")
            
            # Summary
            summary = validation_results.get('dataset_validation_summary', {})
            f.write("VALIDATION SUMMARY\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total Tracks: {summary.get('total_tracks', 0)}\n")
            f.write(f"Successful Validations: {summary.get('successful_validations', 0)}\n")
            f.write(f"Success Rate: {summary.get('validation_success_rate', 0):.1%}\n\n")
            
            # Aggregate statistics
            stats = validation_results.get('aggregate_statistics', {})
            
            if 'tempo_statistics' in stats:
                tempo_stats = stats['tempo_statistics']
                f.write("TEMPO VALIDATION STATISTICS\n")
                f.write("-" * 28 + "\n")
                f.write(f"Accuracy Rate: {tempo_stats['accuracy_rate']:.1%}\n")
                f.write(f"Mean Error: {tempo_stats['mean_error_bpm']:.2f} BPM\n")
                f.write(f"Max Error: {tempo_stats['max_error_bpm']:.2f} BPM\n\n")
            
            if 'key_statistics' in stats:
                key_stats = stats['key_statistics']
                f.write("KEY VALIDATION STATISTICS\n")
                f.write("-" * 26 + "\n")
                f.write(f"Accuracy Rate: {key_stats['accuracy_rate']:.1%}\n")
                f.write(f"Exact Match Rate: {key_stats['exact_match_rate']:.1%}\n\n")
            
            if 'loudness_statistics' in stats:
                loudness_stats = stats['loudness_statistics']
                f.write("LOUDNESS VALIDATION STATISTICS\n")
                f.write("-" * 31 + "\n")
                f.write(f"Accuracy Rate: {loudness_stats['accuracy_rate']:.1%}\n")
                f.write(f"Mean Error: {loudness_stats['mean_error_lufs']:.2f} LUFS\n")
                f.write(f"Max Error: {loudness_stats['max_error_lufs']:.2f} LUFS\n\n")
            
            if 'overall_statistics' in stats:
                overall_stats = stats['overall_statistics']
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 18 + "\n")
                f.write(f"Total Measurements: {overall_stats['total_measurements']}\n")
                f.write(f"Overall Accuracy: {overall_stats['overall_accuracy_rate']:.1%}\n\n")
            
            # Individual results
            individual_validations = validation_results.get('individual_validations', [])
            if individual_validations:
                f.write("INDIVIDUAL TRACK RESULTS\n")
                f.write("-" * 25 + "\n")
                
                for validation in individual_validations[:10]:  # Limit to first 10
                    if 'error' not in validation:
                        f.write(f"\nFile: {validation['file_name']}\n")
                        
                        metrics = validation['validation_metrics']
                        
                        if 'tempo_accuracy' in metrics:
                            tempo = metrics['tempo_accuracy']
                            f.write(f"  Tempo: {'✓' if tempo['accurate'] else '✗'} ")
                            f.write(f"(Error: {tempo['error_bpm']:.1f} BPM)\n")
                        
                        if 'key_accuracy' in metrics:
                            key = metrics['key_accuracy']
                            f.write(f"  Key: {'✓' if key['accurate'] else '✗'} ")
                            f.write(f"({key['predicted_key']} vs {key['ground_truth_key']})\n")
                        
                        if 'loudness_accuracy' in metrics:
                            loudness = metrics['loudness_accuracy']
                            f.write(f"  Loudness: {'✓' if loudness['accurate'] else '✗'} ")
                            f.write(f"(Error: {loudness['error_lufs']:.2f} LUFS)\n")
                        
                        f.write(f"  Overall Score: {metrics.get('overall_accuracy_score', 0):.1%}\n")
        
        logger.info(f"Validation report generated: {output_path}")
    
    def save_validation_results(self, results: Dict[str, any], output_path: Union[str, Path]) -> None:
        """Save validation results to JSON file"""
        output_path = Path(output_path)
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Validation results saved: {output_path}")
    
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