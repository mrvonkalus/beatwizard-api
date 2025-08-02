"""
BeatWizard: Professional AI-Powered Music Analysis
Enhanced audio analysis for music producers

Author: BeatWizard Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "BeatWizard Team"

from .core.enhanced_analyzer import EnhancedAudioAnalyzer
from .core.audio_processor import AudioProcessor
from .analysis.tempo_detector import TempoDetector
from .analysis.key_detector import KeyDetector
from .analysis.frequency_analyzer import FrequencyAnalyzer
from .analysis.loudness_analyzer import LoudnessAnalyzer
from .analysis.stereo_analyzer import StereoAnalyzer

__all__ = [
    'EnhancedAudioAnalyzer',
    'AudioProcessor',
    'TempoDetector',
    'KeyDetector',
    'FrequencyAnalyzer',
    'LoudnessAnalyzer',
    'StereoAnalyzer'
]