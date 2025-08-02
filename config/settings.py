"""
BeatWizard Configuration Settings
Professional audio analysis configuration with industry standards
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from loguru import logger

class AudioSettings(BaseSettings):
    """Audio processing configuration"""
    
    # Audio File Settings
    SUPPORTED_FORMATS: List[str] = ['.wav', '.mp3', '.flac', '.m4a', '.aiff', '.ogg']
    DEFAULT_SAMPLE_RATE: int = 44100
    DEFAULT_HOP_LENGTH: int = 512
    DEFAULT_FRAME_SIZE: int = 2048
    
    # Analysis Settings
    TEMPO_DETECTION_UNITS: str = 'bpm'
    KEY_DETECTION_ALGORITHM: str = 'krumhansl_schmuckler'
    
    # Frequency Band Analysis (7-band professional EQ)
    FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'high_mid': (2000, 4000),
        'presence': (4000, 8000),
        'brilliance': (8000, 20000)
    }
    
    # Loudness Standards (Professional Mixing)
    LUFS_TARGETS: Dict[str, float] = {
        'streaming': -14.0,
        'cd_master': -9.0,
        'club_mix': -8.0,
        'broadcast': -23.0
    }
    
    # Dynamic Range Targets (dB)
    DYNAMIC_RANGE_TARGETS: Dict[str, Tuple[float, float]] = {
        'excellent': (14.0, float('inf')),
        'good': (9.0, 14.0),
        'average': (6.0, 9.0),
        'compressed': (3.0, 6.0),
        'over_compressed': (0.0, 3.0)
    }

class AISettings(BaseSettings):
    """AI integration configuration"""
    
    OPENAI_API_KEY: Optional[str] = Field(default=None, env='OPENAI_API_KEY')
    OPENAI_MODEL: str = 'gpt-4'
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    
    # Feedback Generation
    GENRE_SPECIFIC_ANALYSIS: bool = True
    INCLUDE_TECHNICAL_DETAILS: bool = True
    PROVIDE_ACTIONABLE_SUGGESTIONS: bool = True

class ProjectSettings(BaseSettings):
    """Project structure and paths"""
    
    # Base Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / 'data'
    OUTPUT_DIR: Path = DATA_DIR / 'output'
    REFERENCE_TRACKS_DIR: Path = DATA_DIR / 'reference_tracks'
    TEST_TRACKS_DIR: Path = DATA_DIR / 'test_tracks'
    
    # Analysis Results
    ANALYSIS_CACHE_DIR: Path = OUTPUT_DIR / 'cache'
    COMPARISON_RESULTS_DIR: Path = OUTPUT_DIR / 'comparisons'
    VALIDATION_RESULTS_DIR: Path = OUTPUT_DIR / 'validation'
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: Optional[str] = 'beatwizard.log'
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

class PerformanceSettings(BaseSettings):
    """Performance optimization settings"""
    
    # Processing
    USE_MULTIPROCESSING: bool = True
    MAX_WORKERS: int = 4
    CHUNK_SIZE: int = 8192
    
    # Memory Management
    MAX_CACHE_SIZE: int = 100  # Number of analysis results to cache
    ENABLE_DISK_CACHE: bool = True
    CACHE_EXPIRY_HOURS: int = 24

# Global Settings Instance
audio_settings = AudioSettings()
ai_settings = AISettings()
project_settings = ProjectSettings()
performance_settings = PerformanceSettings()

def setup_logging():
    """Configure logging for the application"""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level=project_settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File logging (if specified)
    if project_settings.LOG_FILE:
        log_path = project_settings.PROJECT_ROOT / project_settings.LOG_FILE
        logger.add(
            sink=str(log_path),
            level=project_settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )

def create_directories():
    """Create necessary project directories"""
    directories = [
        project_settings.DATA_DIR,
        project_settings.OUTPUT_DIR,
        project_settings.REFERENCE_TRACKS_DIR,
        project_settings.TEST_TRACKS_DIR,
        project_settings.ANALYSIS_CACHE_DIR,
        project_settings.COMPARISON_RESULTS_DIR,
        project_settings.VALIDATION_RESULTS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def validate_environment():
    """Validate environment setup"""
    issues = []
    
    # Check for required environment variables
    if not ai_settings.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not set in environment variables")
    
    # Check directory permissions
    try:
        test_file = project_settings.OUTPUT_DIR / 'test_write.tmp'
        test_file.write_text('test')
        test_file.unlink()
    except PermissionError:
        issues.append(f"No write permission to output directory: {project_settings.OUTPUT_DIR}")
    
    if issues:
        logger.warning(f"Environment validation issues: {issues}")
        return False
    
    logger.info("Environment validation passed")
    return True

# Initialize on import
setup_logging()
create_directories()

logger.info("BeatWizard configuration loaded successfully")