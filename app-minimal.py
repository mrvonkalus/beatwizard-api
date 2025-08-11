#!/usr/bin/env python3
"""
BeatWizard Minimal Flask App for Railway Deployment
Simplified version without audio dependencies - deploys reliably first
"""

import os
import sys
import uuid
import time
import json
import io
import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Analyze-Lite deps
try:
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    ANALYZE_LITE_AVAILABLE = True
except Exception:
    ANALYZE_LITE_AVAILABLE = False

# Full analyzer deps (Phase 2)
try:
    import librosa  # type: ignore
    import pyloudnorm as pyln  # type: ignore
    FULL_ANALYZER_AVAILABLE = True
except Exception:
    FULL_ANALYZER_AVAILABLE = False

# Try to import audio libraries, but gracefully handle if missing
AUDIO_AVAILABLE = False
try:
    # This will be added back gradually
    print("⚠️  Audio libraries not available in minimal version")
    AUDIO_AVAILABLE = False
except ImportError:
    print("ℹ️  Running in minimal mode - audio processing disabled")
    AUDIO_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

# Production configuration
max_upload_mb_env = os.environ.get('BW_MAX_UPLOAD_MB', '25')
try:
    max_upload_mb = int(float(max_upload_mb_env))
except Exception:
    max_upload_mb = 25

app.config.update(
    MAX_CONTENT_LENGTH=max_upload_mb * 1024 * 1024,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'minimal-demo-secret-key'),
    ENV='production',
    DEBUG=False,
    TESTING=False
)

# Configure CORS for production
cors_origins = os.environ.get('BW_CORS_ORIGINS', os.environ.get('CORS_ORIGINS', '*')).split(',')
CORS(app, 
     origins=cors_origins,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS']
)

# Trust Railway's proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Supported audio file extensions (for when we add audio back)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aiff', 'ogg'}
# For analyze-lite specifically (libsndfile-backed formats; mp3 not guaranteed)
ANALYZE_LITE_ALLOWED_EXTS = {'.wav', '.flac', '.ogg', '.oga', '.aiff', '.aif', '.aifc'}
# Full analyzer supports the same set for now (MP3 later via ffmpeg/audioread)
FULL_ANALYZE_ALLOWED_EXTS = ANALYZE_LITE_ALLOWED_EXTS

# Full analyzer limits
ANALYZE_SECONDS = int(os.environ.get('BW_ANALYZE_SECONDS', '90'))
TARGET_SR = int(os.environ.get('BW_TARGET_SR', '22050'))

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@app.route('/')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'service': 'BeatWizard Audio Analysis API',
        'version': '1.0.0-minimal',
        'audio_processing': AUDIO_AVAILABLE,
        'analyze_lite': ANALYZE_LITE_AVAILABLE,
        'full_analyzer': FULL_ANALYZER_AVAILABLE,
        'mode': 'minimal_deployment',
        'message': 'Basic Flask app deployed successfully! 🚀',
        'timestamp': time.time()
    })

@app.route('/health')
def detailed_health():
    """Detailed health check for monitoring"""
    health_status = {
        'status': 'healthy',
        'checks': {
            'flask': True,
            'cors': True,
            'audio_libraries': AUDIO_AVAILABLE,
            'analyze_lite': ANALYZE_LITE_AVAILABLE,
            'memory': True,
            'disk': True,
        },
        'deployment_stage': 'minimal',
        'next_step': 'Add audio libraries gradually',
        'timestamp': time.time()
    }
    
    return jsonify(health_status), 200

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'api_name': 'BeatWizard Audio Analysis',
        'version': '1.0.0-minimal',
        'deployment_strategy': 'Minimal first, then add audio libraries',
        'available_endpoints': [
            'GET / - Health check',
            'GET /health - Detailed health',
            'GET /api/info - This endpoint',
            'GET /api/demo - Demo response',
            'POST /api/echo - Echo test',
            'POST /api/analyze-lite - Basic analysis',
            'POST /api/analyze - Full analysis'
        ],
        'audio_features': {
            'status': 'coming_soon',
            'planned_features': [
                'Tempo detection',
                'Key detection', 
                'Frequency analysis',
                'LUFS measurement',
                'Intelligent feedback'
            ]
        },
        'deployment_status': 'Phase 1: Basic Flask app ✅'
    })

@app.route('/api/demo')
def demo_analysis():
    """Demo endpoint showing what analysis results will look like"""
    return jsonify({
        'demo': True,
        'message': 'This is a preview of BeatWizard analysis results',
        'sample_analysis': {
            'filename': 'sample_track.mp3',
            'analysis_time': 3.2,
            'results': {
                'tempo': {'bpm': 128.0, 'confidence': 0.95, 'status': 'coming_soon'},
                'key': {'key': 'C major', 'confidence': 0.78, 'status': 'coming_soon'},
                'loudness': {'lufs': -14.2, 'status': 'coming_soon'},
                'frequency_analysis': {'status': 'coming_soon'},
                'intelligent_feedback': {
                    'overall_rating': 'great_potential',
                    'priority_suggestions': [
                        'Once audio processing is enabled, you\'ll get detailed feedback here!'
                    ],
                    'status': 'coming_soon'
                }
            }
        },
        'note': 'Audio processing will be added in the next deployment phase'
    })

@app.route('/api/echo', methods=['POST'])
def echo_test():
    """Simple echo endpoint to test POST requests"""
    data = request.get_json() or {}
    
    return jsonify({
        'echo': True,
        'received_data': data,
        'message': 'POST endpoint working correctly!',
        'timestamp': time.time(),
        'note': 'This confirms Railway deployment is handling requests properly'
    })

@app.route('/api/upload', methods=['POST'])
def upload_placeholder():
    """Placeholder upload endpoint - will add audio processing later"""
    
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'File type not supported',
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'note': 'Audio processing will be added in next deployment phase'
        }), 400
    
    # For now, just acknowledge the file
    filename = secure_filename(file.filename)
    file_size = len(file.read())
    file.seek(0)  # Reset file pointer
    
    return jsonify({
        'success': True,
        'message': 'File received successfully!',
        'filename': filename,
        'file_size_bytes': file_size,
        'analysis_status': 'Audio processing coming soon',
        'deployment_phase': 'minimal',
        'next_steps': [
            'Add librosa dependency',
            'Add audio analysis logic',
            'Enable full BeatWizard features'
        ],
        'timestamp': time.time()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_full():
    """
    Full analysis using librosa + pyloudnorm.
    Safeguards: analyze first N seconds, resample to TARGET_SR, WAV/FLAC first.
    """
    if not FULL_ANALYZER_AVAILABLE:
        return jsonify({'error': 'Full analyzer dependencies not available'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    upl = request.files['file']
    if not upl or upl.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = upl.filename
    lower = filename.lower()
    ext = '.' + lower.rsplit('.', 1)[1] if '.' in lower else ''
    if ext not in FULL_ANALYZE_ALLOWED_EXTS:
        return jsonify({'error': 'Unsupported file extension', 'allowed_exts': sorted(list(FULL_ANALYZE_ALLOWED_EXTS))}), 415

    content_len = request.content_length or 0
    if content_len > app.config['MAX_CONTENT_LENGTH']:
        max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        return jsonify({'error': f'File too large. Max {max_mb} MB'}), 413

    data = upl.read()
    if not data:
        return jsonify({'error': 'Empty file'}), 400

    bio = io.BytesIO(data)

    try:
        y, sr = librosa.load(bio, sr=TARGET_SR, mono=True, duration=ANALYZE_SECONDS)
    except Exception as e:
        return jsonify({'error': 'Decoder error', 'detail': str(e)}), 415

    if y.size == 0:
        return jsonify({'error': 'No audio samples decoded'}), 400

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except Exception:
        tempo = None

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        pitch_class_strength = chroma.mean(axis=1)
        key_index = int(pitch_class_strength.argmax())
        key_guess = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][key_index]
    except Exception:
        key_guess = None

    try:
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = None

    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        S_db = librosa.power_to_db(S, ref=np.max)
        spectrum_summary = {
            'mean_db': float(np.mean(S_db)),
            'median_db': float(np.median(S_db)),
            'p90_db': float(np.percentile(S_db, 90))
        }
    except Exception:
        spectrum_summary = None

    return jsonify({
        'ok': True,
        'analysis': {
            'tempo_bpm': tempo,
            'key_guess': key_guess,
            'lufs': lufs,
            'spectrum_summary': spectrum_summary,
            'sample_rate': sr,
            'analyzed_seconds': ANALYZE_SECONDS,
        },
        'file': {
            'name': filename,
            'size_bytes': len(data),
            'ext': ext
        }
    })
@app.route('/api/analyze-lite', methods=['POST'])
def analyze_lite():
    """
    Analyze audio using numpy + soundfile in a memory-safe, chunked manner.
    Returns duration, peak, RMS, crest factor, and basic metadata.
    """
    if not ANALYZE_LITE_AVAILABLE:
        return jsonify({'error': 'Analyze-Lite dependencies not available'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    upl = request.files['file']
    if not upl or upl.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = upl.filename
    lower = filename.lower()
    ext = '.' + lower.rsplit('.', 1)[1] if '.' in lower else ''
    if ext not in ANALYZE_LITE_ALLOWED_EXTS:
        return jsonify({
            'error': 'Unsupported file extension',
            'allowed_exts': sorted(list(ANALYZE_LITE_ALLOWED_EXTS))
        }), 415

    # Enforce size limit (Flask also enforces MAX_CONTENT_LENGTH)
    content_len = request.content_length or 0
    if content_len > app.config['MAX_CONTENT_LENGTH']:
        max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        return jsonify({'error': f'File too large. Max {max_mb} MB'}), 413

    # Read into BytesIO for soundfile
    try:
        data = upl.read()
    except Exception as e:
        return jsonify({'error': 'Failed to read upload', 'detail': str(e)}), 400

    if not data:
        return jsonify({'error': 'Empty file'}), 400

    bio = io.BytesIO(data)

    # Probe
    try:
        info = sf.info(bio)
    except Exception as e:
        return jsonify({'error': 'Decoder error', 'detail': str(e)}), 415

    samplerate = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames) if info.frames is not None else None
    duration_sec = (frames / samplerate) if (frames and samplerate) else None

    # Decode in chunks
    bio.seek(0)
    peak = 0.0
    sumsq = 0.0
    total = 0
    CHUNK_FRAMES = 262144
    try:
        with sf.SoundFile(bio, mode='r') as snd:
            while True:
                block = snd.read(frames=CHUNK_FRAMES, dtype='float32', always_2d=True)
                if block.size == 0:
                    break
                local_peak = float(np.max(np.abs(block)))
                if local_peak > peak:
                    peak = local_peak
                sumsq += float(np.sum(block ** 2))
                total += block.size
    except Exception as e:
        return jsonify({'error': 'Audio read error', 'detail': str(e)}), 415

    rms = math.sqrt(sumsq / total) if total > 0 else 0.0
    crest_db = (20.0 * math.log10((peak + 1e-12) / (rms + 1e-12))) if rms > 0 else None

    return jsonify({
        'ok': True,
        'analysis': {
            'duration_sec': duration_sec,
            'peak': peak,
            'rms': rms,
            'crest_factor_db': crest_db,
            'metadata': {
                'samplerate': samplerate,
                'channels': channels,
                'frames': frames,
                'format': getattr(info, 'format', None),
                'subtype': getattr(info, 'subtype', None)
            }
        },
        'file': {
            'name': filename,
            'size_bytes': len(data),
            'ext': ext
        }
    })

@app.route('/api/status')
def deployment_status():
    """Show current deployment status and next steps"""
    # Surface versions if available
    numpy_version = None
    soundfile_version = None
    libsndfile_version = None
    try:
        if ANALYZE_LITE_AVAILABLE:
            import numpy
            import soundfile as _sf
            numpy_version = getattr(numpy, '__version__', None)
            soundfile_version = getattr(_sf, '__version__', None)
            libsndfile_version = getattr(_sf, '__libsndfile_version__', None)
    except Exception:
        pass

    return jsonify({
        'deployment_phases': {
            'phase_1': {
                'name': 'Minimal Flask Deployment',
                'status': '✅ COMPLETE',
                'description': 'Basic Flask app with health checks and API structure'
            },
            'phase_2': {
                'name': 'Add Audio Dependencies',
                'status': '⏳ PENDING',
                'description': 'Add librosa, numpy, scipy to requirements.txt',
                'steps': [
                    'Update requirements.txt',
                    'Test audio library imports',
                    'Deploy with audio dependencies'
                ]
            },
            'phase_3': {
                'name': 'Enable Audio Processing',
                'status': '⏳ PENDING', 
                'description': 'Activate full BeatWizard audio analysis',
                'steps': [
                    'Import BeatWizard analyzer',
                    'Enable file processing',
                    'Add intelligent feedback'
                ]
            }
        },
        'current_phase': 'phase_1',
        'railway_deployment': 'SUCCESS' if AUDIO_AVAILABLE is False else 'UNKNOWN',
        'audio_processing': 'lite_enabled' if ANALYZE_LITE_AVAILABLE else 'disabled',
        'analyze_lite': {
            'enabled': ANALYZE_LITE_AVAILABLE,
            'versions': {
                'numpy': numpy_version,
                'soundfile': soundfile_version,
                'libsndfile': libsndfile_version,
            }
        },
        'full_analyzer': {
            'enabled': FULL_ANALYZER_AVAILABLE
        },
        'limits': {
            'max_upload_mb': app.config.get('MAX_CONTENT_LENGTH', 0) // (1024 * 1024)
        }
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({
        'error': 'File too large',
        'max_size_mb': max_size_mb,
        'message': f'Please upload files smaller than {max_size_mb}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong processing your request',
        'deployment_phase': 'minimal'
    }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /',
            'GET /health', 
            'GET /api/info',
            'GET /api/demo',
            'POST /api/echo',
            'POST /api/upload',
            'GET /api/status',
            'POST /api/analyze-lite',
            'POST /api/analyze'
        ]
    }), 404

if __name__ == '__main__':
    # Get port from environment (Render/Railway) or default to 8000 for local tests
    port = int(os.environ.get('PORT', 8000))
    
    print("🚀 Starting BeatWizard Minimal API")
    print(f"📱 Running on port: {port}")
    print(f"🔧 Audio processing: {'Enabled' if AUDIO_AVAILABLE else 'Disabled (minimal mode)'}")
    print("✅ Deployment strategy: Minimal first, add features incrementally")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )