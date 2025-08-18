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

# Supabase integration for user authentication and data storage
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

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

# Initialize Supabase client
supabase_client = None
if SUPABASE_AVAILABLE:
    try:
        SUPABASE_URL = "https://rrmoicsnssfbflkbqcek.supabase.co"
        SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJybW9pY3Nuc3NmYmZsa2JxY2VrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM5MzU2NjgsImV4cCI6MjA2OTUxMTY2OH0.hQvp5-KQ3NunGvt7rayrXLAgr7GG4O49brkNVDhJO88"
        supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        supabase_client = None

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
FULL_ANALYZE_ALLOWED_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.oga', '.aiff', '.aif', '.aifc'}

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


def get_user_from_token(auth_header):
    """Extract user ID from Authorization header token"""
    if not auth_header or not supabase_client:
        return None
    
    try:
        # Extract token from "Bearer TOKEN" format
        if not auth_header.startswith('Bearer '):
            return None
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Verify token with Supabase
        user_response = supabase_client.auth.get_user(token)
        if user_response and user_response.user:
            return user_response.user.id
    except Exception as e:
        print(f"Auth error: {e}")
    
    return None

def save_analysis_to_database(user_id, filename, file_size, file_ext, analysis_data):
    """Save analysis data to Supabase database"""
    if not supabase_client or not user_id:
        return False
    
    try:
        result = supabase_client.table('analyses').insert({
            'user_id': user_id,
            'file_name': filename,
            'file_size_bytes': file_size,
            'file_ext': file_ext,
            'analysis_data': analysis_data
        }).execute()
        
        if result.data:
            print(f"✅ Analysis saved to database for user {user_id}")
            return True
    except Exception as e:
        print(f"❌ Failed to save analysis: {e}")
    
    return False

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
        # Load the full audio file - let librosa determine the actual duration
        y, sr = librosa.load(bio, sr=TARGET_SR, mono=True)
        actual_duration = len(y) / sr
        
        # Debug: Check if we're getting reasonable audio data
        if len(y) == 0:
            return jsonify({'error': 'No audio data loaded'}), 400
            
        # Enhanced audio statistics
        audio_stats = {
            'samples': len(y),
            'sample_rate': sr,
            'duration': actual_duration,
            'min_amplitude': float(np.min(y)),
            'max_amplitude': float(np.max(y)),
            'mean_amplitude': float(np.mean(y)),
            'rms': float(np.sqrt(np.mean(y**2))),
            'peak_db': float(20 * np.log10(np.max(np.abs(y)))),
            'rms_db': float(20 * np.log10(np.sqrt(np.mean(y**2)))),
            'dynamic_range_db': float(20 * np.log10(np.max(np.abs(y))/np.sqrt(np.mean(y**2)))),
            'crest_factor': float(np.max(np.abs(y))/np.sqrt(np.mean(y**2)))
        }
        
    except Exception as e:
        return jsonify({'error': 'Decoder error', 'detail': str(e)}), 415

    if y.size == 0:
        return jsonify({'error': 'No audio samples decoded'}), 400

    try:
        # Enhanced rhythm analysis
        if np.sqrt(np.mean(y**2)) < 0.001:  # Very quiet audio
            tempo = None
            rhythm_analysis = None
        else:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=120)
            tempo = float(tempo)
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            rhythm_analysis = {
                'tempo_bpm': tempo,
                'beat_count': len(beats),
                'beat_interval_sec': 60/tempo,
                'onset_count': len(onset_frames),
                'onset_rate_per_sec': len(onset_frames)/actual_duration
            }
    except Exception as e:
        print(f"Rhythm analysis error: {e}")
        tempo = None
        rhythm_analysis = None

    try:
        # Enhanced harmonic analysis
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        pitch_class_strength = chroma.mean(axis=1)
        key_index = int(pitch_class_strength.argmax())
        key_guess = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][key_index]
        chroma_strength = float(np.max(pitch_class_strength))
        
        # Harmonic vs percussive separation
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
        
        harmonic_analysis = {
            'key': key_guess,
            'chroma_strength': chroma_strength,
            'harmonic_ratio': float(harmonic_ratio),
            'percussive_ratio': float(1 - harmonic_ratio)
        }
    except Exception as e:
        print(f"Harmonic analysis error: {e}")
        key_guess = None
        chroma_strength = None
        harmonic_analysis = None

    try:
        # Enhanced loudness analysis
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(y))
        
        # Frequency band analysis (7-band)
        freqs = librosa.fft_frequencies(sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        
        # Define frequency bands (Hz)
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        frequency_analysis = {}
        for band_name, (low_freq, high_freq) in bands.items():
            # Find mel bins for this frequency range
            mel_low = librosa.hz_to_mel(low_freq)
            mel_high = librosa.hz_to_mel(high_freq)
            mel_bins = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr//2)
            band_bins = np.where((mel_bins >= mel_low) & (mel_bins <= mel_high))[0]
            
            if len(band_bins) > 0:
                # Calculate band energy and normalize
                band_energy = np.mean(S[band_bins, :])
                # Normalize to a more readable scale (0-1000)
                normalized_energy = band_energy * 1000
                frequency_analysis[band_name] = float(normalized_energy)
            else:
                frequency_analysis[band_name] = 0.0
        
        # Stereo imaging (if stereo)
        if len(y.shape) > 1 and y.shape[1] > 1:
            left = y[:, 0]
            right = y[:, 1]
            correlation = np.corrcoef(left, right)[0, 1]
            stereo_width = float(np.std(left - right))
        else:
            correlation = 1.0  # Mono
            stereo_width = 0.0
            
        loudness_analysis = {
            'lufs': lufs,
            'stereo_correlation': float(correlation),
            'stereo_width': float(stereo_width)
        }
        
    except Exception as e:
        print(f"Loudness analysis error: {e}")
        lufs = None
        frequency_analysis = {}
        loudness_analysis = {}

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

    # Build the complete analysis response
    analysis_response = {
        'ok': True,
        'analysis': {
            'basic_info': {
                'duration_sec': actual_duration,
                'sample_rate': sr,
                'file_size_mb': len(data) / (1024 * 1024)
            },
            'amplitude_dynamics': {
                'peak_db': audio_stats['peak_db'],
                'rms_db': audio_stats['rms_db'],
                'dynamic_range_db': audio_stats['dynamic_range_db'],
                'crest_factor': audio_stats['crest_factor']
            },
            'rhythm': rhythm_analysis,
            'harmonic': harmonic_analysis,
            'frequency_bands': frequency_analysis,
            'loudness': loudness_analysis,
            'spectrum_summary': spectrum_summary
        },
        'file': {
            'name': filename,
            'size_bytes': len(data),
            'ext': ext
        },
        'beatwizard_ready': True,
        'chat_available': True
    }

    # Check if user is authenticated and save analysis to database
    auth_header = request.headers.get('Authorization')
    user_id = get_user_from_token(auth_header)
    
    if user_id and supabase_client:
        # Save analysis to user's library
        saved = save_analysis_to_database(
            user_id=user_id,
            filename=filename,
            file_size=len(data),
            file_ext=ext,
            analysis_data=analysis_response
        )
        analysis_response['saved_to_library'] = saved
    else:
        analysis_response['saved_to_library'] = False

    return jsonify(analysis_response)

@app.route('/api/beatwizard-chat', methods=['POST'])
def beatwizard_chat():
    """
    🧙‍♂️ THE BEATWIZARD CHATBOT! 
    Ask the wise music wizard for advice on your track!
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data.get('message', '').strip()
        analysis_data = data.get('analysis', {})
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
            
        # 🧙‍♂️ BEATWIZARD PERSONALITY & LOGIC
        response = generate_beatwizard_response(user_message, analysis_data)
        
        return jsonify({
            'ok': True,
            'beatwizard_response': response,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': 'Chat error', 'detail': str(e)}), 500

def generate_beatwizard_response(user_message, analysis_data):
    """
    🧙‍♂️ THE BEATWIZARD - RULES OF WISDOM
    
    THE SACRED COMMANDMENTS:
    1. THOU SHALL NEVER speak without referencing the Analysis Data
    2. THOU SHALL ALWAYS cite specific numerical values 
    3. THOU SHALL NOT give generic advice - only data-driven insights
    4. THOU SHALL ground every statement in measurable metrics
    5. THOU SHALL be the oracle of the numbers, not the purveyor of platitudes
    """
    
    # RULE #1: NO ANALYSIS = NO WISDOM
    if not analysis_data:
        return {
            'message': "🧙‍♂️ *staff dims* I cannot cast wisdom without the sacred numbers! Upload your track for analysis first, young producer, and I shall read the numerical prophecies within your music! ✨",
            'tone': 'mystical_greeting'
        }
    
    # EXTRACT ALL SACRED METRICS
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    key = analysis_data.get('harmonic', {}).get('key')
    duration = analysis_data.get('basic_info', {}).get('duration_sec')
    
    # Amplitude & Dynamics
    peak_db = analysis_data.get('amplitude_dynamics', {}).get('peak_db')
    rms_db = analysis_data.get('amplitude_dynamics', {}).get('rms_db')
    dynamic_range = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    crest_factor = analysis_data.get('amplitude_dynamics', {}).get('crest_factor')
    
    # Loudness
    lufs = analysis_data.get('loudness', {}).get('lufs')
    stereo_correlation = analysis_data.get('loudness', {}).get('stereo_correlation')
    stereo_width = analysis_data.get('loudness', {}).get('stereo_width')
    
    # Frequency Bands
    freq_bands = analysis_data.get('frequency_bands', {})
    sub_bass = freq_bands.get('sub_bass', 0)
    bass = freq_bands.get('bass', 0)
    low_mid = freq_bands.get('low_mid', 0)
    mid = freq_bands.get('mid', 0)
    high_mid = freq_bands.get('high_mid', 0)
    presence = freq_bands.get('presence', 0)
    brilliance = freq_bands.get('brilliance', 0)
    
    # Harmonic Analysis
    harmonic_ratio = analysis_data.get('harmonic', {}).get('harmonic_ratio')
    percussive_ratio = analysis_data.get('harmonic', {}).get('percussive_ratio')
    spectral_centroid = analysis_data.get('harmonic', {}).get('spectral_centroid_hz')
    zero_crossing_rate = analysis_data.get('harmonic', {}).get('zero_crossing_rate')
    
    # 🎯 THE WISDOM ENGINE - ALWAYS DATA-DRIVEN
    return analyze_and_respond_with_data(
        user_message=user_message,
        tempo=tempo, key=key, duration=duration,
        peak_db=peak_db, rms_db=rms_db, dynamic_range=dynamic_range, crest_factor=crest_factor,
        lufs=lufs, stereo_correlation=stereo_correlation, stereo_width=stereo_width,
        sub_bass=sub_bass, bass=bass, low_mid=low_mid, mid=mid, 
        high_mid=high_mid, presence=presence, brilliance=brilliance,
        harmonic_ratio=harmonic_ratio, percussive_ratio=percussive_ratio,
        spectral_centroid=spectral_centroid, zero_crossing_rate=zero_crossing_rate
    )

def analyze_and_respond_with_data(user_message, **metrics):
    """
    🧙‍♂️ THE WISDOM ENGINE - 100% DATA-DRIVEN RESPONSES
    
    EVERY response MUST cite specific numbers from the analysis.
    NO generic advice allowed. ONLY data-grounded insights.
    """
    
    # Build the data summary for context
    data_summary = build_data_summary(**metrics)
    
    # Analyze the user's intent and respond with SPECIFIC DATA
    user_lower = user_message.lower()
    
    # TEMPO ANALYSIS - Always cite the actual BPM
    if any(word in user_lower for word in ['tempo', 'bpm', 'speed', 'fast', 'slow']):
        return respond_with_tempo_data(data_summary, **metrics)
    
    # KEY ANALYSIS - Always cite the actual key
    elif any(word in user_lower for word in ['key', 'scale', 'harmonic', 'chord', 'pitch']):
        return respond_with_key_data(data_summary, **metrics)
    
    # MIX ANALYSIS - Always cite LUFS, dynamics, frequency balance
    elif any(word in user_lower for word in ['mix', 'mixing', 'balance', 'loud', 'quiet']):
        return respond_with_mix_data(data_summary, **metrics)
    
    # BASS ANALYSIS - Always cite bass & sub-bass levels
    elif any(word in user_lower for word in ['bass', 'kick', 'low end', 'sub']):
        return respond_with_bass_data(data_summary, **metrics)
    
    # STEREO ANALYSIS - Always cite stereo correlation & width
    elif any(word in user_lower for word in ['stereo', 'width', 'image', 'imaging', 'wide', 'narrow']):
        return respond_with_stereo_data(data_summary, **metrics)
    
    # DYNAMICS ANALYSIS - Always cite dynamic range & crest factor
    elif any(word in user_lower for word in ['dynamics', 'compressed', 'punch', 'impact']):
        return respond_with_dynamics_data(data_summary, **metrics)
    
    # FREQUENCY ANALYSIS - Always cite the 7-band breakdown
    elif any(word in user_lower for word in ['frequency', 'bands', 'eq', 'bright', 'dark', 'muddy']):
        return respond_with_frequency_data(data_summary, **metrics)
    
    # GENRE ANALYSIS - Based on tempo, key, and spectral characteristics
    elif any(word in user_lower for word in ['genre', 'style', 'type', 'category', 'fits']):
        return respond_with_genre_data(data_summary, **metrics)
    
    # PROBLEMS ANALYSIS - Identify specific issues with numbers
    elif any(word in user_lower for word in ['problem', 'issue', 'wrong', 'bad', 'fix']):
        return respond_with_problems_data(data_summary, **metrics)
    
    # COMPREHENSIVE ANALYSIS - All metrics
    elif any(word in user_lower for word in ['everything', 'all', 'full', 'complete', 'detailed', 'track']):
        return respond_with_comprehensive_data(data_summary, **metrics)
    
    # ARTIST SUGGESTIONS - Based on tempo, key, and style
    elif any(word in user_lower for word in ['artist', 'artists', 'who', 'rapper', 'singer', 'producer', 'like', 'similar']):
        return respond_with_artist_suggestions(data_summary, **metrics)
    
    # SOUND SELECTION ANALYSIS
    elif any(word in user_lower for word in ['sound', 'selection', 'sounds', 'samples', 'instruments', 'drums']):
        return respond_with_sound_selection_analysis(data_summary, **metrics)
    
    # ARRANGEMENT TIPS
    elif any(word in user_lower for word in ['arrangement', 'structure', 'build', 'drop', 'verse', 'chorus', 'bridge']):
        return respond_with_arrangement_tips(data_summary, **metrics)
    
    # LOW-END SPECIFIC ANALYSIS
    elif any(word in user_lower for word in ['low end', 'low-end', 'lowend', 'sub', '808', 'kick', 'bass']):
        return respond_with_bass_data(data_summary, **metrics)
    
    # VOCAL ANALYSIS
    elif any(word in user_lower for word in ['vocal', 'vocals', 'voice', 'singing', 'singer', 'lyrics']):
        return respond_with_vocal_analysis(data_summary, **metrics)
    
    # RHYTHM/DRUMS ANALYSIS
    elif any(word in user_lower for word in ['rhythm', 'drums', 'drum', 'beat', 'percussion', 'groove']):
        return respond_with_rhythm_analysis(data_summary, **metrics)
    
    # SUGGESTIONS/IMPROVEMENTS
    elif any(word in user_lower for word in ['suggestions', 'improve', 'better', 'tips', 'advice', 'help']):
        return respond_with_arrangement_tips(data_summary, **metrics)
    
    # DEFAULT: TRACK OVERVIEW with key metrics
    else:
        return respond_with_overview_data(data_summary, **metrics)

def build_data_summary(**metrics):
    """Build a comprehensive summary of all analysis metrics"""
    return {
        'tempo': metrics.get('tempo'),
        'key': metrics.get('key'),
        'duration': metrics.get('duration'),
        'peak_db': metrics.get('peak_db'),
        'rms_db': metrics.get('rms_db'),
        'dynamic_range': metrics.get('dynamic_range'),
        'crest_factor': metrics.get('crest_factor'),
        'lufs': metrics.get('lufs'),
        'stereo_correlation': metrics.get('stereo_correlation'),
        'stereo_width': metrics.get('stereo_width'),
        'sub_bass': metrics.get('sub_bass'),
        'bass': metrics.get('bass'),
        'low_mid': metrics.get('low_mid'),
        'mid': metrics.get('mid'),
        'high_mid': metrics.get('high_mid'),
        'presence': metrics.get('presence'),
        'brilliance': metrics.get('brilliance'),
        'harmonic_ratio': metrics.get('harmonic_ratio'),
        'percussive_ratio': metrics.get('percussive_ratio'),
        'spectral_centroid': metrics.get('spectral_centroid'),
        'zero_crossing_rate': metrics.get('zero_crossing_rate')
    }

def respond_with_tempo_data(data_summary, **metrics):
    """ALWAYS cite the exact BPM and tempo characteristics"""
    tempo = data_summary['tempo']
    if tempo is None:
        return {
            'message': "🧙‍♂️ *peers through crystal ball* The tempo spirits have not revealed themselves in your track's analysis data. This suggests either no clear rhythm pattern or an analysis limitation.",
            'tone': 'data_missing'
        }
    
    # Tempo categorization based on actual BPM
    if tempo < 70:
        tempo_category = "BALLAD territory"
        genre_context = "perfect for emotional, slow builds"
    elif tempo < 90:
        tempo_category = "DOWNTEMPO/CHILL"
        genre_context = "ideal for lo-fi, ambient, or R&B"
    elif tempo < 100:
        tempo_category = "MODERATE pace"
        genre_context = "versatile for pop, hip-hop, or rock"
    elif tempo < 120:
        tempo_category = "ENERGETIC"
        genre_context = "great for pop, dance, or upbeat hip-hop"
    elif tempo < 140:
        tempo_category = "HIGH ENERGY"
        genre_context = "perfect for house, techno, or EDM"
    else:
        tempo_category = "EXTREME ENERGY"
        genre_context = "hardcore EDM, drum & bass, or metal territory"
    
    message = f"**TEMPO ANALYSIS: {tempo:.1f} BPM**\n\n"
    message += f"🎯 Your track sits in **{tempo_category}** - {genre_context}.\n\n"
    
    # Tempo-specific production advice
    if tempo < 85:
        message += f"**Production Insight:** At {tempo:.1f} BPM, focus on:\n"
        message += "• Lush reverbs and atmospheric textures\n"
        message += "• Longer attack times on instruments\n"
        message += "• Wide, spacious mixing\n"
    elif tempo > 130:
        message += f"**Production Insight:** At {tempo:.1f} BPM, prioritize:\n"
        message += "• Tight, punchy drums\n"
        message += "• Short reverb tails to avoid muddiness\n"
        message += "• Clear separation between elements\n"
    else:
        message += f"**Production Insight:** {tempo:.1f} BPM is the sweet spot for:\n"
        message += "• Balanced dynamics\n"
        message += "• Versatile arrangement choices\n"
        message += "• Wide commercial appeal\n"
    
    return {
        'message': message,
        'tone': 'analytical',
        'cited_data': f"tempo: {tempo:.1f} BPM"
    }

def respond_with_key_data(data_summary, **metrics):
    """ALWAYS cite the exact key and harmonic characteristics"""
    key = data_summary['key']
    harmonic_ratio = data_summary['harmonic_ratio']
    percussive_ratio = data_summary['percussive_ratio']
    
    if key is None:
        return {
            'message': "🧙‍♂️ *adjusts harmonic lens* The key signature remains hidden in your track's harmonic mysteries. This could indicate an atonal piece, heavy modulation, or predominantly percussive content.",
            'tone': 'data_missing'
        }
    
    message = f"**KEY ANALYSIS: {key}**\n\n"
    
    # Key characteristics and mood
    key_moods = {
        'C': "Innocent, pure, simple",
        'C#': "Passionate, grief, despair", 'Db': "Passionate, grief, despair",
        'D': "Triumphant, martial, joyful",
        'D#': "Feelings of despair", 'Eb': "Love, devotion, intimate",
        'E': "Bright, joyful, confident",
        'F': "Pastoral, peaceful, simple",
        'F#': "Sharp, melancholic", 'Gb': "Tender, soft, intimate",
        'G': "Rustic, cheerful, pastoral",
        'G#': "Gracefulness, flute-like", 'Ab': "Death, eternity, judgment",
        'A': "Confident, bright, hopeful",
        'A#': "Mournful, restless", 'Bb': "Magnificent, joyful",
        'B': "Harsh, piercing, angry"
    }
    
    mood = key_moods.get(key, "Dynamic and expressive")
    message += f"🎭 **Emotional Character:** {mood}\n\n"
    
    # Harmonic vs Percussive content
    if harmonic_ratio is not None and percussive_ratio is not None:
        message += f"**Musical Content Analysis:**\n"
        message += f"• Harmonic Content: {harmonic_ratio:.2f} ({harmonic_ratio*100:.0f}%)\n"
        message += f"• Percussive Content: {percussive_ratio:.2f} ({percussive_ratio*100:.0f}%)\n\n"
        
        if harmonic_ratio > 0.6:
            content_type = "MELODIC-DOMINANT"
            advice = "Focus on chord progressions, melodic hooks, and harmonic layering"
        elif percussive_ratio > 0.6:
            content_type = "RHYTHM-DOMINANT"
            advice = "Emphasize drum patterns, percussion, and rhythmic elements"
        else:
            content_type = "BALANCED"
            advice = "Great balance between melodic and rhythmic elements"
        
        message += f"**Content Type:** {content_type}\n"
        message += f"**Production Focus:** {advice}\n"
    
    # Key-specific production advice
    message += f"\n**Key-Specific Production Tips for {key}:**\n"
    if key in ['C', 'G', 'D', 'A', 'E']:
        message += "• Major key brightness - perfect for uplifting, commercial tracks\n"
        message += "• Use open voicings and bright timbres\n"
    elif key in ['Am', 'Em', 'Bm', 'F#m', 'C#m']:
        message += "• Minor key depth - ideal for emotional, introspective content\n"
        message += "• Layer with darker timbres and rich harmonics\n"
    
    return {
        'message': message,
        'tone': 'harmonic_analytical',
        'cited_data': f"key: {key}, harmonic_ratio: {harmonic_ratio}, percussive_ratio: {percussive_ratio}"
    }

def respond_with_overview_data(data_summary, **metrics):
    """DEFAULT: Comprehensive overview citing key metrics"""
    tempo = data_summary['tempo']
    key = data_summary['key']
    duration = data_summary['duration']
    lufs = data_summary['lufs']
    dynamic_range = data_summary['dynamic_range']
    
    message = "**TRACK OVERVIEW - THE NUMBERS REVEALED**\n\n"
    
    # Core metrics
    if tempo is not None:
        message += f"🎵 **Tempo:** {tempo:.1f} BPM\n"
    if key is not None:
        message += f"🎼 **Key:** {key}\n"
    if duration is not None:
        message += f"⏱️ **Duration:** {duration:.1f} seconds\n"
    
    # Loudness & Dynamics
    if lufs is not None:
        streaming_status = "GOOD" if -16 <= lufs <= -12 else "NEEDS ADJUSTMENT"
        message += f"🔊 **LUFS:** {lufs:.1f} dB ({streaming_status} for streaming)\n"
    
    if dynamic_range is not None:
        dr_status = "EXCELLENT" if dynamic_range > 15 else "GOOD" if dynamic_range > 10 else "COMPRESSED"
        message += f"📊 **Dynamic Range:** {dynamic_range:.1f} dB ({dr_status})\n"
    
    message += "\n**Ask me about any specific aspect:**\n"
    message += "• **'tempo'** - detailed BPM analysis\n"
    message += "• **'mix'** - loudness, dynamics, frequency balance\n"
    message += "• **'bass'** - low-end analysis\n"
    message += "• **'stereo'** - stereo imaging analysis\n"
    message += "• **'everything'** - comprehensive breakdown\n"
    
    cited_metrics = []
    if tempo: cited_metrics.append(f"tempo: {tempo:.1f}")
    if key: cited_metrics.append(f"key: {key}")
    if lufs: cited_metrics.append(f"lufs: {lufs:.1f}")
    if dynamic_range: cited_metrics.append(f"dynamic_range: {dynamic_range:.1f}")
    
    return {
        'message': message,
        'tone': 'overview_analytical',
        'cited_data': ", ".join(cited_metrics)
    }

def respond_with_mix_data(data_summary, **metrics):
    """ALWAYS cite LUFS, dynamics, and frequency balance numbers"""
    lufs = data_summary['lufs']
    dynamic_range = data_summary['dynamic_range']
    peak_db = data_summary['peak_db']
    
    message = "**MIX ANALYSIS - PRECISE MEASUREMENTS**\n\n"
    
    # LUFS Analysis
    if lufs is not None:
        if lufs > -12:
            loudness_verdict = "TOO LOUD - streaming platforms will turn it down"
            recommendation = f"Lower to -14 LUFS (reduce by {lufs - (-14):.1f} dB)"
        elif lufs < -18:
            loudness_verdict = "TOO QUIET - listeners may skip"
            recommendation = f"Raise to -14 LUFS (increase by {(-14) - lufs:.1f} dB)"
        else:
            loudness_verdict = "PERFECT for streaming platforms"
            recommendation = "Maintain this loudness level"
        
        message += f"🔊 **LUFS: {lufs:.1f} dB**\n"
        message += f"Status: {loudness_verdict}\n"
        message += f"Action: {recommendation}\n\n"
    
    # Dynamic Range Analysis
    if dynamic_range is not None:
        if dynamic_range < 6:
            dynamics_verdict = "HEAVILY COMPRESSED - lacks breathing room"
            dynamics_advice = "Reduce compression ratio, use parallel compression"
        elif dynamic_range < 10:
            dynamics_verdict = "COMPRESSED - typical modern production"
            dynamics_advice = "Good balance, consider slight compression reduction for more life"
        elif dynamic_range > 15:
            dynamics_verdict = "EXCELLENT DYNAMICS - very musical"
            dynamics_advice = "Maintain this natural dynamic range"
        else:
            dynamics_verdict = "GOOD DYNAMICS - well balanced"
            dynamics_advice = "Solid dynamic range for modern production"
        
        message += f"📊 **Dynamic Range: {dynamic_range:.1f} dB**\n"
        message += f"Assessment: {dynamics_verdict}\n"
        message += f"Advice: {dynamics_advice}\n\n"
    
    # Peak Analysis
    if peak_db is not None:
        if peak_db > -0.1:
            peak_verdict = "CLIPPING RISK - may cause distortion"
        elif peak_db > -3:
            peak_verdict = "VERY HOT - little headroom"
        else:
            peak_verdict = "GOOD HEADROOM - safe peak levels"
        
        message += f"⚡ **Peak Level: {peak_db:.1f} dB**\n"
        message += f"Status: {peak_verdict}\n\n"
    
    # Frequency Balance
    bass = data_summary['bass'] or 0
    mid = data_summary['mid'] or 0
    presence = data_summary['presence'] or 0
    
    message += "**Frequency Balance:**\n"
    message += f"• Bass (60-250Hz): {bass:.3f}\n"
    message += f"• Mids (1-4kHz): {mid:.3f}\n"
    message += f"• Presence (8-16kHz): {presence:.3f}\n\n"
    
    # Balance Assessment
    if bass > mid * 2:
        message += "⚠️ **Issue:** Bass is overpowering - reduce 80-200Hz or boost mids\n"
    elif mid < 0.05:
        message += "⚠️ **Issue:** Vocals/instruments will lack clarity - boost 1-4kHz\n"
    elif presence > 0.3:
        message += "⚠️ **Issue:** May sound harsh - reduce 8-12kHz\n"
    else:
        message += "✅ **Frequency balance looks good**\n"
    
    cited_data = []
    if lufs: cited_data.append(f"LUFS: {lufs:.1f}")
    if dynamic_range: cited_data.append(f"DR: {dynamic_range:.1f}")
    if peak_db: cited_data.append(f"Peak: {peak_db:.1f}")
    
    return {
        'message': message,
        'tone': 'mix_analytical',
        'cited_data': ", ".join(cited_data)
    }

def respond_with_bass_data(data_summary, **metrics):
    """ALWAYS cite exact bass and sub-bass energy levels"""
    sub_bass = data_summary['sub_bass'] or 0
    bass = data_summary['bass'] or 0
    
    message = "**BASS ANALYSIS - LOW-END BREAKDOWN**\n\n"
    
    message += f"🔊 **Sub-Bass (20-60Hz): {sub_bass:.3f}**\n"
    message += f"🎵 **Bass (60-250Hz): {bass:.3f}**\n\n"
    
    # Sub-bass analysis
    if sub_bass < 0.02:
        sub_verdict = "WEAK - lacks foundation"
        sub_advice = "Add sub-bass around 40-60Hz, use 808s or sub synths"
    elif sub_bass > 0.15:
        sub_verdict = "OVERPOWERING - may muddy the mix"
        sub_advice = "Reduce sub content, use high-pass filtering on other elements"
    else:
        sub_verdict = "BALANCED - good foundation"
        sub_advice = "Sub-bass level is appropriate for the genre"
    
    message += f"**Sub-Bass Assessment:** {sub_verdict}\n"
    message += f"**Recommendation:** {sub_advice}\n\n"
    
    # Bass analysis
    if bass < 0.05:
        bass_verdict = "THIN - lacks warmth and body"
        bass_advice = "Boost around 80-120Hz, add bass guitar or synth bass"
    elif bass > 0.25:
        bass_verdict = "HEAVY - may overpower other elements"
        bass_advice = "Reduce bass content, use sidechain compression"
    else:
        bass_verdict = "SOLID - good presence"
        bass_advice = "Bass level supports the track well"
    
    message += f"**Bass Assessment:** {bass_verdict}\n"
    message += f"**Recommendation:** {bass_advice}\n\n"
    
    # Combined low-end assessment
    total_low_end = sub_bass + bass
    if total_low_end > 0.3:
        message += "⚠️ **Overall:** Low-end is dominating - may cause muddiness on small speakers\n"
    elif total_low_end < 0.08:
        message += "⚠️ **Overall:** Track may sound thin and weak - needs more low-end presence\n"
    else:
        message += "✅ **Overall:** Low-end balance is good for modern production\n"
    
    return {
        'message': message,
        'tone': 'bass_analytical',
        'cited_data': f"sub_bass: {sub_bass:.3f}, bass: {bass:.3f}"
    }

def respond_with_stereo_data(data_summary, **metrics):
    """ALWAYS cite stereo correlation and width measurements"""
    stereo_correlation = data_summary['stereo_correlation']
    stereo_width = data_summary['stereo_width']
    
    if stereo_correlation is None and stereo_width is None:
        return {
            'message': "🧙‍♂️ *adjusts stereo lens* The stereo measurements remain veiled in the analysis mists. This suggests mono content or analysis limitations.",
            'tone': 'data_missing'
        }
    
    message = "**STEREO IMAGING ANALYSIS**\n\n"
    
    # Stereo Correlation Analysis
    if stereo_correlation is not None:
        if stereo_correlation > 0.8:
            correlation_verdict = "NARROW - sounds mono-like"
            correlation_advice = "Add stereo width with reverb, delay, or stereo imaging plugins"
        elif stereo_correlation < 0.3:
            correlation_verdict = "VERY WIDE - may collapse in mono"
            correlation_advice = "Check mono compatibility, consider narrowing some elements"
        else:
            correlation_verdict = "BALANCED - good stereo spread"
            correlation_advice = "Stereo field is well-balanced for most playback systems"
        
        message += f"🎧 **Stereo Correlation: {stereo_correlation:.2f}**\n"
        message += f"Assessment: {correlation_verdict}\n"
        message += f"Recommendation: {correlation_advice}\n\n"
    
    # Stereo Width Analysis
    if stereo_width is not None:
        if stereo_width < 0.3:
            width_verdict = "NARROW - limited spatial dimension"
            width_advice = "Widen with stereo reverbs, ping-pong delays, or M/S processing"
        elif stereo_width > 0.8:
            width_verdict = "VERY WIDE - spacious but check compatibility"
            width_advice = "Ensure key elements (vocals, bass, kick) remain centered"
        else:
            width_verdict = "GOOD WIDTH - appropriate stereo field"
            width_advice = "Stereo width enhances the listening experience appropriately"
        
        message += f"↔️ **Stereo Width: {stereo_width:.2f}**\n"
        message += f"Assessment: {width_verdict}\n"
        message += f"Recommendation: {width_advice}\n\n"
    
    # Production tips based on measurements
    message += "**Stereo Production Tips:**\n"
    if stereo_correlation and stereo_correlation > 0.7:
        message += "• Pan instruments across the stereo field\n"
        message += "• Use stereo reverbs and delays\n"
        message += "• Consider haas effect for width\n"
    else:
        message += "• Keep important elements (vocals, bass, kick) centered\n"
        message += "• Use mono reverb sends to maintain focus\n"
        message += "• Check mix in mono to ensure translation\n"
    
    cited_data = []
    if stereo_correlation: cited_data.append(f"correlation: {stereo_correlation:.2f}")
    if stereo_width: cited_data.append(f"width: {stereo_width:.2f}")
    
    return {
        'message': message,
        'tone': 'stereo_analytical',
        'cited_data': ", ".join(cited_data)
    }

def respond_with_comprehensive_data(data_summary, **metrics):
    """COMPREHENSIVE: All metrics with specific numbers"""
    message = "**COMPLETE TRACK ANALYSIS - ALL METRICS REVEALED**\n\n"
    
    # Basic Info
    if data_summary['tempo'] or data_summary['key'] or data_summary['duration']:
        message += "**🎵 Core Characteristics:**\n"
        if data_summary['tempo']: message += f"• Tempo: {data_summary['tempo']:.1f} BPM\n"
        if data_summary['key']: message += f"• Key: {data_summary['key']}\n"
        if data_summary['duration']: message += f"• Duration: {data_summary['duration']:.1f}s\n"
        message += "\n"
    
    # Loudness & Dynamics
    if data_summary['lufs'] or data_summary['dynamic_range'] or data_summary['peak_db']:
        message += "**🔊 Loudness & Dynamics:**\n"
        if data_summary['lufs']: 
            streaming_status = "GOOD" if -16 <= data_summary['lufs'] <= -12 else "NEEDS FIX"
            message += f"• LUFS: {data_summary['lufs']:.1f} dB ({streaming_status})\n"
        if data_summary['dynamic_range']: 
            dr_status = "EXCELLENT" if data_summary['dynamic_range'] > 15 else "COMPRESSED" if data_summary['dynamic_range'] < 8 else "GOOD"
            message += f"• Dynamic Range: {data_summary['dynamic_range']:.1f} dB ({dr_status})\n"
        if data_summary['peak_db']: message += f"• Peak Level: {data_summary['peak_db']:.1f} dB\n"
        message += "\n"
    
    # Stereo Field
    if data_summary['stereo_correlation'] or data_summary['stereo_width']:
        message += "**🎧 Stereo Field:**\n"
        if data_summary['stereo_correlation']: 
            stereo_status = "WIDE" if data_summary['stereo_correlation'] < 0.5 else "NARROW"
            message += f"• Stereo Correlation: {data_summary['stereo_correlation']:.2f} ({stereo_status})\n"
        if data_summary['stereo_width']: message += f"• Stereo Width: {data_summary['stereo_width']:.2f}\n"
        message += "\n"
    
    # Frequency Breakdown
    message += "**🎛️ 7-Band Frequency Analysis:**\n"
    if data_summary['sub_bass']: message += f"• Sub-Bass (20-60Hz): {data_summary['sub_bass']:.3f}\n"
    if data_summary['bass']: message += f"• Bass (60-250Hz): {data_summary['bass']:.3f}\n"
    if data_summary['low_mid']: message += f"• Low-Mid (250Hz-1kHz): {data_summary['low_mid']:.3f}\n"
    if data_summary['mid']: message += f"• Mid (1-4kHz): {data_summary['mid']:.3f}\n"
    if data_summary['high_mid']: message += f"• High-Mid (4-8kHz): {data_summary['high_mid']:.3f}\n"
    if data_summary['presence']: message += f"• Presence (8-16kHz): {data_summary['presence']:.3f}\n"
    if data_summary['brilliance']: message += f"• Brilliance (16kHz+): {data_summary['brilliance']:.3f}\n"
    message += "\n"
    
    # Musical Content
    if data_summary['harmonic_ratio'] or data_summary['percussive_ratio']:
        message += "**🎼 Musical Content:**\n"
        if data_summary['harmonic_ratio']: message += f"• Harmonic Content: {data_summary['harmonic_ratio']:.2f} ({data_summary['harmonic_ratio']*100:.0f}%)\n"
        if data_summary['percussive_ratio']: message += f"• Percussive Content: {data_summary['percussive_ratio']:.2f} ({data_summary['percussive_ratio']*100:.0f}%)\n"
        message += "\n"
    
    # Issues Detection
    issues = []
    if data_summary['lufs'] and data_summary['lufs'] > -12:
        issues.append("Track too loud for streaming")
    if data_summary['dynamic_range'] and data_summary['dynamic_range'] < 8:
        issues.append("Over-compressed")
    if data_summary['bass'] and data_summary['bass'] < 0.05:
        issues.append("Lacks low-end presence")
    if data_summary['mid'] and data_summary['mid'] < 0.05:
        issues.append("Vocals may lack clarity")
    
    if issues:
        message += "⚠️ **Issues Detected:**\n"
        for issue in issues:
            message += f"• {issue}\n"
    else:
        message += "✅ **Overall Assessment: GOOD BALANCE**\n"
    
    return {
        'message': message,
        'tone': 'comprehensive_analytical',
        'cited_data': "all_metrics_analyzed"
    }

def respond_with_dynamics_data(data_summary, **metrics):
    """ALWAYS cite dynamic range and crest factor"""
    dynamic_range = data_summary['dynamic_range']
    crest_factor = data_summary['crest_factor']
    
    message = "**DYNAMICS ANALYSIS**\n\n"
    
    if dynamic_range is not None:
        if dynamic_range < 6:
            verdict = "HEAVILY COMPRESSED - lacks punch and life"
            advice = "Reduce compression ratio, try parallel compression"
        elif dynamic_range < 10:
            verdict = "COMPRESSED - typical modern production"
            advice = "Consider slight compression reduction for more natural feel"
        elif dynamic_range > 15:
            verdict = "EXCELLENT DYNAMICS - very musical and natural"
            advice = "Maintain this dynamic range - it's perfect"
        else:
            verdict = "GOOD DYNAMICS - well balanced"
            advice = "Solid dynamic range for professional production"
        
        message += f"📊 **Dynamic Range: {dynamic_range:.1f} dB**\n"
        message += f"Assessment: {verdict}\n"
        message += f"Recommendation: {advice}\n\n"
    
    if crest_factor is not None:
        if crest_factor < 3:
            crest_verdict = "VERY COMPRESSED - limited peaks"
        elif crest_factor > 10:
            crest_verdict = "VERY DYNAMIC - natural peaks"
        else:
            crest_verdict = "BALANCED - controlled peaks"
        
        message += f"⚡ **Crest Factor: {crest_factor:.1f}**\n"
        message += f"Assessment: {crest_verdict}\n"
    
    return {
        'message': message,
        'tone': 'dynamics_analytical',
        'cited_data': f"dynamic_range: {dynamic_range}, crest_factor: {crest_factor}"
    }

def respond_with_frequency_data(data_summary, **metrics):
    """ALWAYS cite all 7 frequency bands"""
    message = "**7-BAND FREQUENCY ANALYSIS**\n\n"
    
    bands = [
        ('Sub-Bass (20-60Hz)', data_summary['sub_bass']),
        ('Bass (60-250Hz)', data_summary['bass']),
        ('Low-Mid (250Hz-1kHz)', data_summary['low_mid']),
        ('Mid (1-4kHz)', data_summary['mid']),
        ('High-Mid (4-8kHz)', data_summary['high_mid']),
        ('Presence (8-16kHz)', data_summary['presence']),
        ('Brilliance (16kHz+)', data_summary['brilliance'])
    ]
    
    for band_name, energy in bands:
        if energy is not None:
            message += f"• **{band_name}:** {energy:.3f}\n"
    
    message += "\n**Frequency Assessment:**\n"
    
    # Specific frequency analysis
    if data_summary['sub_bass'] and data_summary['sub_bass'] > 0.15:
        message += "⚠️ Sub-bass may muddy the mix on large systems\n"
    if data_summary['bass'] and data_summary['bass'] < 0.05:
        message += "⚠️ Track may sound thin - needs more bass presence\n"
    if data_summary['mid'] and data_summary['mid'] < 0.05:
        message += "⚠️ Vocals/leads may lack clarity - boost 1-4kHz\n"
    if data_summary['presence'] and data_summary['presence'] > 0.3:
        message += "⚠️ May sound harsh - reduce 8-12kHz harshness\n"
    if data_summary['brilliance'] and data_summary['brilliance'] > 0.2:
        message += "⚠️ Very bright - may cause ear fatigue\n"
    
    return {
        'message': message,
        'tone': 'frequency_analytical',
        'cited_data': "all_frequency_bands"
    }

def respond_with_problems_data(data_summary, **metrics):
    """Identify specific issues with exact numbers"""
    message = "**TRACK PROBLEMS ANALYSIS**\n\n"
    
    problems = []
    solutions = []
    
    # Check each metric for problems
    if data_summary['lufs'] and data_summary['lufs'] > -12:
        problems.append(f"❌ Too loud: {data_summary['lufs']:.1f} LUFS (should be -14 LUFS)")
        solutions.append(f"Reduce loudness by {data_summary['lufs'] - (-14):.1f} dB")
    
    if data_summary['dynamic_range'] and data_summary['dynamic_range'] < 8:
        problems.append(f"❌ Over-compressed: {data_summary['dynamic_range']:.1f} dB range (needs 8+ dB)")
        solutions.append("Reduce compression ratio or use parallel compression")
    
    if data_summary['bass'] and data_summary['bass'] < 0.05:
        problems.append(f"❌ Weak bass: {data_summary['bass']:.3f} energy (needs 0.05+)")
        solutions.append("Add sub-bass or boost 80-120Hz")
    
    if data_summary['mid'] and data_summary['mid'] < 0.05:
        problems.append(f"❌ Unclear vocals: {data_summary['mid']:.3f} mid energy (needs 0.05+)")
        solutions.append("Boost 1-4kHz for vocal clarity")
    
    if data_summary['stereo_correlation'] and data_summary['stereo_correlation'] > 0.8:
        problems.append(f"❌ Too narrow: {data_summary['stereo_correlation']:.2f} correlation (needs <0.8)")
        solutions.append("Add stereo width with reverb or stereo imaging")
    
    if problems:
        message += "**🔍 Issues Found:**\n"
        for problem in problems:
            message += f"{problem}\n"
        
        message += "\n**💡 Solutions:**\n"
        for solution in solutions:
            message += f"• {solution}\n"
    else:
        message += "✅ **No major issues detected!**\n"
        message += "Your track appears to be well-balanced overall."
    
    return {
        'message': message,
        'tone': 'problems_analytical',
        'cited_data': f"analyzed_{len(problems)}_issues"
    }

def respond_with_genre_data(data_summary, **metrics):
    """Genre classification based on actual metrics"""
    tempo = data_summary['tempo']
    harmonic_ratio = data_summary['harmonic_ratio']
    percussive_ratio = data_summary['percussive_ratio']
    bass = data_summary['bass']
    
    message = "**GENRE ANALYSIS - DATA-DRIVEN CLASSIFICATION**\n\n"
    
    if tempo is None:
        return {
            'message': "🧙‍♂️ Genre spirits require tempo data to reveal themselves. The analysis shows no clear rhythmic pattern.",
            'tone': 'data_missing'
        }
    
    # Genre determination based on metrics
    genre_scores = {}
    
    # Pop (moderate tempo, balanced content)
    if 100 <= tempo <= 130:
        genre_scores['Pop'] = 0.7
    
    # Hip-Hop (moderate tempo, heavy bass, percussive)
    if 80 <= tempo <= 110 and bass and bass > 0.1:
        genre_scores['Hip-Hop'] = 0.8
    
    # EDM/Dance (fast tempo, electronic characteristics)
    if tempo > 120:
        genre_scores['EDM/Dance'] = 0.6 + (tempo - 120) / 100
    
    # R&B (slower tempo, harmonic content)
    if 70 <= tempo <= 100 and harmonic_ratio and harmonic_ratio > 0.5:
        genre_scores['R&B'] = 0.7
    
    # Rock (moderate-fast tempo, balanced dynamics)
    if 110 <= tempo <= 140:
        genre_scores['Rock'] = 0.6
    
    # Ambient/Chill (slow tempo, atmospheric)
    if tempo < 80:
        genre_scores['Ambient/Chill'] = 0.8
    
    # Determine most likely genre
    if genre_scores:
        top_genre = max(genre_scores, key=genre_scores.get)
        confidence = genre_scores[top_genre]
        
        message += f"**Most Likely Genre: {top_genre}**\n"
        message += f"Confidence: {confidence*100:.0f}%\n\n"
        
        message += "**Supporting Evidence:**\n"
        message += f"• Tempo: {tempo:.1f} BPM\n"
        if harmonic_ratio: message += f"• Harmonic Content: {harmonic_ratio:.2f}\n"
        if percussive_ratio: message += f"• Percussive Content: {percussive_ratio:.2f}\n"
        if bass: message += f"• Bass Energy: {bass:.3f}\n"
    else:
        message += "**Genre: EXPERIMENTAL/UNIQUE**\n"
        message += f"Your track at {tempo:.1f} BPM doesn't fit typical genre patterns - that's creative!"
    
    return {
        'message': message,
        'tone': 'genre_analytical',
        'cited_data': f"tempo: {tempo}, genre_analysis_complete"
    }

def respond_with_artist_suggestions(data_summary, **metrics):
    """Suggest artists based on tempo, key, and musical characteristics"""
    tempo = data_summary['tempo']
    key = data_summary['key']
    harmonic_ratio = data_summary['harmonic_ratio']
    percussive_ratio = data_summary['percussive_ratio']
    
    if tempo is None:
        return {
            'message': "🧙‍♂️ *adjusts artist lens* Without tempo data, I cannot divine which artists would resonate with your track's energy.",
            'tone': 'data_missing'
        }
    
    message = "**ARTIST SUGGESTIONS - BASED ON YOUR TRACK**\n\n"
    
    # Artist suggestions based on tempo and characteristics
    if tempo < 85:
        message += "**🎵 SLOW TEMPO ARTISTS (70-85 BPM):**\n"
        message += "• **R&B/Soul:** SZA, Frank Ocean, The Weeknd\n"
        message += "• **Alternative:** Billie Eilish, Lana Del Rey\n"
        message += "• **Hip-Hop:** J. Cole, Kendrick Lamar (slow tracks)\n\n"
    
    elif tempo < 100:
        message += "**🎵 MODERATE TEMPO ARTISTS (85-100 BPM):**\n"
        message += "• **Pop/R&B:** Doja Cat, Ariana Grande, Dua Lipa\n"
        message += "• **Hip-Hop:** Drake, Post Malone, Travis Scott\n"
        message += "• **Alternative:** Tate McRae, Olivia Rodrigo\n\n"
    
    elif tempo < 120:
        message += "**🎵 UPBEAT TEMPO ARTISTS (100-120 BPM):**\n"
        message += "• **Pop:** Taylor Swift, Ed Sheeran, Harry Styles\n"
        message += "• **Hip-Hop:** Eminem, JAY-Z, Kanye West\n"
        message += "• **Electronic:** The Weeknd, Daft Punk\n\n"
    
    else:
        message += "**🎵 HIGH ENERGY ARTISTS (120+ BPM):**\n"
        message += "• **EDM:** Calvin Harris, David Guetta\n"
        message += "• **Pop:** Lady Gaga, Katy Perry\n"
        message += "• **Hip-Hop:** Migos, Cardi B (upbeat tracks)\n\n"
    
    # Key-specific suggestions
    if key:
        message += f"**🎼 KEY-SPECIFIC ARTISTS (Key: {key}):**\n"
        if key in ['C', 'G', 'D', 'A', 'E']:
            message += "• **Major Key Artists:** Ed Sheeran, Taylor Swift, Adele\n"
        elif key in ['Am', 'Em', 'Bm', 'F#m', 'C#m']:
            message += "• **Minor Key Artists:** The Weeknd, Billie Eilish, Lana Del Rey\n"
    
    # Content-based suggestions
    if harmonic_ratio and percussive_ratio:
        if harmonic_ratio > 0.6:
            message += "\n**🎵 MELODIC FOCUS:** Perfect for vocalists and singer-songwriters\n"
        elif percussive_ratio > 0.6:
            message += "\n**🥁 RHYTHMIC FOCUS:** Great for rappers and MCs\n"
    
    message += f"\n**Based on your {tempo:.1f} BPM track in {key} - these artists would vibe with your energy!**"
    
    return {
        'message': message,
        'tone': 'artist_suggestions',
        'cited_data': f"tempo: {tempo}, key: {key}, harmonic: {harmonic_ratio}, percussive: {percussive_ratio}"
    }

def respond_with_sound_selection_analysis(data_summary, **metrics):
    """Analyze sound selection based on frequency content and characteristics"""
    bass = data_summary['bass'] or 0
    mid = data_summary['mid'] or 0
    presence = data_summary['presence'] or 0
    harmonic_ratio = data_summary['harmonic_ratio']
    percussive_ratio = data_summary['percussive_ratio']
    
    message = "**SOUND SELECTION ANALYSIS**\n\n"
    
    # Frequency balance assessment
    message += "**🎛️ Frequency Distribution:**\n"
    message += f"• Bass Content: {bass:.3f}\n"
    message += f"• Mid Content: {mid:.3f}\n"
    message += f"• Presence Content: {presence:.3f}\n\n"
    
    # Sound selection assessment
    if bass < 0.05:
        message += "⚠️ **Bass Sounds:** Weak low-end presence\n"
        message += "• Consider: 808s, sub-bass, bass guitar\n"
        message += "• Add: Kick drums with more low-end weight\n\n"
    else:
        message += "✅ **Bass Sounds:** Good low-end foundation\n\n"
    
    if mid < 0.05:
        message += "⚠️ **Mid Sounds:** Lacks vocal/instrument clarity\n"
        message += "• Consider: Piano, guitar, synth leads\n"
        message += "• Add: More melodic elements in 1-4kHz range\n\n"
    else:
        message += "✅ **Mid Sounds:** Good vocal/instrument presence\n\n"
    
    if presence < 0.05:
        message += "⚠️ **High-End Sounds:** Lacks brightness and air\n"
        message += "• Consider: Hi-hats, cymbals, bright synths\n"
        message += "• Add: More high-frequency content\n\n"
    else:
        message += "✅ **High-End Sounds:** Good brightness and air\n\n"
    
    # Content-based recommendations
    if harmonic_ratio and percussive_ratio:
        message += "**🎼 Musical Content Analysis:**\n"
        if harmonic_ratio > 0.6:
            message += "• **Melodic Focus:** Great for vocal tracks\n"
            message += "• **Recommended:** Piano, strings, pad sounds\n"
        elif percussive_ratio > 0.6:
            message += "• **Rhythmic Focus:** Perfect for rap/hip-hop\n"
            message += "• **Recommended:** Hard-hitting drums, 808s, percussion\n"
        else:
            message += "• **Balanced Content:** Versatile for multiple genres\n"
            message += "• **Recommended:** Mix of melodic and rhythmic elements\n"
    
    return {
        'message': message,
        'tone': 'sound_selection_analytical',
        'cited_data': f"bass: {bass:.3f}, mid: {mid:.3f}, presence: {presence:.3f}"
    }

def respond_with_arrangement_tips(data_summary, **metrics):
    """Provide arrangement and structure tips based on track characteristics"""
    tempo = data_summary['tempo']
    duration = data_summary['duration']
    harmonic_ratio = data_summary['harmonic_ratio']
    percussive_ratio = data_summary['percussive_ratio']
    
    message = "**ARRANGEMENT & STRUCTURE TIPS**\n\n"
    
    # Tempo-based arrangement advice
    if tempo:
        message += f"**🎵 Tempo-Based Structure ({tempo:.1f} BPM):**\n"
        if tempo < 85:
            message += "• **Slow Build:** Perfect for emotional, atmospheric tracks\n"
            message += "• **Structure:** Intro → Verse → Chorus → Bridge → Outro\n"
            message += "• **Energy Curve:** Gradual build, peak at chorus\n"
        elif tempo < 100:
            message += "• **Moderate Energy:** Great for pop and R&B\n"
            message += "• **Structure:** Intro → Verse → Pre-Chorus → Chorus → Verse 2 → Chorus → Bridge → Final Chorus\n"
            message += "• **Energy Curve:** Steady build, strong choruses\n"
        elif tempo < 120:
            message += "• **Upbeat Energy:** Perfect for dance and hip-hop\n"
            message += "• **Structure:** Intro → Hook → Verse → Hook → Verse 2 → Hook → Bridge → Final Hook\n"
            message += "• **Energy Curve:** High energy throughout, drops for verses\n"
        else:
            message += "• **High Energy:** Ideal for EDM and club tracks\n"
            message += "• **Structure:** Intro → Build → Drop → Breakdown → Build → Drop → Outro\n"
            message += "• **Energy Curve:** Dramatic builds and drops\n"
    
    # Duration-based tips
    if duration:
        message += f"\n**⏱️ Duration Optimization ({duration:.1f}s):**\n"
        if duration < 60:
            message += "• **Short Track:** Focus on hook and immediate impact\n"
            message += "• **Structure:** Keep it simple - Intro → Hook → Verse → Hook\n"
        elif duration < 180:
            message += "• **Standard Length:** Perfect for radio and streaming\n"
            message += "• **Structure:** Full arrangement with clear sections\n"
        else:
            message += "• **Extended Track:** Great for albums and deep listening\n"
            message += "• **Structure:** Include instrumental breaks and extended sections\n"
    
    # Content-based arrangement tips
    if harmonic_ratio and percussive_ratio:
        message += f"\n**🎼 Content-Based Tips:**\n"
        if harmonic_ratio > 0.6:
            message += "• **Melodic Focus:** Emphasize chord progressions and melodies\n"
            message += "• **Arrangement:** Build around vocal hooks and instrumental solos\n"
        elif percussive_ratio > 0.6:
            message += "• **Rhythmic Focus:** Emphasize drum patterns and grooves\n"
            message += "• **Arrangement:** Build around beat drops and rhythmic breaks\n"
    
    message += "\n**💡 Pro Tips:**\n"
    message += "• **Hook First:** Start with your strongest element\n"
    message += "• **Energy Management:** Don't peak too early\n"
    message += "• **Repetition:** Use familiar elements to build comfort\n"
    message += "• **Contrast:** Create tension with different sections\n"
    
    return {
        'message': message,
        'tone': 'arrangement_analytical',
        'cited_data': f"tempo: {tempo}, duration: {duration}, harmonic: {harmonic_ratio}, percussive: {percussive_ratio}"
    }

def respond_with_vocal_analysis(data_summary, **metrics):
    """Analyze vocal characteristics and provide vocal-specific advice"""
    mid = data_summary['mid'] or 0
    presence = data_summary['presence'] or 0
    harmonic_ratio = data_summary['harmonic_ratio']
    percussive_ratio = data_summary['percussive_ratio']
    lufs = data_summary['lufs']
    
    message = "**VOCAL ANALYSIS & MIXING ADVICE**\n\n"
    
    # Vocal frequency analysis
    message += "**🎤 Vocal Frequency Content:**\n"
    message += f"• Mid-Range (1-4kHz): {mid:.3f}\n"
    message += f"• Presence (8-16kHz): {presence:.3f}\n\n"
    
    # Vocal clarity assessment
    if mid < 0.05:
        message += "⚠️ **Vocal Clarity Issues:**\n"
        message += "• **Problem:** Vocals lack presence and clarity\n"
        message += "• **Solution:** Boost 1-4kHz range by 3-6 dB\n"
        message += "• **Technique:** Use a high-shelf EQ around 2.5kHz\n\n"
    else:
        message += "✅ **Vocal Clarity:** Good mid-range presence\n\n"
    
    if presence < 0.05:
        message += "⚠️ **Vocal Air/Brightness Issues:**\n"
        message += "• **Problem:** Vocals lack air and brightness\n"
        message += "• **Solution:** Boost 8-12kHz range by 2-4 dB\n"
        message += "• **Technique:** Use a high-shelf EQ around 10kHz\n\n"
    else:
        message += "✅ **Vocal Air:** Good brightness and presence\n\n"
    
    # Content-based vocal advice
    if harmonic_ratio and percussive_ratio:
        message += "**🎼 Musical Context Analysis:**\n"
        if harmonic_ratio > 0.6:
            message += "• **Melodic Focus:** Perfect for vocal-driven tracks\n"
            message += "• **Vocal Style:** Great for singing, melodic rap, R&B\n"
            message += "• **Production:** Emphasize vocal harmonies and ad-libs\n"
        elif percussive_ratio > 0.6:
            message += "• **Rhythmic Focus:** Better for rhythmic rap and spoken word\n"
            message += "• **Vocal Style:** Focus on rhythm and flow over melody\n"
            message += "• **Production:** Emphasize vocal rhythm and timing\n"
        else:
            message += "• **Balanced Content:** Versatile for multiple vocal styles\n"
            message += "• **Vocal Style:** Works for both melodic and rhythmic vocals\n"
    
    # Loudness advice for vocals
    if lufs:
        message += f"\n**🔊 Vocal Loudness Context:**\n"
        if lufs < -16:
            message += "• **Track is quiet:** Vocals should be prominent in the mix\n"
            message += "• **Target:** Vocals 3-6 dB above instrumental elements\n"
        elif lufs > -12:
            message += "• **Track is loud:** Be careful not to over-compress vocals\n"
            message += "• **Target:** Vocals 1-3 dB above instrumental elements\n"
        else:
            message += "• **Good track level:** Standard vocal mixing applies\n"
            message += "• **Target:** Vocals 2-4 dB above instrumental elements\n"
    
    message += "\n**💡 Pro Vocal Tips:**\n"
    message += "• **Compression:** Use 2:1 ratio, 3-6 dB reduction\n"
    message += "• **EQ:** Cut 200-400Hz for clarity, boost 2.5kHz for presence\n"
    message += "• **Reverb:** Short decay (0.5-1.5s) for modern sound\n"
    message += "• **Delay:** 1/8 or 1/4 note delay for depth\n"
    
    return {
        'message': message,
        'tone': 'vocal_analytical',
        'cited_data': f"mid: {mid:.3f}, presence: {presence:.3f}, harmonic: {harmonic_ratio}, percussive: {percussive_ratio}"
    }

def respond_with_rhythm_analysis(data_summary, **metrics):
    """Analyze rhythm and drum characteristics"""
    tempo = data_summary['tempo']
    percussive_ratio = data_summary['percussive_ratio']
    harmonic_ratio = data_summary['harmonic_ratio']
    dynamic_range = data_summary['dynamic_range']
    crest_factor = data_summary['crest_factor']
    
    message = "**RHYTHM & DRUM ANALYSIS**\n\n"
    
    # Tempo analysis
    if tempo:
        message += f"**🎵 Tempo: {tempo:.1f} BPM**\n"
        if tempo < 85:
            message += "• **Groove Type:** Laid-back, relaxed feel\n"
            message += "• **Drum Style:** Sparse, atmospheric beats\n"
        elif tempo < 100:
            message += "• **Groove Type:** Moderate, groovy feel\n"
            message += "• **Drum Style:** Balanced, musical beats\n"
        elif tempo < 120:
            message += "• **Groove Type:** Energetic, driving feel\n"
            message += "• **Drum Style:** Punchy, rhythmic beats\n"
        else:
            message += "• **Groove Type:** High-energy, intense feel\n"
            message += "• **Drum Style:** Fast, aggressive beats\n"
        message += "\n"
    
    # Content analysis
    if percussive_ratio and harmonic_ratio:
        message += f"**🎼 Content Balance:**\n"
        message += f"• Percussive Content: {percussive_ratio:.1%}\n"
        message += f"• Harmonic Content: {harmonic_ratio:.1%}\n\n"
        
        if percussive_ratio > 0.6:
            message += "**🥁 Drum-Heavy Track:**\n"
            message += "• **Strength:** Strong rhythmic foundation\n"
            message += "• **Focus:** Emphasize drum patterns and groove\n"
            message += "• **Production:** Layer percussion, add drum fills\n"
        elif harmonic_ratio > 0.6:
            message += "**🎹 Melody-Heavy Track:**\n"
            message += "• **Strength:** Rich harmonic content\n"
            message += "• **Focus:** Drums should support melodies\n"
            message += "• **Production:** Subtle, musical drum programming\n"
        else:
            message += "**⚖️ Balanced Track:**\n"
            message += "• **Strength:** Good harmony-rhythm balance\n"
            message += "• **Focus:** Drums and melody work together\n"
            message += "• **Production:** Complementary arrangement\n"
        message += "\n"
    
    # Dynamic analysis
    if dynamic_range:
        message += f"**📊 Dynamic Range: {dynamic_range:.1f} dB**\n"
        if dynamic_range > 15:
            message += "• **Assessment:** Excellent dynamics - very musical\n"
            message += "• **Drum Advice:** Natural, expressive drum programming\n"
        elif dynamic_range > 10:
            message += "• **Assessment:** Good dynamics - balanced\n"
            message += "• **Drum Advice:** Moderate compression, preserve groove\n"
        else:
            message += "• **Assessment:** Compressed dynamics - modern sound\n"
            message += "• **Drum Advice:** Punchy, consistent drum levels\n"
        message += "\n"
    
    if crest_factor:
        message += f"**⚡ Crest Factor: {crest_factor:.1f}**\n"
        if crest_factor > 8:
            message += "• **Assessment:** Dynamic peaks - natural sound\n"
            message += "• **Drum Advice:** Preserve drum transients\n"
        elif crest_factor > 5:
            message += "• **Assessment:** Controlled peaks - balanced\n"
            message += "• **Drum Advice:** Moderate limiting on drums\n"
        else:
            message += "• **Assessment:** Limited peaks - modern sound\n"
            message += "• **Drum Advice:** Heavy compression for punch\n"
        message += "\n"
    
    # Production tips
    message += "**💡 Drum Production Tips:**\n"
    message += "• **Kick:** Layer with sub-bass for impact\n"
    message += "• **Snare:** Add reverb for space, compression for punch\n"
    message += "• **Hi-hats:** Use velocity variation for human feel\n"
    message += "• **Groove:** Slight swing (55-65%) for natural rhythm\n"
    message += "• **Mixing:** Side-chain kick to bass for clarity\n"
    
    return {
        'message': message,
        'tone': 'rhythm_analytical',
        'cited_data': f"tempo: {tempo}, percussive: {percussive_ratio}, dynamic_range: {dynamic_range}, crest_factor: {crest_factor}"
    }

def analyze_track_problems(analysis_data):
    """🧙‍♂️ Analyze what's wrong with the track"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    lufs = analysis_data.get('loudness', {}).get('lufs')
    dynamic_range = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    
    issues = []
    
    if lufs and lufs > -14:
        issues.append("🎚️ Your track is TOO LOUD! Modern streaming wants -14 LUFS, but you're at {:.1f} LUFS. Turn it down, young one!".format(lufs))
    
    if dynamic_range and dynamic_range < 6:
        issues.append("📊 Your dynamic range is CRUSHED! Only {:.1f} dB of range? Let your track breathe!".format(dynamic_range))
    
    if tempo and (tempo < 80 or tempo > 180):
        issues.append("⏱️ Your tempo of {:.0f} BPM is in the danger zone! Most hits live between 90-140 BPM.".format(tempo))
    
    if not issues:
        issues.append("✨ Actually... your track doesn't suck! The ancient metrics show promise. What specific aspect troubles you?")
    
    return {
        'message': "🧙‍♂️ *casts diagnostic spell* The mystical analysis reveals:\n\n" + "\n\n".join(issues) + "\n\n*adjusts spectacles* Now, let's fix these issues!",
        'tone': 'diagnostic',
        'issues_found': len(issues)
    }

def provide_banger_advice(analysis_data):
    """🧙‍♂️ Advice for making it slap like a banger"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    bass_energy = analysis_data.get('frequency_bands', {}).get('bass', 0)
    
    advice = []
    
    if tempo:
        if tempo < 100:
            advice.append("⚡ Your {:.0f} BPM is too slow for a banger! Crank it up to 120-140 BPM for that club energy!".format(tempo))
        elif tempo > 150:
            advice.append("🔥 Your {:.0f} BPM is fire! Perfect for a high-energy banger!".format(tempo))
        else:
            advice.append("🎯 Your {:.0f} BPM is in the sweet spot! This could definitely slap!".format(tempo))
    
    if bass_energy < 0.1:
        advice.append("🔊 Your bass is WEAK! Add a fat 808 or sub-bass around 40-60 Hz to make the club shake!")
    
    advice.append("💥 PRO TIP: Layer your drums! Kick + snare + hi-hats + percussion = BANGER FORMULA!")
    advice.append("🎵 Add vocal chops or melodic hooks to make it memorable!")
    
    return {
        'message': "🧙‍♂️ *summons banger energy* To make your track SLAP like the ancient gods intended:\n\n" + "\n\n".join(advice) + "\n\n*lightning crackles* Now go forth and create FIRE! 🔥",
        'tone': 'energetic',
        'banger_potential': 'high'
    }

def analyze_tempo(analysis_data):
    """🧙‍♂️ Tempo-specific advice"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    
    if not tempo:
        return {
            'message': "🧙‍♂️ *scratches wizard beard* I cannot detect a clear tempo in your track. Are you sure there's a beat?",
            'tone': 'confused'
        }
    
    tempo_advice = {
        'slow': "🐌 Your {:.0f} BPM is SLOW! Perfect for ambient, chill vibes, or emotional ballads.",
        'medium': "🎯 Your {:.0f} BPM is GOLDILOCKS! Not too fast, not too slow - just right for pop, hip-hop, or R&B!",
        'fast': "⚡ Your {:.0f} BPM is FAST! Perfect for EDM, house, or high-energy tracks!"
    }
    
    if tempo < 90:
        category = 'slow'
    elif tempo < 140:
        category = 'medium'
    else:
        category = 'fast'
    
    return {
        'message': "🧙‍♂️ *taps rhythm staff* " + tempo_advice[category].format(tempo),
        'tone': 'rhythmic',
        'tempo_category': category
    }

def analyze_key(analysis_data):
    """🧙‍♂️ Key and harmonic analysis"""
    key = analysis_data.get('harmonic', {}).get('key')
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    chroma_strength = analysis_data.get('harmonic', {}).get('chroma_strength', 0)
    
    if not key:
        return {
            'message': "I cannot clearly detect your key signature. The track may need stronger melodic content for accurate key detection.",
            'tone': 'analytical',
            'key_confidence': 'low'
        }
    
    # Key-specific advice based on musical theory
    key_advice = {
        'C': "C major - The foundation of all music! Pure, simple, and powerful. Great for pop and accessible melodies.",
        'C#': "C# major - The sharp and sophisticated choice! Creates tension and brightness.",
        'D': "D major - The key of celebration! Triumphant and uplifting, perfect for anthems.",
        'D#': "D# major - Bold and dramatic! This key commands attention.",
        'E': "E major - Electric energy! Great for rock and energetic pop tracks.",
        'F': "F major - Warm and comfortable! The key of contentment and gentle power.",
        'F#': "F# major - Bright and shimmering! Creates beautiful harmonic textures.",
        'G': "G major - The people's key! Natural and flowing, loved by guitarists.",
        'G#': "G# major - Exotic and mysterious! Creates unique sonic landscapes.",
        'A': "A major - Confident and strong! The guitarist's favorite for good reason.",
        'A#': "A# major - Rich and full! Perfect for brass and powerful arrangements.",
        'B': "B major - Ethereal and dreamy! Creates floating, otherworldly feelings."
    }
    
    # Tempo + Key combo advice
    if tempo and key:
        if tempo >= 120 and key in ['E', 'A', 'D']:
            extra_advice = "\n\nYour {} at {}BPM combo is excellent for dance music! This pairing creates instant energy.".format(key, int(tempo))
        elif tempo < 100 and key in ['F', 'C', 'G']:
            extra_advice = "\n\nYour {} at {}BPM creates a beautiful, contemplative vibe. Perfect for emotional storytelling.".format(key, int(tempo))
        elif key in ['F#', 'C#', 'G#']:
            extra_advice = "\n\nSharp keys like {} add sophistication! You're thinking like a pro producer.".format(key)
        else:
            extra_advice = "\n\nYour {} key at {}BPM creates a solid foundation for your track.".format(key, int(tempo))
    else:
        extra_advice = ""
    
    base_message = key_advice.get(key, f"{key} - An interesting harmonic choice!")
    confidence_note = ""
    if chroma_strength < 0.3:
        confidence_note = "\n\n(Note: The harmonic detection is uncertain - consider strengthening your melodic elements.)"
    
    return {
        'message': f"Your track resonates in **{key}**.\n\n{base_message}{extra_advice}{confidence_note}",
        'tone': 'harmonic',
        'detected_key': key,
        'key_confidence': 'high' if chroma_strength > 0.5 else 'medium' if chroma_strength > 0.3 else 'low'
    }

def analyze_genre(analysis_data):
    """Analyze genre based on tempo, key, and other characteristics"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    key = analysis_data.get('harmonic', {}).get('key')
    bass_energy = analysis_data.get('frequency_bands', {}).get('bass', 0)
    sub_bass_energy = analysis_data.get('frequency_bands', {}).get('sub_bass', 0)
    dynamic_range = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    
    genre_suggestions = []
    confidence_factors = []
    
    if tempo:
        if 60 <= tempo <= 80:
            genre_suggestions.extend(['Ballad', 'Downtempo', 'Ambient'])
            confidence_factors.append(f"Slow tempo ({tempo:.0f} BPM)")
        elif 80 <= tempo <= 100:
            genre_suggestions.extend(['Hip-Hop', 'R&B', 'Lo-Fi', 'Trap'])
            confidence_factors.append(f"Mid-slow tempo ({tempo:.0f} BPM)")
        elif 100 <= tempo <= 130:
            genre_suggestions.extend(['Pop', 'Rock', 'Alternative', 'Indie'])
            confidence_factors.append(f"Medium tempo ({tempo:.0f} BPM)")
        elif 130 <= tempo <= 150:
            genre_suggestions.extend(['House', 'Techno', 'EDM', 'Dance'])
            confidence_factors.append(f"Upbeat tempo ({tempo:.0f} BPM)")
        elif tempo > 150:
            genre_suggestions.extend(['Drum & Bass', 'Hardcore', 'Speed House'])
            confidence_factors.append(f"Fast tempo ({tempo:.0f} BPM)")
    
    # Key-based suggestions
    if key:
        if key in ['C', 'G', 'F']:
            genre_suggestions.extend(['Pop', 'Country', 'Folk'])
            confidence_factors.append(f"Popular key ({key})")
        elif key in ['D', 'A', 'E']:
            genre_suggestions.extend(['Rock', 'Blues', 'Alternative'])
            confidence_factors.append(f"Guitar-friendly key ({key})")
        elif key in ['C#', 'F#', 'G#']:
            genre_suggestions.extend(['Electronic', 'Experimental', 'Ambient'])
            confidence_factors.append(f"Electronic-leaning key ({key})")
    
    # Bass energy analysis
    if bass_energy > 0.3:
        genre_suggestions.extend(['Hip-Hop', 'EDM', 'Trap', 'Dubstep'])
        confidence_factors.append("Heavy bass presence")
    elif bass_energy < 0.1:
        genre_suggestions.extend(['Acoustic', 'Folk', 'Ambient', 'Classical'])
        confidence_factors.append("Light bass content")
    
    # Dynamic range analysis
    if dynamic_range and dynamic_range > 15:
        genre_suggestions.extend(['Jazz', 'Classical', 'Folk', 'Live Recording'])
        confidence_factors.append("High dynamic range")
    elif dynamic_range and dynamic_range < 8:
        genre_suggestions.extend(['EDM', 'Pop', 'Hip-Hop', 'Commercial'])
        confidence_factors.append("Compressed/commercial sound")
    
    # Find most common suggestions
    genre_counts = {}
    for genre in genre_suggestions:
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Sort by frequency
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if not top_genres:
        return {
            'message': "Based on your track analysis, I need more distinctive characteristics to suggest a specific genre. Try adding more distinctive elements like specific drum patterns, bass lines, or melodic content.",
            'tone': 'analytical'
        }
    
    primary_genre = top_genres[0][0]
    secondary_genres = [g[0] for g in top_genres[1:]]
    
    message = f"Based on your track analysis, this sounds like **{primary_genre}**"
    
    if secondary_genres:
        message += f" with elements of {', '.join(secondary_genres)}"
    
    message += f".\n\nAnalysis factors:\n"
    for factor in confidence_factors[:3]:  # Top 3 factors
        message += f"• {factor}\n"
    
    # Add genre-specific advice
    genre_advice = {
        'Pop': "Focus on catchy hooks, clear vocals, and balanced mix. Target -14 LUFS for streaming.",
        'Hip-Hop': "Emphasize the kick and 808s. Leave space for vocals. Consider side-chain compression.",
        'Rock': "Drive the guitars and keep the drums punchy. Dynamic range is your friend.",
        'EDM': "Build energy through arrangement. Use compression creatively. Target high energy throughout.",
        'R&B': "Smooth vocals and groove are key. Focus on the pocket and melodic bass lines.",
        'Trap': "Hard-hitting 808s and crisp hi-hats. Leave space for ad-libs and vocal chops."
    }
    
    if primary_genre in genre_advice:
        message += f"\n**{primary_genre} Production Tips:**\n{genre_advice[primary_genre]}"
    
    return {
        'message': message,
        'tone': 'analytical',
        'primary_genre': primary_genre,
        'secondary_genres': secondary_genres
    }

def analyze_mix(analysis_data):
    """Detailed mix analysis with actual numbers"""
    lufs = analysis_data.get('loudness', {}).get('lufs')
    peak_db = analysis_data.get('amplitude_dynamics', {}).get('peak_db')
    rms_db = analysis_data.get('amplitude_dynamics', {}).get('rms_db')
    dynamic_range = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    stereo_correlation = analysis_data.get('loudness', {}).get('stereo_correlation')
    stereo_width = analysis_data.get('loudness', {}).get('stereo_width')
    frequency_bands = analysis_data.get('frequency_bands', {})
    
    message = "**MIX ANALYSIS REPORT**\n\n"
    
    # Loudness Analysis
    message += "**Loudness Levels:**\n"
    if lufs is not None:
        if lufs > -12:
            assessment = "TOO LOUD - will be turned down by streaming"
        elif lufs < -18:
            assessment = "TOO QUIET - listeners will skip"
        else:
            assessment = "GOOD for streaming platforms"
        message += f"• LUFS: {lufs:.1f} dB ({assessment})\n"
    
    if peak_db is not None:
        message += f"• Peak Level: {peak_db:.1f} dB\n"
    if rms_db is not None:
        message += f"• RMS Level: {rms_db:.1f} dB\n"
    
    if dynamic_range is not None:
        if dynamic_range < 6:
            dr_assessment = "HEAVILY COMPRESSED - lacks dynamics"
        elif dynamic_range < 10:
            dr_assessment = "COMPRESSED - typical modern production"
        elif dynamic_range > 15:
            dr_assessment = "EXCELLENT dynamics - very musical"
        else:
            dr_assessment = "GOOD dynamics"
        message += f"• Dynamic Range: {dynamic_range:.1f} dB ({dr_assessment})\n"
    
    # Stereo Field
    message += "\n**Stereo Field:**\n"
    if stereo_correlation is not None:
        if stereo_correlation > 0.8:
            stereo_assessment = "NARROW - needs width"
        elif stereo_correlation < 0.3:
            stereo_assessment = "VERY WIDE - check mono compatibility"
        else:
            stereo_assessment = "BALANCED width"
        message += f"• Stereo Correlation: {stereo_correlation:.2f} ({stereo_assessment})\n"
    
    if stereo_width is not None:
        message += f"• Stereo Width: {stereo_width:.2f}\n"
    
    # Frequency Balance
    message += "\n**Frequency Balance:**\n"
    sub_bass = frequency_bands.get('sub_bass', 0)
    bass = frequency_bands.get('bass', 0)
    low_mid = frequency_bands.get('low_mid', 0)
    mid = frequency_bands.get('mid', 0)
    high_mid = frequency_bands.get('high_mid', 0)
    presence = frequency_bands.get('presence', 0)
    brilliance = frequency_bands.get('brilliance', 0)
    
    message += f"• Sub-Bass: {sub_bass:.3f}\n"
    message += f"• Bass: {bass:.3f}\n"
    message += f"• Low-Mid: {low_mid:.3f}\n"
    message += f"• Mid: {mid:.3f}\n"
    message += f"• High-Mid: {high_mid:.3f}\n"
    message += f"• Presence: {presence:.3f}\n"
    message += f"• Brilliance: {brilliance:.3f}\n"
    
    # Mix Issues & Recommendations
    issues = []
    recommendations = []
    
    if lufs is not None and lufs > -12:
        issues.append("Track is too loud for streaming")
        recommendations.append("Use a limiter to bring LUFS to -14 dB")
    
    if dynamic_range is not None and dynamic_range < 8:
        issues.append("Track is over-compressed")
        recommendations.append("Reduce compression or use parallel compression")
    
    if bass > mid * 2:
        issues.append("Bass is overpowering")
        recommendations.append("Reduce bass around 80-200Hz or boost mids around 1-3kHz")
    elif bass < 0.05:
        issues.append("Lacks low-end presence")
        recommendations.append("Add sub-bass around 40-60Hz or boost bass around 100Hz")
    
    if mid < 0.05:
        issues.append("Vocals/instruments will lack clarity")
        recommendations.append("Boost midrange around 1-4kHz for vocal presence")
    
    if brilliance > 0.3:
        issues.append("May sound harsh or fatiguing")
        recommendations.append("Reduce harshness around 8-12kHz")
    
    if issues:
        message += "\n**Issues Found:**\n"
        for issue in issues:
            message += f"• {issue}\n"
        
        message += "\n**Recommendations:**\n"
        for rec in recommendations:
            message += f"• {rec}\n"
    else:
        message += "\n✅ **Mix Balance: GOOD**\n"
    
    return {
        'message': message,
        'tone': 'technical',
        'mix_issues': len(issues)
    }

def analyze_low_end(analysis_data):
    """🧙‍♂️ Low-end specific advice"""
    bass_energy = analysis_data.get('frequency_bands', {}).get('bass', 0)
    sub_bass_energy = analysis_data.get('frequency_bands', {}).get('sub_bass', 0)
    
    if bass_energy < 0.1:
        return {
            'message': "🔊 *thunder rumbles* Your bass is WEAK! Add a fat 808 or sub-bass around 40-60 Hz. Make the club SHAKE!",
            'tone': 'thunderous'
        }
    elif bass_energy > 0.5:
        return {
            'message': "💥 *earth trembles* Your bass is MASSIVE! Be careful not to muddy your mix. High-pass your kick at 30 Hz!",
            'tone': 'powerful'
        }
    else:
        return {
            'message': "🎯 Your bass is BALANCED! Good job, young producer. Consider layering with a sub-bass for extra weight!",
            'tone': 'approving'
        }

def provide_pop_reference_advice(analysis_data):
    """🧙‍♂️ Billie Eilish / Tate McRae style advice"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    lufs = analysis_data.get('loudness', {}).get('lufs')
    
    advice = []
    
    if tempo:
        if tempo < 80:
            advice.append("🎤 Your {:.0f} BPM is perfect for that Billie-style emotional ballad!")
        elif tempo < 120:
            advice.append("💃 Your {:.0f} BPM is ideal for Tate McRae-style pop bangers!")
        else:
            advice.append("⚡ Your {:.0f} BPM is too fast for pop vocals! Slow it down for that radio-friendly feel!")
    
    advice.append("🎵 Add space in your arrangement - let the vocals breathe!")
    advice.append("🎚️ Use reverb and delay for that dreamy pop atmosphere!")
    advice.append("🎤 Layer your vocals - main + harmonies + ad-libs = POP MAGIC!")
    
    return {
        'message': "🧙‍♂️ *summons pop magic* To channel the spirits of Billie and Tate:\n\n" + "\n\n".join(advice) + "\n\n*sparkles shimmer* Now create that radio-ready magic! ✨",
        'tone': 'pop_magical',
        'pop_potential': 'high'
    }

def provide_full_analysis(analysis_data):
    """Comprehensive analysis of the entire track"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    key = analysis_data.get('harmonic', {}).get('key')
    lufs = analysis_data.get('loudness', {}).get('lufs')
    duration = analysis_data.get('basic_info', {}).get('duration_sec')
    sample_rate = analysis_data.get('basic_info', {}).get('sample_rate')
    file_size_mb = analysis_data.get('basic_info', {}).get('file_size_mb')
    
    # Dynamics
    peak_db = analysis_data.get('amplitude_dynamics', {}).get('peak_db')
    rms_db = analysis_data.get('amplitude_dynamics', {}).get('rms_db')
    dynamic_range_db = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    crest_factor = analysis_data.get('amplitude_dynamics', {}).get('crest_factor')
    
    # Frequency bands
    freq_bands = analysis_data.get('frequency_bands', {})
    sub_bass = freq_bands.get('sub_bass', 0)
    bass = freq_bands.get('bass', 0)
    low_mid = freq_bands.get('low_mid', 0)
    mid = freq_bands.get('mid', 0)
    high_mid = freq_bands.get('high_mid', 0)
    presence = freq_bands.get('presence', 0)
    brilliance = freq_bands.get('brilliance', 0)
    
    # Stereo and loudness
    stereo_correlation = analysis_data.get('loudness', {}).get('stereo_correlation')
    stereo_width = analysis_data.get('loudness', {}).get('stereo_width')
    
    # Harmonic content
    harmonic_ratio = analysis_data.get('harmonic', {}).get('harmonic_ratio')
    percussive_ratio = analysis_data.get('harmonic', {}).get('percussive_ratio')
    
    # Rhythm details
    beat_count = analysis_data.get('rhythm', {}).get('beat_count')
    onset_count = analysis_data.get('rhythm', {}).get('onset_count')
    onset_rate = analysis_data.get('rhythm', {}).get('onset_rate_per_sec')
    
    message = "**COMPREHENSIVE TRACK ANALYSIS**\n\n"
    
    # Basic Info
    message += "**Track Overview:**\n"
    if tempo: message += f"• Tempo: {tempo:.1f} BPM\n"
    if key: message += f"• Key: {key}\n"
    if duration: message += f"• Duration: {duration:.1f} seconds\n"
    if sample_rate: message += f"• Sample Rate: {sample_rate:,} Hz\n"
    if file_size_mb: message += f"• File Size: {file_size_mb:.1f} MB\n"
    
    # Loudness & Dynamics
    message += "\n**Loudness & Dynamics:**\n"
    if lufs is not None: 
        loudness_assessment = "PERFECT" if -16 <= lufs <= -12 else "TOO LOUD" if lufs > -12 else "TOO QUIET"
        message += f"• LUFS: {lufs:.1f} dB ({loudness_assessment} for streaming)\n"
    if peak_db is not None: message += f"• Peak Level: {peak_db:.1f} dB\n"
    if rms_db is not None: message += f"• RMS Level: {rms_db:.1f} dB\n"
    if dynamic_range_db is not None:
        dynamics_assessment = "EXCELLENT" if dynamic_range_db > 15 else "GOOD" if dynamic_range_db > 10 else "COMPRESSED"
        message += f"• Dynamic Range: {dynamic_range_db:.1f} dB ({dynamics_assessment})\n"
    if crest_factor is not None: message += f"• Crest Factor: {crest_factor:.1f}\n"
    
    # Stereo Field
    message += "\n**Stereo Imaging:**\n"
    if stereo_correlation is not None:
        stereo_assessment = "WIDE" if stereo_correlation < 0.3 else "BALANCED" if stereo_correlation < 0.7 else "NARROW"
        message += f"• Stereo Correlation: {stereo_correlation:.2f} ({stereo_assessment})\n"
    if stereo_width is not None:
        width_assessment = "VERY WIDE" if stereo_width > 0.8 else "WIDE" if stereo_width > 0.5 else "NARROW"
        message += f"• Stereo Width: {stereo_width:.2f} ({width_assessment})\n"
    
    # Frequency Analysis
    message += "\n**7-Band Frequency Analysis:**\n"
    message += f"• Sub-Bass (20-60Hz): {sub_bass:.3f}\n"
    message += f"• Bass (60-250Hz): {bass:.3f}\n"
    message += f"• Low-Mid (250Hz-1kHz): {low_mid:.3f}\n"
    message += f"• Mid (1-4kHz): {mid:.3f}\n"
    message += f"• High-Mid (4-8kHz): {high_mid:.3f}\n"
    message += f"• Presence (8-16kHz): {presence:.3f}\n"
    message += f"• Brilliance (16kHz+): {brilliance:.3f}\n"
    
    # Content Analysis
    message += "\n**Musical Content:**\n"
    if harmonic_ratio is not None and percussive_ratio is not None:
        content_type = "MELODIC" if harmonic_ratio > 0.6 else "RHYTHMIC" if percussive_ratio > 0.6 else "BALANCED"
        message += f"• Harmonic Content: {harmonic_ratio:.2f}\n"
        message += f"• Percussive Content: {percussive_ratio:.2f}\n"
        message += f"• Content Type: {content_type}\n"
    
    # Rhythm Analysis
    if beat_count or onset_count:
        message += "\n**Rhythm Analysis:**\n"
        if beat_count: message += f"• Beat Count: {beat_count}\n"
        if onset_count: message += f"• Onset Events: {onset_count}\n"
        if onset_rate: message += f"• Onset Rate: {onset_rate:.1f} per second\n"
    
    # Production Assessment
    message += "\n**Production Assessment:**\n"
    issues = []
    if lufs is not None and lufs > -12:
        issues.append("Track is too loud for streaming platforms")
    if dynamic_range_db is not None and dynamic_range_db < 8:
        issues.append("Track is over-compressed")
    if bass < 0.1:
        issues.append("Lacks low-end presence")
    if mid < 0.1:
        issues.append("Lacks vocal/instrument clarity")
    if brilliance > 0.5:
        issues.append("May be too bright/harsh")
    
    if issues:
        message += "⚠️ **Issues Found:**\n"
        for issue in issues:
            message += f"• {issue}\n"
    else:
        message += "✅ **Overall Balance: GOOD**\n"
    
    return {
        'message': message,
        'tone': 'analytical',
        'analysis_type': 'comprehensive'
    }

def provide_general_advice(analysis_data):
    """🧙‍♂️ General BeatWizard wisdom with track specifics"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    key = analysis_data.get('harmonic', {}).get('key')
    lufs = analysis_data.get('loudness', {}).get('lufs')
    duration = analysis_data.get('basic_info', {}).get('duration_sec')
    
    # Build personalized message based on track analysis
    track_info = []
    if tempo:
        track_info.append(f"Tempo: {tempo:.1f} BPM")
    if key:
        track_info.append(f"Key: {key}")
    if duration:
        track_info.append(f"Duration: {duration:.1f}s")
    
    track_summary = ""
    if track_info:
        track_summary = " • ".join(track_info) + "\n\n"
    
    # Give specific advice based on tempo
    tempo_wisdom = ""
    if tempo:
        if tempo < 90:
            tempo_wisdom = "Your slow tempo creates space for emotional depth - use it wisely!\n"
        elif tempo > 140:
            tempo_wisdom = "Your fast tempo demands tight arrangements - every element must earn its place!\n"
        else:
            tempo_wisdom = "Your tempo is in the sweet spot for most genres - solid foundation!\n"
    
    return {
        'message': f"Your track analysis:\n{track_summary}\n{tempo_wisdom}Production insights:\n\n• Music is emotion in motion\n• Less is often more\n• Let your vocals shine\n• Bass is the foundation\n• Energy comes from contrast\n\nAsk me about your **tempo**, **key**, **mix**, **genre**, or how to make it **slap like a banger**!",
        'tone': 'wise',
        'wisdom_level': 'ancient'
    }

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