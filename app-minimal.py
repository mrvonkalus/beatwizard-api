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
                band_energy = np.mean(S[band_bins, :])
                frequency_analysis[band_name] = float(band_energy)
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
    🧙‍♂️ Generate BeatWizard's mystical response based on user message and track analysis
    """
    # Default mystical greeting
    if not analysis_data:
        return {
            'message': "🧙‍♂️ *adjusts wizard hat* Ah, young producer! I sense you seek the ancient wisdom of the beats, yet no track analysis lies before me. Upload your musical creation first, and then I shall bestow upon you the sacred knowledge of production! ✨",
            'tone': 'mystical_greeting'
        }
    
    # Extract key metrics for analysis
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    key = analysis_data.get('harmonic', {}).get('key')
    lufs = analysis_data.get('loudness', {}).get('lufs')
    dynamic_range = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    bass_energy = analysis_data.get('frequency_bands', {}).get('bass', 0)
    sub_bass_energy = analysis_data.get('frequency_bands', {}).get('sub_bass', 0)
    
    # 🎯 BEATWIZARD RESPONSE LOGIC
    user_lower = user_message.lower()
    
    # Check for specific questions
    if any(word in user_lower for word in ['suck', 'bad', 'terrible', 'awful', 'trash']):
        return analyze_track_problems(analysis_data)
    elif any(word in user_lower for word in ['slap', 'banger', 'hit', 'fire', 'dope']):
        return provide_banger_advice(analysis_data)
    elif any(word in user_lower for word in ['tempo', 'bpm', 'speed']):
        return analyze_tempo(analysis_data)
    elif any(word in user_lower for word in ['key', 'scale', 'harmonic', 'chord']):
        return analyze_key(analysis_data)
    elif any(word in user_lower for word in ['mix', 'mixing', 'balance']):
        return analyze_mix(analysis_data)
    elif any(word in user_lower for word in ['bass', 'kick', 'low end']):
        return analyze_low_end(analysis_data)
    elif any(word in user_lower for word in ['billie', 'eilish', 'tate', 'mcrae']):
        return provide_pop_reference_advice(analysis_data)
    else:
        return provide_general_advice(analysis_data)

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
            'message': "🧙‍♂️ *squints at harmonic crystals* The ancient spirits are unclear about your key signature... Perhaps the track needs stronger melodic content for the mystical algorithms to detect! ✨",
            'tone': 'mystical',
            'key_confidence': 'low'
        }
    
    # Key-specific advice based on musical theory
    key_advice = {
        'C': "🎹 C major - The foundation of all music! Pure, simple, and powerful. Great for pop and accessible melodies.",
        'C#': "✨ C# major - The sharp and sophisticated choice! Creates tension and brightness.",
        'D': "🌅 D major - The key of celebration! Triumphant and uplifting, perfect for anthems.",
        'D#': "🔥 D# major - Bold and dramatic! This key commands attention.",
        'E': "⚡ E major - Electric energy! Great for rock and energetic pop tracks.",
        'F': "🎵 F major - Warm and comfortable! The key of contentment and gentle power.",
        'F#': "🌟 F# major - Bright and shimmering! Creates beautiful harmonic textures.",
        'G': "🎶 G major - The people's key! Natural and flowing, loved by guitarists.",
        'G#': "💫 G# major - Exotic and mysterious! Creates unique sonic landscapes.",
        'A': "🎸 A major - Confident and strong! The guitarist's favorite for good reason.",
        'A#': "🎺 A# major - Rich and full! Perfect for brass and powerful arrangements.",
        'B': "🌙 B major - Ethereal and dreamy! Creates floating, otherworldly feelings."
    }
    
    # Tempo + Key combo advice
    if tempo and key:
        if tempo >= 120 and key in ['E', 'A', 'D']:
            extra_advice = "\n\n🔥 Your {} at {}BPM combo is FIRE for dance music! This pairing creates instant energy!".format(key, int(tempo))
        elif tempo < 100 and key in ['F', 'C', 'G']:
            extra_advice = "\n\n💫 Your {} at {}BPM creates a beautiful, contemplative vibe. Perfect for emotional storytelling!".format(key, int(tempo))
        elif key in ['F#', 'C#', 'G#']:
            extra_advice = "\n\n✨ Sharp keys like {} add sophistication! You're thinking like a pro producer.".format(key)
        else:
            extra_advice = "\n\n🎵 Your {} key at {}BPM creates a solid foundation for your track!".format(key, int(tempo))
    else:
        extra_advice = ""
    
    base_message = key_advice.get(key, f"🎵 {key} - An interesting harmonic choice!")
    confidence_note = ""
    if chroma_strength < 0.3:
        confidence_note = "\n\n🔮 (The harmonic detection is a bit uncertain - consider strengthening your melodic elements!)"
    
    return {
        'message': f"🧙‍♂️ *consulting the Circle of Fifths* Your track resonates in **{key}**!\n\n{base_message}{extra_advice}{confidence_note}",
        'tone': 'harmonic',
        'detected_key': key,
        'key_confidence': 'high' if chroma_strength > 0.5 else 'medium' if chroma_strength > 0.3 else 'low'
    }

def analyze_mix(analysis_data):
    """🧙‍♂️ Mix analysis and advice"""
    lufs = analysis_data.get('loudness', {}).get('lufs')
    dynamic_range = analysis_data.get('amplitude_dynamics', {}).get('dynamic_range_db')
    frequency_bands = analysis_data.get('frequency_bands', {})
    
    mix_advice = []
    
    if lufs and lufs > -14:
        mix_advice.append("🎚️ Your mix is TOO HOT! Target -14 LUFS for streaming. Use a limiter, young one!")
    
    if dynamic_range and dynamic_range < 8:
        mix_advice.append("📊 Your mix is CRUSHED! Aim for 8-12 dB of dynamic range for breathing room.")
    
    bass = frequency_bands.get('bass', 0)
    mid = frequency_bands.get('mid', 0)
    high = frequency_bands.get('presence', 0) + frequency_bands.get('brilliance', 0)
    
    if bass > mid * 2:
        mix_advice.append("🔊 Your bass is DOMINATING! Balance it with your mids around 1-4 kHz.")
    elif mid > bass * 2:
        mix_advice.append("🎵 Your mids are THIN! Add some warmth in the 200-800 Hz range.")
    
    return {
        'message': "🧙‍♂️ *casts mixing spell* Your mix analysis reveals:\n\n" + "\n\n".join(mix_advice) + "\n\n*adjusts spectral glasses* These adjustments will bring balance to your track!",
        'tone': 'technical',
        'mix_issues': len(mix_advice)
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

def provide_general_advice(analysis_data):
    """🧙‍♂️ General BeatWizard wisdom with track specifics"""
    tempo = analysis_data.get('rhythm', {}).get('tempo_bpm')
    key = analysis_data.get('harmonic', {}).get('key')
    lufs = analysis_data.get('loudness', {}).get('lufs')
    duration = analysis_data.get('basic_info', {}).get('duration_sec')
    
    # Build personalized message based on track analysis
    track_info = []
    if tempo:
        track_info.append(f"🎵 Tempo: {tempo:.1f} BPM")
    if key:
        track_info.append(f"🎹 Key: {key}")
    if duration:
        track_info.append(f"⏱️ Duration: {duration:.1f}s")
    
    track_summary = ""
    if track_info:
        track_summary = "Your track analysis:\n" + " • ".join(track_info) + "\n\n"
    
    # Give specific advice based on tempo
    tempo_wisdom = ""
    if tempo:
        if tempo < 90:
            tempo_wisdom = "🐌 Your slow tempo creates space for emotional depth - use it wisely!\n"
        elif tempo > 140:
            tempo_wisdom = "⚡ Your fast tempo demands tight arrangements - every element must earn its place!\n"
        else:
            tempo_wisdom = "🎯 Your tempo is in the sweet spot for most genres - solid foundation!\n"
    
    return {
        'message': f"🧙‍♂️ *consulting the mystical analysis scrolls*\n\n{track_summary}{tempo_wisdom}Remember the ancient production wisdom:\n\n🎵 Music is emotion in motion\n🎚️ Less is often more\n🎤 Let your vocals shine\n🔊 Bass is the foundation\n⚡ Energy comes from contrast\n\n*staff glows* Ask me about your **tempo**, **key**, **mix**, or how to make it **slap like a banger**! ✨",
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