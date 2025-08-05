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
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

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
app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max file size (reduced)
    SECRET_KEY=os.environ.get('SECRET_KEY', 'minimal-demo-secret-key'),
    ENV='production',
    DEBUG=False,
    TESTING=False
)

# Configure CORS for production
cors_origins = os.environ.get('CORS_ORIGINS', '*').split(',')
CORS(app, 
     origins=cors_origins,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS']
)

# Trust Railway's proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Supported audio file extensions (for when we add audio back)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aiff', 'ogg'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'service': 'BeatWizard Audio Analysis API',
        'version': '1.0.0-minimal',
        'audio_processing': AUDIO_AVAILABLE,
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
            'POST /api/echo - Echo test'
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

@app.route('/api/status')
def deployment_status():
    """Show current deployment status and next steps"""
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
        'railway_deployment': 'SUCCESS' if AUDIO_AVAILABLE is False else 'UNKNOWN'
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
            'GET /api/status'
        ]
    }), 404

if __name__ == '__main__':
    # Get port from environment (Railway provides this)
    port = int(os.environ.get('PORT', 8080))
    
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