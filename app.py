#!/usr/bin/env python3
"""
BeatWizard Production Flask App for Railway Deployment
Professional AI-powered music analysis platform
"""

import os
import sys
import uuid
import tempfile
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import boto3
from botocore.exceptions import ClientError

# Import our BeatWizard system
from beatwizard import EnhancedAudioAnalyzer
from config.settings import setup_logging
from loguru import logger

# Initialize Flask app
app = Flask(__name__)

# Production configuration
app.config.update(
    MAX_CONTENT_LENGTH=200 * 1024 * 1024,  # 200MB max file size
    SECRET_KEY=os.environ.get('SECRET_KEY', 'production-secret-key-change-this'),
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

# Initialize BeatWizard analyzer
analyzer = None

# Supported audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aiff', 'ogg'}

def init_analyzer():
    """Initialize the audio analyzer with error handling"""
    global analyzer
    try:
        analyzer = EnhancedAudioAnalyzer()
        logger.info("✅ BeatWizard analyzer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {e}")
        analyzer = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_cloud_storage():
    """Setup cloud storage client (optional)"""
    try:
        if os.environ.get('AWS_ACCESS_KEY_ID'):
            return boto3.client('s3')
    except Exception as e:
        logger.warning(f"Cloud storage not configured: {e}")
    return None

# Initialize cloud storage
s3_client = setup_cloud_storage()

# Initialize components at startup
init_analyzer()

@app.route('/')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'service': 'BeatWizard Audio Analysis API',
        'version': '1.0.0',
        'analyzer_status': 'ready' if analyzer else 'initializing',
        'timestamp': time.time()
    })

@app.route('/health')
def detailed_health():
    """Detailed health check for monitoring"""
    health_status = {
        'status': 'healthy',
        'checks': {
            'analyzer': analyzer is not None,
            'memory': True,  # Could add memory checks
            'disk': True,    # Could add disk space checks
        },
        'timestamp': time.time()
    }
    
    all_healthy = all(health_status['checks'].values())
    status_code = 200 if all_healthy else 503
    
    if not all_healthy:
        health_status['status'] = 'unhealthy'
    
    return jsonify(health_status), status_code

@app.route('/api/upload', methods=['POST'])
def upload_and_analyze():
    """Handle file upload and audio analysis"""
    if not analyzer:
        return jsonify({
            'error': 'Audio analyzer not available',
            'details': 'Service is initializing, please try again in a few moments'
        }), 503
    
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'File type not supported',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }), 400
    
    # Get analysis parameters
    skill_level = request.form.get('skill_level', 'beginner')
    genre = request.form.get('genre', 'electronic')
    goals = request.form.get('goals', 'streaming').split(',')
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    
    # Use temporary file for processing
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        logger.info(f"🔄 Starting analysis for: {filename} (ID: {analysis_id})")
        start_time = time.time()
        
        # Run BeatWizard analysis
        analysis_results = analyzer.analyze_track(temp_file_path)
        
        # Generate intelligent feedback
        feedback = analyzer.generate_intelligent_feedback(
            analysis_results,
            skill_level=skill_level,
            genre=genre,
            goals=goals
        )
        
        analysis_time = time.time() - start_time
        logger.info(f"✅ Analysis completed in {analysis_time:.1f}s for {filename}")
        
        # Prepare response
        response_data = {
            'success': True,
            'analysis_id': analysis_id,
            'filename': filename,
            'analysis_time': round(analysis_time, 2),
            'analysis_results': make_json_serializable(analysis_results),
            'intelligent_feedback': make_json_serializable(feedback),
            'parameters': {
                'skill_level': skill_level,
                'genre': genre,
                'goals': goals
            },
            'timestamp': time.time()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_details = {
            'error': f'Analysis failed: {str(e)}',
            'analysis_id': analysis_id,
            'filename': filename,
            'traceback': traceback.format_exc() if app.debug else None
        }
        logger.error(f"❌ Analysis failed for {filename}: {e}")
        return jsonify(error_details), 500
        
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.route('/api/analyze', methods=['POST'])
def analyze_existing_file():
    """Analyze a file that's already been uploaded (if using cloud storage)"""
    data = request.get_json()
    
    if not data or 'file_url' not in data:
        return jsonify({'error': 'file_url required'}), 400
    
    # This would be implemented if using cloud storage
    return jsonify({'error': 'Cloud file analysis not implemented yet'}), 501

@app.route('/api/feedback/<analysis_id>', methods=['GET'])
def get_feedback(analysis_id):
    """Get feedback for a specific analysis (if stored)"""
    # This would be implemented with a database
    return jsonify({'error': 'Feedback retrieval not implemented yet'}), 501

@app.route('/api/formats')
def supported_formats():
    """Get list of supported audio formats"""
    return jsonify({
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'note': 'Upload files in high quality for best analysis results'
    })

def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'tolist'):  # Other numpy-like objects
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Other numpy-like scalars
        return obj.item()
    else:
        return obj

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
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong processing your request'
    }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

# Initialize logging
setup_logging()

if __name__ == '__main__':
    # Get port from environment (Railway provides this)
    port = int(os.environ.get('PORT', 8080))
    
    # Initialize the app
    initialize_app()
    
    logger.info("🚀 Starting BeatWizard Production API")
    logger.info(f"📱 Running on port: {port}")
    logger.info(f"🔧 Debug mode: {app.debug}")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )