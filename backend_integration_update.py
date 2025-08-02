#!/usr/bin/env python3
"""
Updated BeatWizard Backend with CORS Support for React Frontend
This file shows the modifications needed to your existing web_app_demo.py
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from pathlib import Path
import json
from werkzeug.utils import secure_filename
import time

# Import our BeatWizard system
from beatwizard import EnhancedAudioAnalyzer

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, origins=[
    "http://localhost:3000",  # React development server
    "https://your-domain.vercel.app",  # Production frontend (replace with your domain)
], supports_credentials=True)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-change-this'

# Initialize BeatWizard
analyzer = EnhancedAudioAnalyzer()

# Create upload directory
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for frontend"""
    return jsonify({
        'status': 'healthy',
        'service': 'BeatWizard API',
        'version': '1.0.0'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    # Get analysis settings from form data
    skill_level = request.form.get('skill_level', 'beginner')
    genre = request.form.get('genre', 'electronic')
    include_advanced = request.form.get('include_advanced_features', 'true').lower() == 'true'
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    analysis_id = str(uuid.uuid4())
    file_path = Path(app.config['UPLOAD_FOLDER']) / f"{analysis_id}_{filename}"
    file.save(file_path)
    
    try:
        print(f"🔄 Starting analysis for: {filename}")
        start_time = time.time()
        
        # Run BeatWizard analysis
        analysis_results = analyzer.analyze_track(str(file_path))
        
        # Generate intelligent feedback
        feedback = analyzer.generate_intelligent_feedback(
            analysis_results,
            skill_level=skill_level,
            genre=genre,
            goals=['streaming']
        )
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.1f}s")
        
        # Prepare results for frontend
        results = {
            'analysis_id': analysis_id,
            'filename': filename,
            'analysis_results': analysis_results,
            'intelligent_feedback': feedback,
            'analysis_time': analysis_time,
            'settings': {
                'skill_level': skill_level,
                'genre': genre,
                'include_advanced_features': include_advanced
            }
        }
        
        # Save results
        results_path = Path(app.config['UPLOAD_FOLDER']) / f"{analysis_id}_results.json"
        with open(results_path, 'w') as f:
            # Make results JSON serializable
            serializable_results = make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Clean up uploaded file
        file_path.unlink()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'redirect': f'/api/analysis/{analysis_id}',
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        # Clean up uploaded file on error
        if file_path.exists():
            file_path.unlink()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis_api(analysis_id):
    """API endpoint for analysis results"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    results_path = Path(app.config['UPLOAD_FOLDER']) / f"{analysis_id}_results.json"
    
    if not results_path.exists():
        return jsonify({'error': 'Analysis not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/api/feedback', methods=['POST'])
def generate_feedback():
    """Generate intelligent feedback for existing analysis"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        analysis_results = data.get('analysis_results')
        skill_level = data.get('skill_level', 'beginner')
        genre = data.get('genre', 'electronic')
        goals = data.get('goals', ['streaming'])
        
        if not analysis_results:
            return jsonify({'error': 'Analysis results required'}), 400
        
        # Generate intelligent feedback
        feedback = analyzer.generate_intelligent_feedback(
            analysis_results,
            skill_level=skill_level,
            genre=genre,
            goals=goals
        )
        
        return jsonify({
            'success': True,
            'feedback': feedback
        })
        
    except Exception as e:
        print(f"❌ Feedback generation failed: {e}")
        return jsonify({'error': f'Feedback generation failed: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

def make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj

if __name__ == '__main__':
    print("🚀 Starting BeatWizard API Server...")
    print("📱 React Frontend: http://localhost:3000")
    print("🔗 API Server: http://localhost:8080")
    print("🎵 Upload endpoint: http://localhost:8080/upload")
    print("📊 Health check: http://localhost:8080/health")
    
    # Install flask-cors if not installed
    try:
        import flask_cors
    except ImportError:
        print("⚠️  Installing flask-cors...")
        os.system("pip install flask-cors")
        print("✅ flask-cors installed")
    
    app.run(debug=True, host='0.0.0.0', port=8080)