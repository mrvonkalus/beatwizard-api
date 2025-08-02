#!/usr/bin/env python3
"""
BeatWizard Web App MVP Demo
Simple Flask web interface for BeatWizard analysis
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
from pathlib import Path
import json
from werkzeug.utils import secure_filename
import time

# Import our BeatWizard system
from beatwizard import EnhancedAudioAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-change-this'

# Initialize BeatWizard
analyzer = EnhancedAudioAnalyzer()

# Create upload directory
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/analyze')
def analyze_page():
    """Analysis upload page"""
    return render_template('analyze.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    analysis_id = str(uuid.uuid4())
    file_path = Path(app.config['UPLOAD_FOLDER']) / f"{analysis_id}_{filename}"
    file.save(file_path)
    
    try:
        # Run BeatWizard analysis
        print(f"🔄 Starting analysis for: {filename}")
        start_time = time.time()
        
        analysis_results = analyzer.analyze_track(str(file_path))
        
        # Generate intelligent feedback
        feedback = analyzer.generate_intelligent_feedback(
            analysis_results,
            skill_level=request.form.get('skill_level', 'beginner'),
            genre=request.form.get('genre', 'electronic'),
            goals=['streaming']
        )
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.1f}s")
        
        # Save results
        results = {
            'analysis_id': analysis_id,
            'filename': filename,
            'analysis_results': analysis_results,
            'intelligent_feedback': feedback,
            'analysis_time': analysis_time
        }
        
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
            'redirect': f'/results/{analysis_id}'
        })
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    """Display analysis results"""
    results_path = Path(app.config['UPLOAD_FOLDER']) / f"{analysis_id}_results.json"
    
    if not results_path.exists():
        return "Analysis not found", 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return render_template('results.html', results=results)

@app.route('/api/analysis/<analysis_id>')
def get_analysis_api(analysis_id):
    """API endpoint for analysis results"""
    results_path = Path(app.config['UPLOAD_FOLDER']) / f"{analysis_id}_results.json"
    
    if not results_path.exists():
        return jsonify({'error': 'Analysis not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/demo')
def demo():
    """Demo page showing example analysis"""
    # Load a sample analysis result
    sample_results = {
        'filename': 'sample_track.mp3',
        'analysis_time': 45.2,
        'intelligent_feedback': {
            'production_feedback': {
                'priority_issues': [
                    {
                        'element': 'kick',
                        'severity': 'high',
                        'issue': 'Your kick needs more punch and clarity',
                        'solution': 'Try a different kick sample with more 60-80Hz content',
                        'beginner_note': 'The kick is the foundation - it needs to be strong and clear'
                    }
                ],
                'quick_wins': [
                    {
                        'action': 'EQ adjustment',
                        'description': 'Simple EQ tweaks can dramatically improve your mix',
                        'steps': ['High-pass non-bass elements around 80Hz', 'Cut harsh frequencies around 3kHz']
                    }
                ]
            },
            'sound_selection': {
                'splice_pack_suggestions': [
                    'Splice - Modern Trap Kicks Vol. 3',
                    'KSHMR Kick Collection',
                    'Loopmasters - Deep House Kicks'
                ]
            },
            'overall_assessment': {
                'overall_rating': 'developing',
                'commercial_potential': 'moderate',
                'motivational_message': 'You\'re making solid progress! Focus on the kick and you\'ll see big improvements.'
            }
        }
    }
    
    return render_template('results.html', results=sample_results, is_demo=True)

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
    print("🚀 Starting BeatWizard Web App...")
    print("📱 Access at: http://localhost:8080")
    print("🎵 Upload tracks at: http://localhost:8080/analyze")
    print("🎮 See demo at: http://localhost:8080/demo")
    
    app.run(debug=True, host='0.0.0.0', port=8080)