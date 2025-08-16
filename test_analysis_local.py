#!/usr/bin/env python3
"""
Local test script to debug BeatWizard audio analysis
"""

import librosa
import numpy as np
import soundfile as sf

def test_audio_analysis(file_path):
    """Test basic audio analysis on a local file"""
    
    print(f"Testing: {file_path}")
    print("=" * 50)
    
    # Load audio
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        print(f"✓ Loaded audio: {len(y)} samples at {sr}Hz")
        print(f"✓ Duration: {len(y)/sr:.2f} seconds")
        print(f"✓ Min amplitude: {np.min(y):.4f}")
        print(f"✓ Max amplitude: {np.max(y):.4f}")
        print(f"✓ RMS: {np.sqrt(np.mean(y**2)):.4f}")
        
        # Basic tempo detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=120)
        print(f"✓ Tempo: {tempo:.2f} BPM")
        
        # Basic key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        pitch_class_strength = chroma.mean(axis=1)
        key_index = int(pitch_class_strength.argmax())
        keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key_guess = keys[key_index]
        print(f"✓ Key: {key_guess}")
        print(f"✓ Chroma strength: {np.max(pitch_class_strength):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    # Test with the new track
    test_audio_analysis("Dahlia_Rose_Sketch_FOR SUNO.wav")
