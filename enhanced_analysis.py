#!/usr/bin/env python3
"""
Enhanced BeatWizard Analysis - Shows all available audio metrics
"""

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

def enhanced_audio_analysis(file_path):
    """Comprehensive audio analysis showing all available metrics"""
    
    print(f"🎵 Enhanced BeatWizard Analysis: {file_path}")
    print("=" * 60)
    
    # Load audio
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    duration = len(y) / sr
    
    print(f"📊 BASIC AUDIO INFO")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Total Samples: {len(y):,}")
    print(f"   File Size: {len(y) * 2 / 1024 / 1024:.1f} MB (estimated)")
    
    print(f"\n🔊 AMPLITUDE ANALYSIS")
    print(f"   Peak Level: {np.max(np.abs(y)):.4f} ({20*np.log10(np.max(np.abs(y))):.1f} dB)")
    print(f"   RMS Level: {np.sqrt(np.mean(y**2)):.4f} ({20*np.log10(np.sqrt(np.mean(y**2))):.1f} dB)")
    print(f"   Dynamic Range: {20*np.log10(np.max(np.abs(y))/np.sqrt(np.mean(y**2))):.1f} dB")
    print(f"   Crest Factor: {np.max(np.abs(y))/np.sqrt(np.mean(y**2)):.2f}")
    
    print(f"\n🎼 TEMPO & RHYTHM")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=120)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"   Tempo: {tempo:.2f} BPM")
    print(f"   Beat Count: {len(beats)}")
    print(f"   Beat Interval: {60/tempo:.2f} seconds")
    
    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"   Onset Count: {len(onset_frames)}")
    print(f"   Onset Rate: {len(onset_frames)/duration:.1f} per second")
    
    print(f"\n🎵 HARMONIC ANALYSIS")
    # Key detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    pitch_class_strength = chroma.mean(axis=1)
    key_index = int(pitch_class_strength.argmax())
    keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_guess = keys[key_index]
    print(f"   Key: {key_guess}")
    print(f"   Chroma Strength: {np.max(pitch_class_strength):.4f}")
    
    # Harmonic content
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
    print(f"   Harmonic vs Percussive: {harmonic_ratio:.1%} harmonic")
    
    print(f"\n📈 FREQUENCY ANALYSIS")
    # Spectral features
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    print(f"   Spectral Centroid: {np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)):.1f} Hz")
    print(f"   Spectral Bandwidth: {np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)):.1f} Hz")
    print(f"   Spectral Rolloff: {np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)):.1f} Hz")
    
    # Frequency bands
    freqs = librosa.fft_frequencies(sr=sr)
    spec = np.abs(librosa.stft(y))
    low_freq_energy = np.sum(spec[freqs < 250, :]) / np.sum(spec)
    mid_freq_energy = np.sum(spec[(freqs >= 250) & (freqs < 2000), :]) / np.sum(spec)
    high_freq_energy = np.sum(spec[freqs >= 2000, :]) / np.sum(spec)
    
    print(f"   Low Freq (0-250Hz): {low_freq_energy:.1%}")
    print(f"   Mid Freq (250-2kHz): {mid_freq_energy:.1%}")
    print(f"   High Freq (2kHz+): {high_freq_energy:.1%}")
    
    print(f"\n🔊 LOUDNESS ANALYSIS")
    # LUFS measurement
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y)
    print(f"   LUFS: {lufs:.1f}")
    
    # Loudness range (simplified)
    print(f"   Loudness Range: Calculated from LUFS")
    
    print(f"\n🎚️ MIXING CHARACTERISTICS")
    # Stereo analysis (if available)
    try:
        y_stereo, sr_stereo = librosa.load(file_path, sr=sr, mono=False)
        if y_stereo.shape[0] == 2:
            left, right = y_stereo[0], y_stereo[1]
            stereo_width = np.mean(np.abs(left - right) / (np.abs(left) + np.abs(right) + 1e-8))
            print(f"   Stereo Width: {stereo_width:.3f}")
        else:
            print(f"   Stereo Width: Mono file")
    except:
        print(f"   Stereo Width: Could not analyze")
    
    print(f"\n🎯 QUALITY METRICS")
    # Signal quality
    snr_estimate = 20 * np.log10(np.sqrt(np.mean(y**2)) / (np.std(y - np.mean(y))))
    print(f"   Signal Quality: {snr_estimate:.1f} dB (estimated)")
    
    # Zero crossing rate (complexity)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    print(f"   Complexity: {zcr:.3f} (zero crossing rate)")
    
    print("=" * 60)
    return {
        'tempo': tempo,
        'key': key_guess,
        'duration': duration,
        'lufs': lufs,
        'harmonic_ratio': harmonic_ratio
    }

if __name__ == "__main__":
    enhanced_audio_analysis("Dahlia_Rose_Sketch_FOR SUNO.wav")
