#!/usr/bin/env python3
"""
BeatWizard Quick Start Example
Simple example showing basic usage of the enhanced audio analysis system
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from beatwizard import EnhancedAudioAnalyzer


def analyze_track(audio_file_path):
    """
    Analyze a single audio track with BeatWizard
    
    Args:
        audio_file_path: Path to the audio file
    """
    print(f"🎵 Analyzing: {Path(audio_file_path).name}")
    print("-" * 50)
    
    # Initialize the enhanced analyzer
    analyzer = EnhancedAudioAnalyzer()
    
    try:
        # Perform comprehensive analysis
        results = analyzer.analyze_track(audio_file_path)
        
        # Extract key metrics
        print("📊 Analysis Results:")
        print()
        
        # Tempo
        if 'tempo_analysis' in results:
            tempo_data = results['tempo_analysis']
            tempo = tempo_data.get('primary_tempo')
            confidence = tempo_data.get('confidence', 0.0)
            
            if tempo:
                print(f"🎼 Tempo: {tempo:.1f} BPM (confidence: {confidence:.2f})")
            else:
                print("🎼 Tempo: Could not detect")
        
        # Key
        if 'key_analysis' in results:
            key_data = results['key_analysis']
            key = key_data.get('primary_key')
            confidence = key_data.get('confidence', 0.0)
            
            if key:
                print(f"🎹 Key: {key} (confidence: {confidence:.2f})")
            else:
                print("🎹 Key: Could not detect")
        
        # Loudness
        if 'loudness_analysis' in results:
            loudness_data = results['loudness_analysis']
            lufs = loudness_data.get('integrated_loudness')
            
            if lufs and lufs > -100:
                print(f"🔊 Loudness: {lufs:.1f} LUFS")
                
                # Check platform compliance
                compliance = loudness_data.get('compliance_analysis', {})
                platform_compliance = compliance.get('platform_compliance', {})
                compliant_platforms = [p for p, data in platform_compliance.items() 
                                     if data.get('overall_compliant', False)]
                
                if compliant_platforms:
                    print(f"   ✅ Compliant with: {', '.join(compliant_platforms)}")
                else:
                    print("   ⚠️  Not compliant with streaming platforms")
            else:
                print("🔊 Loudness: Could not measure")
        
        # Overall Quality
        if 'overall_assessment' in results:
            assessment = results['overall_assessment']
            quality = assessment.get('overall_quality', 'unknown')
            mastering_ready = assessment.get('mastering_readiness', False)
            
            print(f"⭐ Overall Quality: {quality.title()}")
            print(f"🎛️  Mastering Ready: {'Yes' if mastering_ready else 'No'}")
        
        # Quick recommendations
        if 'professional_insights' in results:
            insights = results['professional_insights']
            
            # Show top 3 recommendations
            all_recommendations = []
            
            for category in ['mixing_suggestions', 'eq_recommendations', 'mastering_suggestions']:
                all_recommendations.extend(insights.get(category, []))
            
            if all_recommendations:
                print("\n💡 Top Recommendations:")
                for i, rec in enumerate(all_recommendations[:3], 1):
                    print(f"   {i}. {rec}")
        
        print("\n✅ Analysis completed successfully!")
        
        # Export summary
        summary = analyzer.get_analysis_summary(results)
        print(f"\n📄 Summary: {summary}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


def main():
    """Main function"""
    print("🎵 BeatWizard Quick Start Example")
    print("=================================")
    
    # Example usage - replace with your audio file path
    audio_file = "path/to/your/audio/file.wav"
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"\n⚠️  Audio file not found: {audio_file}")
        print("\nTo use this example:")
        print("1. Replace 'path/to/your/audio/file.wav' with your actual file path")
        print("2. Supported formats: .wav, .mp3, .flac, .m4a")
        print("3. Run: python example.py")
        print("\nExample:")
        print("   audio_file = 'data/test_tracks/my_song.wav'")
        return
    
    # Analyze the track
    results = analyze_track(audio_file)
    
    if results:
        print(f"\n🎉 Ready to use BeatWizard for professional audio analysis!")
        print("   Check out demo.py for a comprehensive demonstration")


if __name__ == "__main__":
    main()