#!/usr/bin/env python3
"""
BeatWizard Intelligent Feedback Demo
===================================

Advanced demo showcasing the new intelligent feedback capabilities:
- Sound selection analysis ("your kick is trash, try this...")
- Rhythmic pattern analysis and groove feedback
- Harmonic progression analysis and chord suggestions
- AI-powered contextual feedback for different skill levels
- Sample pack recommendations and Splice-style suggestions
- Genre-specific advice and learning paths

Perfect for producers at all levels - especially beginners needing guidance!
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from beatwizard import EnhancedAudioAnalyzer
from loguru import logger
import time


def print_section_header(title: str, subtitle: str = ""):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"   {title}")
    if subtitle:
        print(f"   {subtitle}")
    print("="*60)


def print_priority_feedback(feedback: Dict, title: str):
    """Print priority feedback in an eye-catching format"""
    print(f"\n🚨 {title.upper()}")
    print("-" * 40)
    
    priority_issues = feedback.get('production_feedback', {}).get('priority_issues', [])
    
    for issue in priority_issues[:3]:  # Show top 3 priority issues
        severity_emoji = "🔥" if issue.get('severity') == 'high' else "⚠️"
        print(f"\n{severity_emoji} {issue.get('element', '').upper()}: {issue.get('issue', '')}")
        print(f"   💡 FIX: {issue.get('solution', '')}")
        
        if issue.get('beginner_note'):
            print(f"   🎓 BEGINNER TIP: {issue.get('beginner_note', '')}")


def print_sound_selection_feedback(feedback: Dict):
    """Print detailed sound selection feedback"""
    sound_selection = feedback.get('sound_selection', {})
    element_feedback = sound_selection.get('element_feedback', {})
    
    print(f"\n🎵 SOUND SELECTION BREAKDOWN")
    print("-" * 40)
    
    for element, element_data in element_feedback.items():
        print(f"\n🔊 {element.upper()} ANALYSIS:")
        print(f"   Issue: {element_data.get('recommendation', 'No issues found')}")
        
        specific_advice = element_data.get('specific_advice', [])
        if specific_advice:
            print(f"   Specific fixes:")
            for advice in specific_advice[:3]:
                print(f"   • {advice}")
        
        # Show rookie mistakes if available
        rookie_mistakes = element_data.get('rookie_mistakes', [])
        if rookie_mistakes:
            print(f"   ❌ Common mistakes to avoid:")
            for mistake in rookie_mistakes[:2]:
                print(f"   • {mistake}")


def print_sample_suggestions(feedback: Dict):
    """Print sample pack and Splice suggestions"""
    sample_suggestions = feedback.get('sample_suggestions', {})
    splice_suggestions = feedback.get('sound_selection', {}).get('splice_pack_suggestions', [])
    
    print(f"\n🎛️ SAMPLE RECOMMENDATIONS")
    print("-" * 40)
    
    if splice_suggestions:
        print(f"\n📦 Splice Pack Suggestions:")
        for suggestion in splice_suggestions[:5]:
            print(f"   • {suggestion}")
    
    # Show sample pack recommendations by category
    for category, packs in sample_suggestions.items():
        if packs:
            category_name = category.replace('_', ' ').title()
            print(f"\n🎹 {category_name}:")
            for pack in packs[:3]:
                print(f"   • {pack}")


def print_creative_feedback(feedback: Dict):
    """Print creative suggestions and arrangement ideas"""
    creative = feedback.get('creative_suggestions', {})
    arrangement = feedback.get('arrangement', {})
    
    print(f"\n🎨 CREATIVE ENHANCEMENT SUGGESTIONS")
    print("-" * 40)
    
    # Arrangement suggestions
    arrangement_suggestions = arrangement.get('arrangement_suggestions', [])
    if arrangement_suggestions:
        print(f"\n🎼 Arrangement Ideas:")
        for suggestion in arrangement_suggestions[:3]:
            print(f"   • {suggestion}")
    
    # Sound design suggestions
    sound_design = creative.get('sound_design', [])
    if sound_design:
        print(f"\n🔧 Sound Design Tips:")
        for tip in sound_design[:3]:
            print(f"   • {tip}")
    
    # Genre-specific suggestions
    genre_specific = creative.get('genre_specific', [])
    if genre_specific:
        print(f"\n🎵 Genre-Specific Ideas:")
        for idea in genre_specific[:3]:
            print(f"   • {idea}")


def print_learning_path(feedback: Dict, skill_level: str):
    """Print personalized learning path"""
    learning_path = feedback.get('learning_path', {})
    
    print(f"\n📚 YOUR LEARNING PATH ({skill_level.upper()})")
    print("-" * 40)
    
    immediate_focus = learning_path.get('immediate_focus', [])
    if immediate_focus:
        print(f"\n🎯 Focus This Week:")
        for focus in immediate_focus:
            print(f"   • {focus}")
    
    next_month = learning_path.get('next_month', [])
    if next_month:
        print(f"\n📅 Next Month Goals:")
        for goal in next_month[:3]:
            print(f"   • {goal}")
    
    long_term = learning_path.get('long_term_goals', [])
    if long_term:
        print(f"\n🚀 Long-term Goals:")
        for goal in long_term[:2]:
            print(f"   • {goal}")


def print_reference_tracks(feedback: Dict):
    """Print reference track recommendations"""
    references = feedback.get('references', [])
    
    if references:
        print(f"\n🎧 REFERENCE TRACKS TO STUDY")
        print("-" * 40)
        
        for ref in references[:3]:
            print(f"\n🎵 {ref.get('artist', 'Unknown')} - \"{ref.get('track', 'Unknown')}\"")
            print(f"   Why: {ref.get('why', 'Great example track')}")
            
            focus_areas = ref.get('focus_areas', [])
            if focus_areas:
                print(f"   Focus on: {', '.join(focus_areas)}")


def print_overall_assessment(feedback: Dict, skill_level: str):
    """Print overall track assessment"""
    assessment = feedback.get('overall_assessment', {})
    
    print(f"\n📊 OVERALL TRACK ASSESSMENT")
    print("-" * 40)
    
    rating = assessment.get('overall_rating', 'unknown')
    commercial_potential = assessment.get('commercial_potential', 'unknown')
    
    # Rating with appropriate emoji
    rating_emojis = {
        'strong': '🔥',
        'developing': '📈',
        'needs_work': '🔧',
        'early_stage': '🌱'
    }
    
    rating_emoji = rating_emojis.get(rating, '❓')
    print(f"\n{rating_emoji} Overall Rating: {rating.replace('_', ' ').title()}")
    print(f"💰 Commercial Potential: {commercial_potential.replace('_', ' ').title()}")
    
    # Motivational message
    motivational = assessment.get('motivational_message', '')
    if motivational:
        print(f"\n💪 {motivational}")
    
    # Next steps
    next_steps = assessment.get('next_version_focus', [])
    if next_steps:
        print(f"\n🎯 For Your Next Version:")
        for step in next_steps[:3]:
            print(f"   • {step}")


def demonstrate_skill_level_differences(analyzer: EnhancedAudioAnalyzer, audio_file: str):
    """Demonstrate how feedback changes based on skill level"""
    print_section_header("SKILL-LEVEL ADAPTIVE FEEDBACK", "See how feedback changes for different producers")
    
    # Run analysis once
    print("🔄 Running complete analysis...")
    analysis_results = analyzer.analyze_track(audio_file)
    
    skill_levels = ['beginner', 'intermediate', 'advanced']
    
    for skill_level in skill_levels:
        print(f"\n" + "="*50)
        print(f"   🎓 FEEDBACK FOR {skill_level.upper()} PRODUCER")
        print("="*50)
        
        # Generate feedback for this skill level
        feedback = analyzer.generate_intelligent_feedback(
            analysis_results, 
            skill_level=skill_level,
            genre='electronic',
            goals=['streaming']
        )
        
        # Show different aspects of feedback
        print_priority_feedback(feedback, f"Priority Issues ({skill_level})")
        
        # Show learning path (different for each level)
        learning_path = feedback.get('learning_path', {})
        immediate_focus = learning_path.get('immediate_focus', [])
        
        print(f"\n📚 {skill_level.upper()} FOCUS AREAS:")
        for focus in immediate_focus[:2]:
            print(f"   • {focus}")
        
        # Show skill-specific tips
        production_feedback = feedback.get('production_feedback', {})
        detailed_feedback = production_feedback.get('detailed_feedback', [])
        
        if detailed_feedback:
            print(f"\n💡 {skill_level.upper()} TIPS:")
            for tip in detailed_feedback[:2]:
                print(f"   • {tip}")


def demonstrate_genre_specific_feedback(analyzer: EnhancedAudioAnalyzer, audio_file: str):
    """Demonstrate genre-specific feedback"""
    print_section_header("GENRE-SPECIFIC FEEDBACK", "Tailored advice for different music styles")
    
    # Run analysis once
    analysis_results = analyzer.analyze_track(audio_file)
    
    genres = ['house', 'trap', 'techno', 'dubstep']
    
    for genre in genres:
        print(f"\n📻 {genre.upper()} GENRE FEEDBACK:")
        print("-" * 30)
        
        feedback = analyzer.generate_intelligent_feedback(
            analysis_results,
            skill_level='beginner',
            genre=genre,
            goals=['streaming', 'club']
        )
        
        # Show genre-specific creative suggestions
        creative_suggestions = feedback.get('creative_suggestions', {})
        genre_specific = creative_suggestions.get('genre_specific', [])
        
        for suggestion in genre_specific[:2]:
            print(f"   • {suggestion}")


def run_comprehensive_feedback_demo():
    """Run comprehensive intelligent feedback demonstration"""
    print_section_header("🎵 BEATWIZARD INTELLIGENT FEEDBACK SYSTEM", "Advanced AI-Powered Producer Guidance")
    
    print("""
🚀 Welcome to the Future of Music Production Feedback!
======================================================

This demo showcases BeatWizard's new intelligent feedback system:

✅ Sound Selection Analysis - "Your kick is weak, try these samples..."
✅ Rhythmic Pattern Analysis - Groove and timing feedback  
✅ Harmonic Progression Analysis - Chord and melody suggestions
✅ AI-Powered Contextual Advice - Tailored to your skill level
✅ Sample Pack Recommendations - Splice-style suggestions
✅ Genre-Specific Guidance - House, Trap, Techno, etc.
✅ Learning Path Generation - Personalized improvement roadmap

Perfect for:
🎹 Beginners learning the fundamentals
🎛️ Intermediate producers refining their skills  
🎚️ Advanced producers seeking creative inspiration
🎵 Anyone wanting specific, actionable feedback
    """)
    
    # Initialize analyzer
    print("\n🔧 Initializing BeatWizard Enhanced Analysis System...")
    analyzer = EnhancedAudioAnalyzer()
    
    # Find audio files
    test_tracks_dir = Path("data/test_tracks")
    audio_files = []
    
    # Look for audio files
    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        audio_files.extend(test_tracks_dir.glob(ext))
    
    if not audio_files:
        print(f"""
⚠️  No audio files found in {test_tracks_dir}
   
   Please add an audio file to test the intelligent feedback system.
   Supported formats: .mp3, .wav, .flac, .m4a
   
   Try adding:
   • A track you're currently working on
   • A reference track in your genre
   • Any music file to see the feedback in action
        """)
        return
    
    # Use the first audio file found
    audio_file = str(audio_files[0])
    print(f"\n🎵 Analyzing: {Path(audio_file).name}")
    
    # Run comprehensive analysis with intelligent feedback
    print("\n🔄 Running complete enhanced analysis...")
    start_time = time.time()
    
    analysis_results = analyzer.analyze_track(audio_file)
    
    analysis_time = time.time() - start_time
    print(f"✅ Analysis completed in {analysis_time:.1f} seconds")
    
    # Generate intelligent feedback for a beginner producer
    print("\n🤖 Generating intelligent feedback...")
    feedback = analyzer.generate_intelligent_feedback(
        analysis_results,
        skill_level='beginner',
        genre='electronic',
        goals=['streaming', 'club']
    )
    
    # Display comprehensive feedback
    print_section_header("🎯 PRIORITY ISSUES", "Fix these first for maximum impact")
    print_priority_feedback(feedback, "High Priority Fixes")
    
    print_section_header("🎵 SOUND SELECTION FEEDBACK", "Specific advice on your samples and sounds")
    print_sound_selection_feedback(feedback)
    
    print_section_header("📦 SAMPLE RECOMMENDATIONS", "Upgrade your sound library")
    print_sample_suggestions(feedback)
    
    print_section_header("🎨 CREATIVE SUGGESTIONS", "Take your track to the next level")
    print_creative_feedback(feedback)
    
    print_section_header("📚 PERSONALIZED LEARNING PATH", "Your roadmap to improvement")
    print_learning_path(feedback, 'beginner')
    
    print_section_header("🎧 REFERENCE TRACKS", "Study these for inspiration")
    print_reference_tracks(feedback)
    
    print_section_header("📊 OVERALL ASSESSMENT", "Your track's current status")
    print_overall_assessment(feedback, 'beginner')
    
    # Demonstrate skill level differences
    print("\n" + "🎓" * 60)
    demonstrate_skill_level_differences(analyzer, audio_file)
    
    # Demonstrate genre-specific feedback
    print("\n" + "🎵" * 60)
    demonstrate_genre_specific_feedback(analyzer, audio_file)
    
    print_section_header("🎉 DEMO COMPLETED", "BeatWizard Intelligent Feedback System")
    
    print(f"""
🚀 What You Just Experienced:
============================

✅ Advanced sound selection analysis with specific sample recommendations
✅ Rhythmic and harmonic analysis for better arrangements
✅ AI-powered feedback tailored to skill level and genre  
✅ Actionable advice that sounds like it's from an experienced producer
✅ Sample pack suggestions (Splice-style recommendations)
✅ Personalized learning paths for continuous improvement
✅ Reference track recommendations for inspiration

🎯 This is exactly what young producers need:
   • Specific, actionable feedback instead of generic advice
   • Learning guidance appropriate for their skill level  
   • Sample recommendations to improve their sound library
   • Creative suggestions to spark new ideas
   • Technical guidance to improve their mixing

💡 Your BeatWizard system now provides:
   • Professional-grade analysis capabilities
   • Intelligent, contextual feedback generation
   • Wide range of skill level support
   • Genre-specific guidance
   • Sample library integration potential
   • Scalable architecture for thousands of users

🎵 Ready to help producers at every level create better music!
    """)


if __name__ == "__main__":
    try:
        run_comprehensive_feedback_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("\n🔧 Check that all dependencies are installed and audio files are present.")