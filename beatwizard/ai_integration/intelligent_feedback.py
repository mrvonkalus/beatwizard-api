"""
Intelligent Feedback Generator - AI-powered music production feedback
Generates contextual, skill-level appropriate feedback for producers
"""

import json
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config.settings import ai_settings


class IntelligentFeedbackGenerator:
    """
    Generate intelligent, contextual feedback for music producers
    Combines technical analysis with AI-powered insights
    """
    
    def __init__(self):
        """Initialize the feedback generator"""
        self.openai_available = OPENAI_AVAILABLE and ai_settings.OPENAI_API_KEY is not None
        
        if self.openai_available:
            openai.api_key = ai_settings.OPENAI_API_KEY
            logger.info("OpenAI integration enabled")
        else:
            logger.warning("OpenAI integration not available - using rule-based feedback")
        
        # Load sample libraries and reference database
        self.sample_library = self._load_sample_library()
        self.reference_tracks = self._load_reference_tracks()
        
        logger.debug("IntelligentFeedbackGenerator initialized")
    
    def generate_comprehensive_feedback(self, 
                                      analysis_results: Dict[str, any],
                                      skill_level: str = 'beginner',
                                      genre: Optional[str] = None,
                                      goals: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Generate comprehensive, intelligent feedback
        
        Args:
            analysis_results: Complete analysis results from BeatWizard
            skill_level: Producer skill level ('beginner', 'intermediate', 'advanced')
            genre: Target genre (if known)
            goals: Production goals (e.g., ['streaming', 'club', 'radio'])
            
        Returns:
            Dictionary with comprehensive feedback
        """
        logger.info(f"Generating feedback for {skill_level} level producer")
        
        # Extract key information from analysis
        feedback_context = self._extract_feedback_context(analysis_results, genre, goals)
        
        # Generate different types of feedback
        feedback = {}
        
        # Core production feedback
        feedback['production_feedback'] = self._generate_production_feedback(feedback_context, skill_level)
        
        # Sound selection feedback
        feedback['sound_selection'] = self._generate_sound_selection_feedback(feedback_context, skill_level)
        
        # Arrangement and structure feedback
        feedback['arrangement'] = self._generate_arrangement_feedback(feedback_context, skill_level)
        
        # Technical feedback (mixing/mastering)
        feedback['technical'] = self._generate_technical_feedback(feedback_context, skill_level)
        
        # Creative suggestions
        feedback['creative_suggestions'] = self._generate_creative_suggestions(feedback_context, skill_level, genre)
        
        # Next steps and learning path
        feedback['learning_path'] = self._generate_learning_path(feedback_context, skill_level)
        
        # Reference recommendations
        feedback['references'] = self._recommend_reference_tracks(feedback_context, genre)
        
        # Sample pack suggestions
        feedback['sample_suggestions'] = self._suggest_sample_packs(feedback_context, genre)
        
        # Overall assessment
        feedback['overall_assessment'] = self._generate_overall_assessment(feedback_context, skill_level)
        
        return feedback
    
    def _extract_feedback_context(self, analysis_results: Dict[str, any], genre: Optional[str], goals: Optional[List[str]]) -> Dict[str, any]:
        """Extract relevant context for feedback generation"""
        context = {
            'genre': genre or 'electronic',
            'goals': goals or ['streaming'],
            'skill_indicators': {},
            'main_issues': [],
            'strengths': [],
            'technical_metrics': {}
        }
        
        # Extract overall assessment
        overall = analysis_results.get('overall_assessment', {})
        context['overall_quality'] = overall.get('overall_quality', 'unknown')
        context['mastering_ready'] = overall.get('mastering_readiness', False)
        context['commercial_ready'] = overall.get('commercial_readiness', False)
        
        # Extract key metrics
        tempo_analysis = analysis_results.get('tempo_analysis', {})
        context['tempo'] = tempo_analysis.get('primary_tempo')
        context['tempo_confidence'] = tempo_analysis.get('confidence', 0.0)
        
        key_analysis = analysis_results.get('key_analysis', {})
        context['key'] = key_analysis.get('primary_key')
        context['key_confidence'] = key_analysis.get('confidence', 0.0)
        
        loudness_analysis = analysis_results.get('loudness_analysis', {})
        context['lufs'] = loudness_analysis.get('integrated_loudness')
        context['dynamic_range'] = loudness_analysis.get('dynamic_range_analysis', {}).get('dynamic_range_quality', 'unknown')
        
        # Extract technical metrics
        frequency_analysis = analysis_results.get('frequency_analysis', {})
        context['frequency_balance'] = frequency_analysis.get('overall_assessment', {}).get('quality_rating', 'unknown')
        
        stereo_analysis = analysis_results.get('stereo_analysis', {})
        context['stereo_quality'] = stereo_analysis.get('overall_assessment', {}).get('overall_quality', 'unknown')
        
        # Extract sound selection results if available
        if 'sound_selection_analysis' in analysis_results:
            sound_selection = analysis_results['sound_selection_analysis']
            context['sound_selection_quality'] = sound_selection.get('overall_sound_selection', {}).get('overall_quality', 'unknown')
            context['element_issues'] = self._extract_element_issues(sound_selection)
        
        # Extract rhythm analysis if available
        if 'rhythm_analysis' in analysis_results:
            rhythm_analysis = analysis_results['rhythm_analysis']
            context['rhythm_quality'] = rhythm_analysis.get('overall_rhythm', {}).get('overall_quality', 'unknown')
            context['groove_issues'] = rhythm_analysis.get('overall_rhythm', {}).get('weaknesses', [])
        
        # Extract harmony analysis if available
        if 'harmony_analysis' in analysis_results:
            harmony_analysis = analysis_results['harmony_analysis']
            context['harmony_quality'] = harmony_analysis.get('overall_harmony', {}).get('overall_quality', 'unknown')
            context['chord_issues'] = harmony_analysis.get('chord_analysis', {}).get('issues', [])
        
        # Determine skill level indicators
        context['skill_indicators'] = self._assess_skill_indicators(context)
        
        return context
    
    def _extract_element_issues(self, sound_selection: Dict[str, any]) -> Dict[str, List[str]]:
        """Extract issues with individual elements"""
        element_issues = {}
        
        elements = ['kick_analysis', 'snare_analysis', 'bass_analysis', 'melody_analysis']
        
        for element in elements:
            if element in sound_selection:
                element_name = element.replace('_analysis', '')
                element_data = sound_selection[element]
                
                if element_data.get('quality') in ['poor', 'weak']:
                    element_issues[element_name] = element_data.get('issues', [])
        
        return element_issues
    
    def _assess_skill_indicators(self, context: Dict[str, any]) -> Dict[str, any]:
        """Assess producer skill level based on technical metrics"""
        indicators = {
            'technical_competency': 'beginner',
            'creative_sophistication': 'beginner',
            'production_experience': 'beginner'
        }
        
        # Technical competency indicators
        technical_score = 0
        
        if context.get('lufs') and -18 <= context['lufs'] <= -6:
            technical_score += 1
        
        if context.get('frequency_balance') in ['good', 'excellent']:
            technical_score += 1
        
        if context.get('stereo_quality') in ['good', 'excellent']:
            technical_score += 1
        
        if context.get('mastering_ready'):
            technical_score += 1
        
        if technical_score >= 3:
            indicators['technical_competency'] = 'advanced'
        elif technical_score >= 2:
            indicators['technical_competency'] = 'intermediate'
        
        # Creative sophistication indicators
        creative_score = 0
        
        if context.get('harmony_quality') in ['good', 'excellent']:
            creative_score += 1
        
        if context.get('rhythm_quality') in ['good', 'excellent']:
            creative_score += 1
        
        if context.get('tempo_confidence', 0) > 0.8:
            creative_score += 1
        
        if creative_score >= 2:
            indicators['creative_sophistication'] = 'intermediate'
        if creative_score >= 3:
            indicators['creative_sophistication'] = 'advanced'
        
        return indicators
    
    def _generate_production_feedback(self, context: Dict[str, any], skill_level: str) -> Dict[str, any]:
        """Generate core production feedback"""
        feedback = {
            'priority_issues': [],
            'quick_wins': [],
            'detailed_feedback': [],
            'encouragement': []
        }
        
        # Identify priority issues
        element_issues = context.get('element_issues', {})
        
        for element, issues in element_issues.items():
            if element == 'kick' and issues:
                feedback['priority_issues'].append({
                    'element': 'kick',
                    'severity': 'high',
                    'issue': 'Your kick needs work - it\'s the foundation of your track',
                    'solution': 'Try a different kick sample with more punch and clarity',
                    'beginner_note': 'The kick is like the heartbeat of your song - it needs to be strong and clear'
                })
            
            elif element == 'bass' and issues:
                feedback['priority_issues'].append({
                    'element': 'bass',
                    'severity': 'high',
                    'issue': 'Bass lacks presence and definition',
                    'solution': 'Choose a bass with more fundamental frequency content (60-150Hz)',
                    'beginner_note': 'Good bass should be felt as much as heard - it fills out the low end'
                })
        
        # Technical issues
        if context.get('lufs') and context['lufs'] < -25:
            feedback['priority_issues'].append({
                'element': 'levels',
                'severity': 'high',
                'issue': 'Track is way too quiet for streaming platforms',
                'solution': f'Increase overall level by {abs(context["lufs"] + 14):.1f} dB',
                'beginner_note': 'Streaming platforms expect tracks around -14 LUFS for optimal playback'
            })
        
        # Quick wins
        if context.get('frequency_balance') == 'needs_improvement':
            feedback['quick_wins'].append({
                'action': 'EQ adjustment',
                'description': 'Simple EQ tweaks can dramatically improve your mix',
                'steps': [
                    'High-pass everything except kick and bass around 80-100Hz',
                    'Cut harsh frequencies around 2-5kHz if the mix sounds harsh',
                    'Add gentle high-shelf above 10kHz for air and sparkle'
                ]
            })
        
        # Skill-level specific feedback
        if skill_level == 'beginner':
            feedback['detailed_feedback'].extend([
                'Focus on getting your kick and snare to hit hard first',
                'Don\'t worry about complex processing - good samples are 80% of the battle',
                'Use reference tracks constantly - A/B your mix with professional songs',
                'Less is more - don\'t add too many elements at once'
            ])
            
            feedback['encouragement'].extend([
                'Every producer started where you are - keep experimenting!',
                'The fact that you\'re analyzing your tracks shows you\'re serious about improving',
                'Focus on one element at a time - you\'ll see faster progress'
            ])
        
        elif skill_level == 'intermediate':
            feedback['detailed_feedback'].extend([
                'Your technical foundation is solid - now focus on creative elements',
                'Consider the emotional impact of your sound choices',
                'Experiment with automation to add movement and interest',
                'Start thinking about the frequency spectrum as real estate - every element needs its space'
            ])
        
        return feedback
    
    def _generate_sound_selection_feedback(self, context: Dict[str, any], skill_level: str) -> Dict[str, any]:
        """Generate specific sound selection feedback"""
        feedback = {
            'element_feedback': {},
            'sample_recommendations': {},
            'layering_suggestions': [],
            'splice_pack_suggestions': []
        }
        
        element_issues = context.get('element_issues', {})
        
        # Kick feedback
        if 'kick' in element_issues:
            feedback['element_feedback']['kick'] = {
                'current_issues': element_issues['kick'],
                'recommendation': 'Your kick is holding back the whole track',
                'specific_advice': [
                    'Look for kicks with a solid thump around 60-80Hz',
                    'Make sure it has enough punch in the 100-200Hz range',
                    'Avoid kicks that are all sub-bass with no definition'
                ],
                'rookie_mistakes': [
                    'Using the default kick from your DAW',
                    'Choosing kicks that are too quiet or weak',
                    'Not considering how the kick fits with the bass'
                ]
            }
            
            # Genre-specific kick suggestions
            genre = context.get('genre', 'electronic')
            feedback['sample_recommendations']['kick'] = self._get_genre_kick_suggestions(genre)
        
        # Bass feedback
        if 'bass' in element_issues:
            feedback['element_feedback']['bass'] = {
                'current_issues': element_issues['bass'],
                'recommendation': 'Your bass needs more character and presence',
                'specific_advice': [
                    'Layer a sub bass (sine wave) with a mid bass (saw/square)',
                    'High-pass your bass around 30-40Hz to clean up the sub',
                    'Use compression to make the bass more consistent'
                ],
                'common_fixes': [
                    'Add harmonic distortion for character',
                    'Use sidechain compression with the kick',
                    'EQ to carve out space for the kick'
                ]
            }
        
        # Snare feedback
        if 'snare' in element_issues:
            feedback['element_feedback']['snare'] = {
                'recommendation': 'Your snare needs more crack and presence',
                'layering_tip': 'Layer a punchy snare with a clap for more impact',
                'eq_suggestion': 'Boost around 200Hz for body and 5kHz for crack'
            }
        
        # Sample pack suggestions based on issues
        feedback['splice_pack_suggestions'] = self._generate_sample_pack_suggestions(element_issues, context.get('genre'))
        
        return feedback
    
    def _generate_arrangement_feedback(self, context: Dict[str, any], skill_level: str) -> Dict[str, any]:
        """Generate arrangement and structure feedback"""
        feedback = {
            'structure_analysis': {},
            'arrangement_suggestions': [],
            'energy_curve': {},
            'transition_ideas': []
        }
        
        # Analyze rhythm and harmony for arrangement feedback
        rhythm_quality = context.get('rhythm_quality', 'unknown')
        harmony_quality = context.get('harmony_quality', 'unknown')
        
        if rhythm_quality == 'poor':
            feedback['arrangement_suggestions'].extend([
                'Your track needs more rhythmic variation to keep listeners engaged',
                'Try adding fills every 8 or 16 bars',
                'Consider using breakdowns to create dynamic contrast'
            ])
        
        if harmony_quality == 'poor':
            feedback['arrangement_suggestions'].extend([
                'Add more harmonic movement with chord changes',
                'Try the classic I-V-vi-IV progression',
                'Use simple triads before getting into complex chords'
            ])
        
        # Skill-level specific arrangement advice
        if skill_level == 'beginner':
            feedback['arrangement_suggestions'].extend([
                'Start with a simple 8-bar loop and build from there',
                'Follow the intro-verse-chorus-verse-chorus-bridge-chorus structure',
                'Add one new element every 8-16 bars to build energy'
            ])
        
        return feedback
    
    def _generate_technical_feedback(self, context: Dict[str, any], skill_level: str) -> Dict[str, any]:
        """Generate technical mixing/mastering feedback"""
        feedback = {
            'mixing_priority': [],
            'mastering_readiness': {},
            'streaming_compliance': {},
            'technical_improvements': []
        }
        
        # Loudness issues
        lufs = context.get('lufs')
        if lufs and lufs < -20:
            feedback['mixing_priority'].append({
                'issue': 'Track is too quiet',
                'solution': f'Increase overall level by {abs(lufs + 14):.1f} dB',
                'importance': 'critical',
                'why': 'Streaming platforms normalize to around -14 LUFS'
            })
        
        # Frequency balance issues
        freq_balance = context.get('frequency_balance')
        if freq_balance in ['poor', 'needs_improvement']:
            feedback['mixing_priority'].append({
                'issue': 'Frequency balance needs work',
                'solution': 'Use EQ to balance the frequency spectrum',
                'importance': 'high',
                'specific_actions': [
                    'High-pass non-bass elements around 80-100Hz',
                    'Check for harsh frequencies around 2-5kHz',
                    'Add gentle high-shelf for air and presence'
                ]
            })
        
        # Mastering readiness
        mastering_ready = context.get('mastering_ready', False)
        feedback['mastering_readiness'] = {
            'ready': mastering_ready,
            'blockers': [] if mastering_ready else ['Fix mixing issues first'],
            'next_steps': ['Send to mastering engineer'] if mastering_ready else ['Continue mixing work']
        }
        
        return feedback
    
    def _generate_creative_suggestions(self, context: Dict[str, any], skill_level: str, genre: Optional[str]) -> Dict[str, any]:
        """Generate creative enhancement suggestions"""
        suggestions = {
            'sound_design': [],
            'arrangement_ideas': [],
            'genre_specific': [],
            'experimental': []
        }
        
        # Genre-specific suggestions
        if genre:
            suggestions['genre_specific'] = self._get_genre_specific_suggestions(genre, context)
        
        # General creative suggestions based on skill level
        if skill_level == 'beginner':
            suggestions['sound_design'].extend([
                'Try layering different sounds to create unique textures',
                'Use reverb and delay to add space and depth',
                'Experiment with filtering to create movement'
            ])
            
            suggestions['arrangement_ideas'].extend([
                'Add a simple breakdown by removing elements',
                'Try doubling your melody an octave higher for thickness',
                'Use panning to create width and interest'
            ])
        
        return suggestions
    
    def _generate_learning_path(self, context: Dict[str, any], skill_level: str) -> Dict[str, any]:
        """Generate personalized learning path"""
        learning_path = {
            'immediate_focus': [],
            'next_month': [],
            'long_term_goals': [],
            'recommended_resources': []
        }
        
        # Determine immediate focus based on biggest issues
        element_issues = context.get('element_issues', {})
        
        if 'kick' in element_issues:
            learning_path['immediate_focus'].append('Master kick drum selection and processing')
        
        if context.get('lufs', 0) < -20:
            learning_path['immediate_focus'].append('Learn about loudness standards and gain staging')
        
        if context.get('frequency_balance') == 'poor':
            learning_path['immediate_focus'].append('Study EQ fundamentals and frequency balance')
        
        # Skill-level specific learning paths
        if skill_level == 'beginner':
            learning_path['next_month'].extend([
                'Learn your DAW\'s stock plugins thoroughly',
                'Study song structure in your favorite genre',
                'Practice basic mixing techniques (EQ, compression)',
                'Build a library of go-to samples and presets'
            ])
            
            learning_path['long_term_goals'].extend([
                'Complete 10 full tracks (quantity over quality at first)',
                'Develop your unique sound and style',
                'Learn advanced synthesis and sound design',
                'Study music theory and harmony'
            ])
        
        return learning_path
    
    def _recommend_reference_tracks(self, context: Dict[str, any], genre: Optional[str]) -> List[Dict[str, any]]:
        """Recommend reference tracks for comparison"""
        references = []
        
        # Get genre-specific references
        if genre in self.reference_tracks:
            genre_refs = self.reference_tracks[genre]
            references.extend(genre_refs[:3])
        
        # Add general high-quality references
        references.extend([
            {
                'artist': 'Flume',
                'track': 'Never Be Like You',
                'why': 'Excellent example of modern electronic production with great sound selection',
                'focus_areas': ['kick selection', 'frequency balance', 'creative arrangements']
            },
            {
                'artist': 'Skrillex',
                'track': 'Scary Monsters and Nice Sprites',
                'why': 'Iconic bass sound design and energy building',
                'focus_areas': ['bass design', 'arrangement', 'dynamics']
            }
        ])
        
        return references[:5]
    
    def _suggest_sample_packs(self, context: Dict[str, any], genre: Optional[str]) -> Dict[str, List[str]]:
        """Suggest specific sample packs based on analysis"""
        suggestions = {
            'kick_packs': [],
            'full_construction_kits': [],
            'one_shots': [],
            'loops': []
        }
        
        element_issues = context.get('element_issues', {})
        
        # Kick-specific suggestions
        if 'kick' in element_issues:
            suggestions['kick_packs'].extend([
                'Splice - Modern Trap Kicks Vol. 3',
                'Loopmasters - Deep House Kicks',
                'Sample Magic - Techno Kicks Collection',
                'KSHMR - Kick Collection'
            ])
        
        # Genre-specific pack suggestions
        if genre:
            genre_packs = self.sample_library.get(genre, {})
            for pack_type, packs in genre_packs.items():
                if pack_type in suggestions:
                    suggestions[pack_type].extend(packs[:3])
        
        return suggestions
    
    def _generate_overall_assessment(self, context: Dict[str, any], skill_level: str) -> Dict[str, any]:
        """Generate overall track assessment and summary"""
        assessment = {
            'overall_rating': 'unknown',
            'strengths': [],
            'main_weaknesses': [],
            'commercial_potential': 'unknown',
            'next_version_focus': [],
            'motivational_message': ''
        }
        
        # Determine overall rating
        quality_indicators = [
            context.get('overall_quality', 'poor'),
            context.get('sound_selection_quality', 'poor'),
            context.get('rhythm_quality', 'poor'),
            context.get('harmony_quality', 'poor')
        ]
        
        good_indicators = sum(1 for q in quality_indicators if q in ['good', 'excellent'])
        
        if good_indicators >= 3:
            assessment['overall_rating'] = 'strong'
        elif good_indicators >= 2:
            assessment['overall_rating'] = 'developing'
        elif good_indicators >= 1:
            assessment['overall_rating'] = 'needs_work'
        else:
            assessment['overall_rating'] = 'early_stage'
        
        # Generate motivational message based on skill level and progress
        if skill_level == 'beginner':
            if assessment['overall_rating'] in ['developing', 'strong']:
                assessment['motivational_message'] = "You're making solid progress! Your fundamentals are coming together nicely."
            else:
                assessment['motivational_message'] = "Every producer starts here - focus on the basics and you'll see rapid improvement!"
        
        # Commercial potential
        if context.get('mastering_ready') and context.get('commercial_ready'):
            assessment['commercial_potential'] = 'high'
        elif assessment['overall_rating'] in ['developing', 'strong']:
            assessment['commercial_potential'] = 'moderate'
        else:
            assessment['commercial_potential'] = 'needs_development'
        
        return assessment
    
    def _get_genre_kick_suggestions(self, genre: str) -> List[str]:
        """Get genre-specific kick suggestions"""
        genre_kicks = {
            'house': ['Deep house kicks', '4/4 kicks', 'Classic house kicks'],
            'trap': ['808 kicks', 'Trap kicks', 'Heavy sub kicks'],
            'techno': ['Techno kicks', 'Industrial kicks', 'Minimal kicks'],
            'dubstep': ['Dubstep kicks', 'Heavy kicks', 'Distorted kicks'],
            'pop': ['Pop kicks', 'Radio-ready kicks', 'Punchy kicks']
        }
        
        return genre_kicks.get(genre.lower(), ['Balanced kicks', 'All-purpose kicks', 'Studio kicks'])
    
    def _generate_sample_pack_suggestions(self, element_issues: Dict[str, List[str]], genre: Optional[str]) -> List[str]:
        """Generate specific sample pack suggestions"""
        suggestions = []
        
        if 'kick' in element_issues:
            suggestions.append(f"Splice - {genre.title() if genre else 'Modern'} Kick Collection")
            suggestions.append("KSHMR Kick Samples Vol. 1")
        
        if 'snare' in element_issues:
            suggestions.append(f"Loopmasters - {genre.title() if genre else 'Electronic'} Snares")
            suggestions.append("Sample Magic - Snare Rush")
        
        if 'bass' in element_issues:
            suggestions.append(f"Splice - {genre.title() if genre else 'Electronic'} Bass Collection")
            suggestions.append("Future Bass Essentials")
        
        return suggestions[:5]
    
    def _get_genre_specific_suggestions(self, genre: str, context: Dict[str, any]) -> List[str]:
        """Get genre-specific creative suggestions"""
        suggestions = {
            'house': [
                'Add classic house piano stabs',
                'Use sidechain compression for that pumping feel',
                'Layer white noise sweeps for transitions'
            ],
            'trap': [
                'Add 808 glides and pitch bends',
                'Use triplet hi-hat patterns',
                'Layer vocal chops and ad-libs'
            ],
            'techno': [
                'Add industrial percussion elements',
                'Use rhythmic delays and reverbs',
                'Create tension with filter automation'
            ],
            'dubstep': [
                'Design wobble bass sounds',
                'Use half-time drum patterns',
                'Add vocal chops with heavy processing'
            ]
        }
        
        return suggestions.get(genre.lower(), ['Experiment with your genre\'s signature sounds'])
    
    def _load_sample_library(self) -> Dict[str, any]:
        """Load sample library database"""
        # This would be expanded with a real database of sample packs
        return {
            'house': {
                'kick_packs': ['Deep House Kicks Vol. 1', 'Classic House Drums'],
                'full_construction_kits': ['House Essentials Kit', 'Deep Vibes Construction Kit']
            },
            'trap': {
                'kick_packs': ['Trap 808 Collection', 'Heavy Trap Kicks'],
                'full_construction_kits': ['Modern Trap Kit', 'Dark Trap Essentials']
            },
            'techno': {
                'kick_packs': ['Industrial Techno Kicks', 'Minimal Techno Drums'],
                'full_construction_kits': ['Underground Techno Kit']
            }
        }
    
    def _load_reference_tracks(self) -> Dict[str, List[Dict[str, any]]]:
        """Load reference track database"""
        return {
            'house': [
                {'artist': 'Disclosure', 'track': 'Latch', 'focus': ['groove', 'vocal integration']},
                {'artist': 'Calvin Harris', 'track': 'Feel So Close', 'focus': ['arrangement', 'energy']}
            ],
            'trap': [
                {'artist': 'RL Grime', 'track': 'Core', 'focus': ['808 design', 'dynamics']},
                {'artist': 'Baauer', 'track': 'Harlem Shake', 'focus': ['arrangement', 'drops']}
            ],
            'techno': [
                {'artist': 'Amelie Lens', 'track': 'Higher', 'focus': ['groove', 'minimalism']},
                {'artist': 'Charlotte de Witte', 'track': 'Reaction', 'focus': ['energy', 'progression']}
            ]
        }