"""
Natural Language Understanding Engine for BeatWizard
Advanced NLU system that understands music production queries with context awareness
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import difflib

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config.settings import ai_settings


class QueryIntent(Enum):
    """Enumeration of possible user intents"""
    ANALYSIS_QUESTION = "analysis_question"      # "What's wrong with my bass?"
    IMPROVEMENT_REQUEST = "improvement_request"   # "How can I make this better?"
    COMPARISON_QUERY = "comparison_query"        # "How does this compare to X?"
    LEARNING_QUESTION = "learning_question"      # "How do I learn mixing?"
    TECHNICAL_HELP = "technical_help"           # "What is LUFS?"
    CREATIVE_SUGGESTION = "creative_suggestion"  # "Give me ideas for this track"
    WORKFLOW_QUESTION = "workflow_question"      # "What should I do next?"
    FEEDBACK_REQUEST = "feedback_request"        # "Rate my track"
    TROUBLESHOOTING = "troubleshooting"         # "Why does my kick sound weak?"
    GENRE_ADVICE = "genre_advice"               # "How to make better trap music?"
    UNKNOWN = "unknown"


class MusicElement(Enum):
    """Music production elements that can be referenced"""
    KICK = "kick"
    SNARE = "snare"
    BASS = "bass"
    MELODY = "melody"
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    VOCALS = "vocals"
    LEAD = "lead"
    PLUCK = "pluck"
    PAD = "pad"
    HIHAT = "hihat"
    PERCUSSION = "percussion"
    DRUMS = "drums"
    SYNTH = "synth"
    SAMPLE = "sample"
    LOOP = "loop"
    ARRANGEMENT = "arrangement"
    STRUCTURE = "structure"
    MIX = "mix"
    MASTER = "master"
    LEVELS = "levels"
    EQ = "eq"
    COMPRESSION = "compression"
    REVERB = "reverb"
    DELAY = "delay"
    STEREO = "stereo"
    FREQUENCY = "frequency"
    LOUDNESS = "loudness"
    DYNAMICS = "dynamics"
    TEMPO = "tempo"
    KEY = "key"
    GENRE = "genre"
    ENERGY = "energy"
    VIBE = "vibe"
    OVERALL = "overall"


class SkillLevel(Enum):
    """User skill levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"


@dataclass
class QueryContext:
    """Context for understanding user queries"""
    user_skill_level: SkillLevel = SkillLevel.BEGINNER
    current_genre: Optional[str] = None
    conversation_history: List[str] = None
    analysis_results: Optional[Dict[str, Any]] = None
    last_focused_element: Optional[MusicElement] = None
    session_goals: List[str] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.session_goals is None:
            self.session_goals = []


@dataclass
class ParsedQuery:
    """Parsed and understood user query"""
    original_query: str
    intent: QueryIntent
    primary_elements: List[MusicElement]
    secondary_elements: List[MusicElement]
    urgency_level: int  # 1-5, 5 being most urgent
    emotional_context: Optional[str]  # frustrated, curious, excited, etc.
    specificity_level: int  # 1-5, how specific is the question
    requires_analysis_data: bool
    suggested_response_style: str  # detailed, quick, encouraging, technical
    confidence_score: float  # How confident we are in the parsing
    follow_up_suggestions: List[str]
    context_dependencies: List[str]  # What context is needed for good response


class NLUEngine:
    """
    Advanced Natural Language Understanding Engine for BeatWizard
    Understands music production queries with sophisticated context awareness
    """
    
    def __init__(self):
        """Initialize the NLU engine"""
        self.openai_available = OPENAI_AVAILABLE and ai_settings.OPENAI_API_KEY is not None
        
        if self.openai_available:
            openai.api_key = ai_settings.OPENAI_API_KEY
            logger.info("OpenAI integration enabled for advanced NLU")
        else:
            logger.warning("OpenAI not available - using pattern-based NLU")
        
        # Load language patterns and knowledge base
        self._load_music_vocabulary()
        self._load_intent_patterns()
        self._load_context_patterns()
        
        logger.info("NLU Engine initialized with advanced music understanding")
    
    def understand_query(self, query: str, context: QueryContext) -> ParsedQuery:
        """
        Main method to understand and parse user queries
        
        Args:
            query: Raw user input
            context: Current conversation and analysis context
            
        Returns:
            Parsed query with intent, elements, and response guidance
        """
        logger.debug(f"Understanding query: '{query}'")
        
        # Normalize and clean the query
        normalized_query = self._normalize_query(query)
        
        # Extract emotional context and urgency
        emotional_context = self._extract_emotional_context(query)
        urgency_level = self._determine_urgency(query, emotional_context)
        
        # Determine primary intent
        intent = self._classify_intent(normalized_query, context)
        
        # Extract music elements being referenced
        primary_elements, secondary_elements = self._extract_music_elements(normalized_query, context)
        
        # Determine specificity and response needs
        specificity_level = self._assess_specificity(normalized_query, primary_elements)
        requires_analysis_data = self._requires_analysis_data(intent, primary_elements)
        
        # Determine optimal response style
        response_style = self._determine_response_style(
            intent, context.user_skill_level, emotional_context, urgency_level
        )
        
        # Generate follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(intent, primary_elements, context)
        
        # Assess context dependencies
        context_dependencies = self._assess_context_dependencies(intent, primary_elements, context)
        
        # Calculate confidence in our understanding
        confidence_score = self._calculate_confidence(
            intent, primary_elements, specificity_level, context
        )
        
        parsed_query = ParsedQuery(
            original_query=query,
            intent=intent,
            primary_elements=primary_elements,
            secondary_elements=secondary_elements,
            urgency_level=urgency_level,
            emotional_context=emotional_context,
            specificity_level=specificity_level,
            requires_analysis_data=requires_analysis_data,
            suggested_response_style=response_style,
            confidence_score=confidence_score,
            follow_up_suggestions=follow_up_suggestions,
            context_dependencies=context_dependencies
        )
        
        logger.info(f"Query understood: Intent={intent.value}, Elements={[e.value for e in primary_elements]}, Confidence={confidence_score:.2f}")
        
        return parsed_query
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better processing"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Handle common contractions and abbreviations
        contractions = {
            "what's": "what is",
            "how's": "how is",
            "that's": "that is",
            "i'm": "i am",
            "can't": "cannot",
            "won't": "will not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Handle music production abbreviations
        abbreviations = {
            "eq": "equalizer",
            "comp": "compression",
            "db": "decibels",
            "hz": "hertz",
            "khz": "kilohertz",
            "bpm": "beats per minute",
            "daw": "digital audio workstation",
            "vst": "virtual studio technology",
            "midi": "musical instrument digital interface",
            "lufs": "loudness units full scale",
            "rms": "root mean square"
        }
        
        for abbr, full in abbreviations.items():
            # Use word boundaries to avoid replacing parts of words
            normalized = re.sub(rf'\b{abbr}\b', full, normalized)
        
        return normalized
    
    def _extract_emotional_context(self, query: str) -> Optional[str]:
        """Extract emotional context from the query"""
        emotional_indicators = {
            'frustrated': ['frustrated', 'annoying', 'hate', 'terrible', 'awful', 'sucks', 'wrong', 'bad'],
            'confused': ['confused', 'don\'t understand', 'unclear', 'lost', 'help', 'explain'],
            'excited': ['amazing', 'love', 'awesome', 'great', 'excited', 'perfect'],
            'curious': ['wondering', 'curious', 'interested', 'want to know', 'how does'],
            'urgent': ['urgent', 'quickly', 'asap', 'immediately', 'need now', 'rush'],
            'disappointed': ['disappointed', 'expected', 'thought it would', 'not working'],
            'determined': ['want to improve', 'get better', 'learn', 'master', 'perfect']
        }
        
        query_lower = query.lower()
        
        for emotion, indicators in emotional_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return emotion
        
        return None
    
    def _determine_urgency(self, query: str, emotional_context: Optional[str]) -> int:
        """Determine urgency level (1-5) of the query"""
        urgency_score = 1
        
        # High urgency indicators
        high_urgency_words = ['urgent', 'quickly', 'asap', 'immediately', 'deadline', 'due', 'tomorrow']
        if any(word in query.lower() for word in high_urgency_words):
            urgency_score = 5
        
        # Medium urgency indicators
        medium_urgency_words = ['need', 'help', 'stuck', 'problem', 'issue', 'wrong']
        if any(word in query.lower() for word in medium_urgency_words):
            urgency_score = max(urgency_score, 3)
        
        # Emotional context affects urgency
        if emotional_context in ['frustrated', 'urgent']:
            urgency_score = max(urgency_score, 4)
        elif emotional_context in ['confused', 'disappointed']:
            urgency_score = max(urgency_score, 3)
        
        return min(urgency_score, 5)
    
    def _classify_intent(self, query: str, context: QueryContext) -> QueryIntent:
        """Classify the user's intent based on the query and context"""
        
        # If OpenAI is available, use it for sophisticated intent classification
        if self.openai_available:
            return self._classify_intent_with_ai(query, context)
        
        # Otherwise use pattern-based classification
        return self._classify_intent_with_patterns(query, context)
    
    def _classify_intent_with_ai(self, query: str, context: QueryContext) -> QueryIntent:
        """Use OpenAI to classify intent with sophisticated understanding"""
        try:
            system_prompt = """You are an expert music production assistant. Classify the user's intent from their query.

Available intents:
- analysis_question: Asking about specific analysis results ("What's wrong with my bass?")
- improvement_request: Asking how to improve something ("How can I make this better?")
- comparison_query: Comparing to references ("How does this compare to X?")
- learning_question: Asking how to learn something ("How do I learn mixing?")
- technical_help: Asking about technical concepts ("What is LUFS?")
- creative_suggestion: Requesting creative ideas ("Give me ideas for this track")
- workflow_question: Asking about process/next steps ("What should I do next?")
- feedback_request: Requesting evaluation ("Rate my track")
- troubleshooting: Solving specific problems ("Why does my kick sound weak?")
- genre_advice: Genre-specific advice ("How to make better trap music?")

Consider the conversation context and return only the intent name."""

            user_prompt = f"Query: '{query}'\nUser skill level: {context.user_skill_level.value}\nGenre context: {context.current_genre or 'unknown'}"
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            intent_text = response.choices[0].message.content.strip().lower()
            
            # Map response to enum
            for intent in QueryIntent:
                if intent.value in intent_text:
                    return intent
            
            return QueryIntent.UNKNOWN
            
        except Exception as e:
            logger.warning(f"OpenAI intent classification failed: {e}")
            return self._classify_intent_with_patterns(query, context)
    
    def _classify_intent_with_patterns(self, query: str, context: QueryContext) -> QueryIntent:
        """Classify intent using pattern matching"""
        
        # Analysis question patterns
        analysis_patterns = [
            r'what.*wrong', r'what.*issue', r'analyze', r'analysis',
            r'problem.*with', r'issue.*with', r'what.*think.*about'
        ]
        
        # Improvement request patterns
        improvement_patterns = [
            r'how.*improve', r'how.*better', r'how.*fix', r'make.*better',
            r'suggestions', r'advice', r'tips', r'help.*with'
        ]
        
        # Learning question patterns
        learning_patterns = [
            r'how.*learn', r'teach.*me', r'tutorial', r'guide',
            r'how.*do', r'how.*make', r'steps.*to'
        ]
        
        # Technical help patterns
        technical_patterns = [
            r'what.*is', r'define', r'meaning', r'explain',
            r'lufs', r'compression', r'equalizer', r'frequency'
        ]
        
        # Creative suggestion patterns
        creative_patterns = [
            r'ideas', r'creative', r'inspiration', r'suggestions.*for',
            r'what.*should.*add', r'give.*me'
        ]
        
        # Workflow question patterns
        workflow_patterns = [
            r'what.*next', r'next.*step', r'workflow', r'process',
            r'should.*do', r'order', r'sequence'
        ]
        
        # Feedback request patterns
        feedback_patterns = [
            r'rate.*my', r'feedback.*on', r'thoughts.*on', r'opinion',
            r'what.*think', r'evaluate', r'review'
        ]
        
        # Troubleshooting patterns
        troubleshooting_patterns = [
            r'why.*not', r'why.*does', r'not.*working', r'sounds.*wrong',
            r'problem', r'issue', r'fix', r'solve'
        ]
        
        # Genre advice patterns
        genre_patterns = [
            r'trap.*music', r'house.*music', r'techno', r'dubstep',
            r'genre', r'style', r'type.*of.*music'
        ]
        
        # Check patterns in order of specificity
        pattern_groups = [
            (troubleshooting_patterns, QueryIntent.TROUBLESHOOTING),
            (genre_patterns, QueryIntent.GENRE_ADVICE),
            (analysis_patterns, QueryIntent.ANALYSIS_QUESTION),
            (improvement_patterns, QueryIntent.IMPROVEMENT_REQUEST),
            (learning_patterns, QueryIntent.LEARNING_QUESTION),
            (technical_patterns, QueryIntent.TECHNICAL_HELP),
            (creative_patterns, QueryIntent.CREATIVE_SUGGESTION),
            (workflow_patterns, QueryIntent.WORKFLOW_QUESTION),
            (feedback_patterns, QueryIntent.FEEDBACK_REQUEST)
        ]
        
        for patterns, intent in pattern_groups:
            if any(re.search(pattern, query) for pattern in patterns):
                return intent
        
        # Context-based fallback
        if context.analysis_results:
            return QueryIntent.ANALYSIS_QUESTION
        
        return QueryIntent.UNKNOWN
    
    def _extract_music_elements(self, query: str, context: QueryContext) -> Tuple[List[MusicElement], List[MusicElement]]:
        """Extract primary and secondary music elements from the query"""
        primary_elements = []
        secondary_elements = []
        
        # Music element keywords with synonyms
        element_keywords = {
            MusicElement.KICK: ['kick', 'kick drum', 'bd', 'bass drum'],
            MusicElement.SNARE: ['snare', 'snare drum', 'sd'],
            MusicElement.BASS: ['bass', 'bassline', 'sub bass', '808', 'low end'],
            MusicElement.MELODY: ['melody', 'melodic', 'tune', 'main melody'],
            MusicElement.HARMONY: ['harmony', 'chords', 'chord progression', 'harmonic'],
            MusicElement.RHYTHM: ['rhythm', 'rhythmic', 'groove', 'feel'],
            MusicElement.VOCALS: ['vocals', 'voice', 'singing', 'vocal'],
            MusicElement.LEAD: ['lead', 'lead synth', 'main synth'],
            MusicElement.PLUCK: ['pluck', 'plucky', 'stabs'],
            MusicElement.PAD: ['pad', 'pads', 'atmospheric'],
            MusicElement.HIHAT: ['hihat', 'hi hat', 'hats', 'hi-hat'],
            MusicElement.PERCUSSION: ['percussion', 'perc', 'drums'],
            MusicElement.DRUMS: ['drums', 'drum kit', 'drumming'],
            MusicElement.SYNTH: ['synth', 'synthesizer', 'synthetic'],
            MusicElement.SAMPLE: ['sample', 'samples', 'sampling'],
            MusicElement.LOOP: ['loop', 'loops', 'looping'],
            MusicElement.ARRANGEMENT: ['arrangement', 'arrange', 'structure'],
            MusicElement.STRUCTURE: ['structure', 'song structure', 'arrangement'],
            MusicElement.MIX: ['mix', 'mixing', 'mixdown'],
            MusicElement.MASTER: ['master', 'mastering', 'master bus'],
            MusicElement.LEVELS: ['levels', 'level', 'volume', 'gain'],
            MusicElement.EQ: ['eq', 'equalizer', 'equalization', 'frequency'],
            MusicElement.COMPRESSION: ['compression', 'compressor', 'compress'],
            MusicElement.REVERB: ['reverb', 'reverb', 'spatial'],
            MusicElement.DELAY: ['delay', 'echo', 'delay effect'],
            MusicElement.STEREO: ['stereo', 'stereo field', 'width', 'panning'],
            MusicElement.FREQUENCY: ['frequency', 'freq', 'hz', 'spectrum'],
            MusicElement.LOUDNESS: ['loudness', 'loud', 'lufs', 'volume'],
            MusicElement.DYNAMICS: ['dynamics', 'dynamic range', 'punch'],
            MusicElement.TEMPO: ['tempo', 'bpm', 'speed', 'timing'],
            MusicElement.KEY: ['key', 'scale', 'pitch', 'tuning'],
            MusicElement.GENRE: ['genre', 'style', 'type'],
            MusicElement.ENERGY: ['energy', 'energetic', 'power'],
            MusicElement.VIBE: ['vibe', 'mood', 'feeling', 'atmosphere'],
            MusicElement.OVERALL: ['overall', 'track', 'song', 'music', 'everything']
        }
        
        # Find all mentioned elements
        mentioned_elements = []
        for element, keywords in element_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    mentioned_elements.append((element, keyword))
        
        # Prioritize elements based on context and specificity
        if mentioned_elements:
            # Sort by keyword length (more specific first) and frequency
            mentioned_elements.sort(key=lambda x: (-len(x[1]), x[1]))
            
            # Primary elements are the most specific/important ones
            primary_count = min(3, len(mentioned_elements))
            primary_elements = [elem[0] for elem in mentioned_elements[:primary_count]]
            secondary_elements = [elem[0] for elem in mentioned_elements[primary_count:6]]
        
        # Context-based element inference
        if not primary_elements and context.last_focused_element:
            primary_elements = [context.last_focused_element]
        
        # Remove duplicates while preserving order
        primary_elements = list(dict.fromkeys(primary_elements))
        secondary_elements = list(dict.fromkeys(secondary_elements))
        
        return primary_elements, secondary_elements
    
    def _assess_specificity(self, query: str, primary_elements: List[MusicElement]) -> int:
        """Assess how specific the query is (1-5 scale)"""
        specificity_score = 1
        
        # Length and detail indicators
        word_count = len(query.split())
        if word_count > 15:
            specificity_score += 1
        if word_count > 25:
            specificity_score += 1
        
        # Specific music elements mentioned
        if len(primary_elements) >= 2:
            specificity_score += 1
        if len(primary_elements) >= 3:
            specificity_score += 1
        
        # Technical terms
        technical_terms = ['lufs', 'hertz', 'decibels', 'compression', 'equalizer', 'frequency']
        if any(term in query.lower() for term in technical_terms):
            specificity_score += 1
        
        # Specific problems or goals
        specific_indicators = ['exactly', 'specifically', 'precisely', 'detailed', 'step by step']
        if any(indicator in query.lower() for indicator in specific_indicators):
            specificity_score += 1
        
        return min(specificity_score, 5)
    
    def _requires_analysis_data(self, intent: QueryIntent, primary_elements: List[MusicElement]) -> bool:
        """Determine if the query requires analysis data to answer properly"""
        
        # Intents that typically need analysis data
        analysis_dependent_intents = [
            QueryIntent.ANALYSIS_QUESTION,
            QueryIntent.IMPROVEMENT_REQUEST,
            QueryIntent.FEEDBACK_REQUEST,
            QueryIntent.TROUBLESHOOTING
        ]
        
        if intent in analysis_dependent_intents:
            return True
        
        # Elements that typically need analysis data
        analysis_dependent_elements = [
            MusicElement.KICK, MusicElement.SNARE, MusicElement.BASS,
            MusicElement.FREQUENCY, MusicElement.LOUDNESS, MusicElement.DYNAMICS,
            MusicElement.STEREO, MusicElement.LEVELS, MusicElement.OVERALL
        ]
        
        if any(elem in primary_elements for elem in analysis_dependent_elements):
            return True
        
        return False
    
    def _determine_response_style(self, intent: QueryIntent, skill_level: SkillLevel, 
                                emotional_context: Optional[str], urgency_level: int) -> str:
        """Determine the optimal response style"""
        
        # Base style on skill level
        if skill_level == SkillLevel.BEGINNER:
            base_style = "simple"
        elif skill_level == SkillLevel.INTERMEDIATE:
            base_style = "balanced"
        else:
            base_style = "technical"
        
        # Adjust for emotional context
        if emotional_context == 'frustrated':
            return "reassuring"
        elif emotional_context == 'confused':
            return "clarifying"
        elif emotional_context == 'excited':
            return "enthusiastic"
        elif emotional_context == 'urgent':
            return "direct"
        
        # Adjust for urgency
        if urgency_level >= 4:
            return "quick"
        elif urgency_level <= 2:
            return "comprehensive"
        
        # Adjust for intent
        if intent == QueryIntent.LEARNING_QUESTION:
            return "educational"
        elif intent == QueryIntent.TECHNICAL_HELP:
            return "explanatory"
        elif intent == QueryIntent.CREATIVE_SUGGESTION:
            return "inspiring"
        
        return base_style
    
    def _generate_follow_up_suggestions(self, intent: QueryIntent, primary_elements: List[MusicElement], 
                                      context: QueryContext) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        
        # Intent-based suggestions
        if intent == QueryIntent.ANALYSIS_QUESTION:
            suggestions.extend([
                "Would you like specific improvement suggestions?",
                "Should I explain why this might be happening?",
                "Want to compare this to professional references?"
            ])
        
        elif intent == QueryIntent.IMPROVEMENT_REQUEST:
            suggestions.extend([
                "Would you like step-by-step instructions?",
                "Should I suggest specific tools or plugins?",
                "Want to learn the theory behind this?"
            ])
        
        elif intent == QueryIntent.LEARNING_QUESTION:
            suggestions.extend([
                "Would you like a practice exercise?",
                "Should I recommend learning resources?",
                "Want to see this applied to your current track?"
            ])
        
        # Element-based suggestions
        if MusicElement.KICK in primary_elements:
            suggestions.append("Want kick selection recommendations?")
        
        if MusicElement.BASS in primary_elements:
            suggestions.append("Should I analyze the kick-bass relationship?")
        
        if MusicElement.MIX in primary_elements:
            suggestions.append("Want a complete mixing checklist?")
        
        # Context-based suggestions
        if context.current_genre:
            suggestions.append(f"Want {context.current_genre}-specific advice?")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _assess_context_dependencies(self, intent: QueryIntent, primary_elements: List[MusicElement], 
                                   context: QueryContext) -> List[str]:
        """Assess what context information is needed for optimal response"""
        dependencies = []
        
        # Check if analysis results are needed but missing
        if self._requires_analysis_data(intent, primary_elements) and not context.analysis_results:
            dependencies.append("analysis_results")
        
        # Check if genre context would help
        genre_dependent_intents = [QueryIntent.GENRE_ADVICE, QueryIntent.CREATIVE_SUGGESTION]
        if intent in genre_dependent_intents and not context.current_genre:
            dependencies.append("genre_context")
        
        # Check if skill level assessment is needed
        if context.user_skill_level == SkillLevel.BEGINNER and intent == QueryIntent.TECHNICAL_HELP:
            dependencies.append("skill_assessment")
        
        # Check if conversation history would provide better context
        if intent == QueryIntent.IMPROVEMENT_REQUEST and len(context.conversation_history) < 2:
            dependencies.append("conversation_history")
        
        return dependencies
    
    def _calculate_confidence(self, intent: QueryIntent, primary_elements: List[MusicElement], 
                            specificity_level: int, context: QueryContext) -> float:
        """Calculate confidence in our understanding of the query"""
        confidence = 0.5  # Base confidence
        
        # Intent classification confidence
        if intent != QueryIntent.UNKNOWN:
            confidence += 0.2
        
        # Element extraction confidence
        if primary_elements:
            confidence += 0.1 * min(len(primary_elements), 3)
        
        # Specificity confidence
        confidence += 0.05 * specificity_level
        
        # Context availability confidence
        if context.analysis_results:
            confidence += 0.1
        if context.current_genre:
            confidence += 0.05
        if len(context.conversation_history) > 0:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _load_music_vocabulary(self):
        """Load comprehensive music production vocabulary"""
        # This would be expanded with a comprehensive music vocabulary database
        self.music_vocabulary = {
            'elements': [elem.value for elem in MusicElement],
            'genres': ['house', 'techno', 'trap', 'dubstep', 'pop', 'hip hop', 'trance', 'drum and bass'],
            'effects': ['reverb', 'delay', 'chorus', 'flanger', 'phaser', 'distortion', 'compression'],
            'technical_terms': ['lufs', 'rms', 'peak', 'hz', 'khz', 'db', 'bpm', 'semitone']
        }
    
    def _load_intent_patterns(self):
        """Load intent classification patterns"""
        # This would be expanded with sophisticated pattern matching
        self.intent_patterns = {
            QueryIntent.ANALYSIS_QUESTION: [
                r'what.*wrong', r'analyze.*this', r'problem.*with'
            ],
            QueryIntent.IMPROVEMENT_REQUEST: [
                r'how.*improve', r'make.*better', r'fix.*this'
            ]
            # ... more patterns
        }
    
    def _load_context_patterns(self):
        """Load context understanding patterns"""
        # This would include patterns for understanding context and references
        self.context_patterns = {
            'previous_reference': [r'this', r'that', r'it', r'the.*one'],
            'comparison': [r'compared.*to', r'like.*in', r'similar.*to'],
            'continuation': [r'also', r'and', r'plus', r'additionally']
        }
    
    def update_context(self, context: QueryContext, new_query: str, response: str):
        """Update conversation context with new information"""
        context.conversation_history.append(f"Q: {new_query}")
        context.conversation_history.append(f"A: {response}")
        
        # Keep only recent history to avoid token limits
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
        
        # Extract potential new context information
        # This could be expanded to learn about user preferences, skill level, etc.
        pass


# Example usage and testing
if __name__ == "__main__":
    # Initialize NLU engine
    nlu = NLUEngine()
    
    # Create sample context
    context = QueryContext(
        user_skill_level=SkillLevel.BEGINNER,
        current_genre="trap",
        conversation_history=[]
    )
    
    # Test queries
    test_queries = [
        "What's wrong with my kick drum?",
        "How can I make my bass sound better?",
        "My track sounds muddy, help!",
        "What is LUFS and why does it matter?",
        "Give me some creative ideas for this beat",
        "What should I do next to improve my mixing?"
    ]
    
    for query in test_queries:
        parsed = nlu.understand_query(query, context)
        print(f"\nQuery: {query}")
        print(f"Intent: {parsed.intent.value}")
        print(f"Elements: {[e.value for e in parsed.primary_elements]}")
        print(f"Response Style: {parsed.suggested_response_style}")
        print(f"Confidence: {parsed.confidence_score:.2f}")
