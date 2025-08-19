"""
Conversation Manager for BeatWizard
Manages intelligent conversations about music production with context awareness
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

from .nlu_engine import NLUEngine, QueryContext, SkillLevel, ParsedQuery
from .intelligent_response_generator import IntelligentResponseGenerator
from .intelligent_feedback import IntelligentFeedbackGenerator


@dataclass
class ConversationSession:
    """Represents an ongoing conversation session"""
    session_id: str
    user_id: Optional[str]
    skill_level: SkillLevel
    current_genre: Optional[str]
    conversation_history: List[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    session_goals: List[str]
    preferred_response_style: str
    created_at: float
    last_activity: float
    context_memory: Dict[str, Any]  # Long-term context memory
    
    def __post_init__(self):
        if not self.conversation_history:
            self.conversation_history = []
        if not self.session_goals:
            self.session_goals = []
        if not self.context_memory:
            self.context_memory = {}


class ConversationManager:
    """
    Manages intelligent conversations about music production
    Combines NLU understanding with intelligent response generation
    """
    
    def __init__(self):
        """Initialize the conversation manager"""
        self.nlu_engine = NLUEngine()
        self.response_generator = IntelligentResponseGenerator()
        self.feedback_generator = IntelligentFeedbackGenerator()
        
        # Active conversation sessions
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Conversation analytics
        self.conversation_analytics = {
            'total_conversations': 0,
            'common_questions': {},
            'user_skill_progression': {},
            'successful_resolutions': 0
        }
        
        logger.info("Conversation Manager initialized with advanced NLU and response generation")
    
    def start_conversation(self, session_id: str, user_id: Optional[str] = None, 
                         skill_level: SkillLevel = SkillLevel.BEGINNER,
                         current_genre: Optional[str] = None,
                         analysis_results: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """
        Start a new conversation session
        
        Args:
            session_id: Unique session identifier
            user_id: Optional user identifier for personalization
            skill_level: User's music production skill level
            current_genre: Current genre being worked on
            analysis_results: Current track analysis results
            
        Returns:
            New conversation session
        """
        logger.info(f"Starting conversation session: {session_id}")
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            skill_level=skill_level,
            current_genre=current_genre,
            conversation_history=[],
            analysis_results=analysis_results,
            session_goals=[],
            preferred_response_style="adaptive",
            created_at=time.time(),
            last_activity=time.time(),
            context_memory={}
        )
        
        self.active_sessions[session_id] = session
        self.conversation_analytics['total_conversations'] += 1
        
        # Generate welcome message
        welcome_message = self._generate_welcome_message(session)
        
        self._add_to_conversation_history(session, "system", welcome_message)
        
        logger.debug(f"Conversation session started for skill level: {skill_level.value}")
        
        return session
    
    def process_user_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate intelligent response
        
        Args:
            session_id: Session identifier
            user_input: User's question or input
            
        Returns:
            Intelligent response with context and suggestions
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, starting new session")
            session = self.start_conversation(session_id)
        else:
            session = self.active_sessions[session_id]
        
        session.last_activity = time.time()
        
        logger.info(f"Processing user input: '{user_input[:50]}...' in session {session_id}")
        
        # Create query context from session
        query_context = self._create_query_context(session)
        
        # Understand the user's query using NLU
        parsed_query = self.nlu_engine.understand_query(user_input, query_context)
        
        # Generate intelligent response
        response = self.response_generator.generate_response(parsed_query, query_context)
        
        # Update conversation history
        self._add_to_conversation_history(session, "user", user_input)
        self._add_to_conversation_history(session, "assistant", response)
        
        # Update session context based on the conversation
        self._update_session_context(session, parsed_query, response)
        
        # Update analytics
        self._update_analytics(parsed_query, response)
        
        # Add session management information
        response['session_info'] = {
            'session_id': session_id,
            'conversation_length': len(session.conversation_history),
            'skill_level': session.skill_level.value,
            'genre_context': session.current_genre,
            'has_analysis': session.analysis_results is not None
        }
        
        logger.debug(f"Response generated with confidence: {response.get('confidence', 0):.2f}")
        
        return response
    
    def update_analysis_results(self, session_id: str, analysis_results: Dict[str, Any]) -> None:
        """
        Update session with new analysis results
        
        Args:
            session_id: Session identifier
            analysis_results: New analysis results to incorporate
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.analysis_results = analysis_results
            session.last_activity = time.time()
            
            # Add context about new analysis
            self._add_to_conversation_history(
                session, 
                "system", 
                {"type": "analysis_update", "message": "New track analysis available"}
            )
            
            logger.info(f"Analysis results updated for session {session_id}")
        else:
            logger.warning(f"Attempted to update analysis for non-existent session {session_id}")
    
    def update_user_skill_level(self, session_id: str, new_skill_level: SkillLevel) -> None:
        """
        Update user's skill level based on interactions
        
        Args:
            session_id: Session identifier
            new_skill_level: Updated skill level
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            old_level = session.skill_level
            session.skill_level = new_skill_level
            session.last_activity = time.time()
            
            # Add context about skill progression
            self._add_to_conversation_history(
                session,
                "system",
                {
                    "type": "skill_update",
                    "message": f"Skill level updated from {old_level.value} to {new_skill_level.value}"
                }
            )
            
            logger.info(f"Skill level updated for session {session_id}: {old_level.value} -> {new_skill_level.value}")
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation summary with key insights
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Analyze conversation patterns
        user_messages = [msg for msg in session.conversation_history if msg.get('sender') == 'user']
        assistant_messages = [msg for msg in session.conversation_history if msg.get('sender') == 'assistant']
        
        # Extract common themes and topics
        discussed_topics = self._extract_discussed_topics(session)
        progress_indicators = self._assess_user_progress(session)
        
        summary = {
            'session_info': {
                'session_id': session_id,
                'duration_minutes': (time.time() - session.created_at) / 60,
                'message_count': len(session.conversation_history),
                'user_messages': len(user_messages),
                'assistant_responses': len(assistant_messages)
            },
            'user_profile': {
                'skill_level': session.skill_level.value,
                'current_genre': session.current_genre,
                'session_goals': session.session_goals,
                'has_analysis_data': session.analysis_results is not None
            },
            'conversation_analysis': {
                'discussed_topics': discussed_topics,
                'progress_indicators': progress_indicators,
                'most_frequent_questions': self._get_frequent_question_types(session),
                'resolution_status': self._assess_resolution_status(session)
            },
            'recommendations': {
                'next_steps': self._recommend_next_steps(session),
                'learning_focus': self._recommend_learning_focus(session),
                'skill_development': self._recommend_skill_development(session)
            }
        }
        
        return summary
    
    def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        End a conversation session and generate summary
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final conversation summary
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Generate final summary
        summary = self.get_conversation_summary(session_id)
        
        # Add farewell message
        farewell_message = self._generate_farewell_message(session)
        self._add_to_conversation_history(session, "system", farewell_message)
        
        # Archive session
        self._archive_session(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Conversation session {session_id} ended")
        
        return summary
    
    def get_conversation_suggestions(self, session_id: str) -> List[str]:
        """
        Get intelligent conversation suggestions based on current context
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of suggested questions or topics
        """
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        suggestions = []
        
        # Suggestions based on analysis results
        if session.analysis_results:
            suggestions.extend(self._get_analysis_based_suggestions(session))
        
        # Suggestions based on skill level
        suggestions.extend(self._get_skill_based_suggestions(session))
        
        # Suggestions based on genre
        if session.current_genre:
            suggestions.extend(self._get_genre_based_suggestions(session))
        
        # Suggestions based on conversation history
        suggestions.extend(self._get_history_based_suggestions(session))
        
        # Remove duplicates and limit to top suggestions
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:6]
    
    def _create_query_context(self, session: ConversationSession) -> QueryContext:
        """Create query context from conversation session"""
        return QueryContext(
            user_skill_level=session.skill_level,
            current_genre=session.current_genre,
            conversation_history=[msg.get('content', '') for msg in session.conversation_history if isinstance(msg.get('content'), str)],
            analysis_results=session.analysis_results,
            last_focused_element=session.context_memory.get('last_focused_element'),
            session_goals=session.session_goals
        )
    
    def _add_to_conversation_history(self, session: ConversationSession, sender: str, content: Any) -> None:
        """Add message to conversation history"""
        message = {
            'sender': sender,
            'content': content,
            'timestamp': time.time()
        }
        
        session.conversation_history.append(message)
        
        # Limit conversation history to prevent memory issues
        if len(session.conversation_history) > 50:
            session.conversation_history = session.conversation_history[-40:]
    
    def _update_session_context(self, session: ConversationSession, parsed_query: ParsedQuery, response: Dict[str, Any]) -> None:
        """Update session context based on conversation"""
        
        # Update last focused elements
        if parsed_query.primary_elements:
            session.context_memory['last_focused_element'] = parsed_query.primary_elements[0]
        
        # Update session goals based on conversation
        if parsed_query.intent.value not in ['unknown', 'technical_help']:
            intent_goal = f"Improve {parsed_query.intent.value.replace('_', ' ')}"
            if intent_goal not in session.session_goals:
                session.session_goals.append(intent_goal)
                
        # Update response style preference based on user engagement
        if response.get('confidence', 0) > 0.8:
            # High confidence responses might indicate good style match
            session.preferred_response_style = parsed_query.suggested_response_style
        
        # Update genre if detected in conversation
        if parsed_query.primary_elements and not session.current_genre:
            # Try to infer genre from conversation context
            inferred_genre = self._infer_genre_from_elements(parsed_query.primary_elements)
            if inferred_genre:
                session.current_genre = inferred_genre
    
    def _update_analytics(self, parsed_query: ParsedQuery, response: Dict[str, Any]) -> None:
        """Update conversation analytics"""
        
        # Track common question types
        intent = parsed_query.intent.value
        if intent in self.conversation_analytics['common_questions']:
            self.conversation_analytics['common_questions'][intent] += 1
        else:
            self.conversation_analytics['common_questions'][intent] = 1
        
        # Track successful resolutions (high confidence responses)
        if response.get('confidence', 0) > 0.7:
            self.conversation_analytics['successful_resolutions'] += 1
    
    def _generate_welcome_message(self, session: ConversationSession) -> Dict[str, Any]:
        """Generate personalized welcome message"""
        
        welcome_messages = {
            SkillLevel.BEGINNER: "Welcome to BeatWizard! I'm here to help you learn music production. Feel free to ask about any aspect of your tracks - from basic concepts to specific feedback!",
            SkillLevel.INTERMEDIATE: "Hey there! Ready to take your production skills to the next level? I can help with analysis, creative ideas, and advanced techniques.",
            SkillLevel.ADVANCED: "Welcome back! Let's dive deep into your production. I'm here for detailed analysis, workflow optimization, and creative breakthroughs."
        }
        
        welcome = welcome_messages.get(session.skill_level, welcome_messages[SkillLevel.BEGINNER])
        
        if session.analysis_results:
            welcome += " I can see you have analysis results ready - feel free to ask specific questions about your track!"
        
        if session.current_genre:
            welcome += f" I notice you're working on {session.current_genre} - I can provide genre-specific advice!"
        
        return {
            'type': 'welcome',
            'message': welcome,
            'suggestions': self._get_initial_suggestions(session)
        }
    
    def _generate_farewell_message(self, session: ConversationSession) -> Dict[str, Any]:
        """Generate personalized farewell message"""
        
        progress_summary = self._assess_user_progress(session)
        
        farewell = "Thanks for the great conversation! "
        
        if progress_summary.get('showed_improvement'):
            farewell += "I could see you making progress throughout our discussion. "
        
        farewell += "Keep experimenting and trust your creative instincts. "
        
        if session.skill_level == SkillLevel.BEGINNER:
            farewell += "Remember: every producer started where you are - consistency is key!"
        else:
            farewell += "Keep pushing boundaries and exploring new techniques!"
        
        return {
            'type': 'farewell',
            'message': farewell,
            'session_summary': progress_summary
        }
    
    def _extract_discussed_topics(self, session: ConversationSession) -> List[str]:
        """Extract main topics discussed in the conversation"""
        topics = set()
        
        for message in session.conversation_history:
            content = message.get('content', '')
            if isinstance(content, dict) and content.get('metadata'):
                elements = content['metadata'].get('elements_discussed', [])
                topics.update(elements)
        
        return list(topics)
    
    def _assess_user_progress(self, session: ConversationSession) -> Dict[str, Any]:
        """Assess user progress during the conversation"""
        
        user_messages = [msg for msg in session.conversation_history if msg.get('sender') == 'user']
        
        progress = {
            'showed_improvement': False,
            'asked_followup_questions': False,
            'engaged_with_suggestions': False,
            'progressed_in_complexity': False
        }
        
        # Check for follow-up questions (indicates engagement)
        if len(user_messages) > 2:
            progress['asked_followup_questions'] = True
        
        # Check for progression in question complexity
        if len(user_messages) > 3:
            early_questions = [msg['content'] for msg in user_messages[:2] if isinstance(msg.get('content'), str)]
            later_questions = [msg['content'] for msg in user_messages[-2:] if isinstance(msg.get('content'), str)]
            
            # Simple heuristic: later questions are longer/more specific
            avg_early_length = sum(len(q.split()) for q in early_questions) / max(len(early_questions), 1)
            avg_later_length = sum(len(q.split()) for q in later_questions) / max(len(later_questions), 1)
            
            if avg_later_length > avg_early_length * 1.2:
                progress['progressed_in_complexity'] = True
        
        return progress
    
    def _get_analysis_based_suggestions(self, session: ConversationSession) -> List[str]:
        """Get suggestions based on analysis results"""
        suggestions = []
        
        if not session.analysis_results:
            return ["Upload a track for detailed analysis and feedback"]
        
        # Check for common issues in analysis
        overall_assessment = session.analysis_results.get('overall_assessment', {})
        
        if overall_assessment.get('overall_quality') == 'poor':
            suggestions.append("What are the main issues with my track?")
            suggestions.append("How can I improve the overall quality?")
        
        if 'sound_selection_analysis' in session.analysis_results:
            sound_analysis = session.analysis_results['sound_selection_analysis']
            if sound_analysis.get('kick_analysis', {}).get('quality') == 'poor':
                suggestions.append("Help me choose a better kick drum")
            if sound_analysis.get('bass_analysis', {}).get('quality') == 'poor':
                suggestions.append("My bass needs work - any suggestions?")
        
        return suggestions
    
    def _get_skill_based_suggestions(self, session: ConversationSession) -> List[str]:
        """Get suggestions based on skill level"""
        
        suggestions = {
            SkillLevel.BEGINNER: [
                "What should I focus on as a beginner?",
                "How do I structure a basic track?",
                "What are the essential mixing techniques?",
                "How do I choose good samples?"
            ],
            SkillLevel.INTERMEDIATE: [
                "How can I make my tracks more professional?",
                "What advanced techniques should I learn?",
                "How do I develop my unique sound?",
                "Tips for better arrangement and structure?"
            ],
            SkillLevel.ADVANCED: [
                "How can I optimize my workflow?",
                "Advanced mixing and mastering techniques?",
                "Creative sound design approaches?",
                "Industry-standard production practices?"
            ]
        }
        
        return suggestions.get(session.skill_level, suggestions[SkillLevel.BEGINNER])
    
    def _get_genre_based_suggestions(self, session: ConversationSession) -> List[str]:
        """Get suggestions based on current genre"""
        
        genre_suggestions = {
            'trap': [
                "How do I make better trap beats?",
                "What makes a good 808 pattern?",
                "Trap mixing and arrangement tips?"
            ],
            'house': [
                "Essential elements of house music?",
                "How to create that house groove?",
                "House music arrangement structure?"
            ],
            'techno': [
                "Techno production fundamentals?",
                "Creating driving techno rhythms?",
                "Industrial sound design for techno?"
            ]
        }
        
        return genre_suggestions.get(session.current_genre, [])
    
    def _get_history_based_suggestions(self, session: ConversationSession) -> List[str]:
        """Get suggestions based on conversation history"""
        suggestions = []
        
        # If they've been asking about technical concepts, suggest practical application
        technical_questions = sum(1 for msg in session.conversation_history 
                                if 'technical' in str(msg.get('content', '')).lower())
        
        if technical_questions > 1:
            suggestions.append("How do I apply these concepts to my track?")
            suggestions.append("Can you show me practical examples?")
        
        return suggestions
    
    def _get_initial_suggestions(self, session: ConversationSession) -> List[str]:
        """Get initial conversation suggestions for new session"""
        base_suggestions = [
            "What's wrong with my track?",
            "How can I improve my mixing?",
            "Give me feedback on my beat",
            "What should I learn next?"
        ]
        
        # Add context-specific suggestions
        if session.analysis_results:
            base_suggestions.insert(0, "Analyze my track and give me feedback")
        
        if session.current_genre:
            base_suggestions.append(f"Give me {session.current_genre} production tips")
        
        return base_suggestions[:4]
    
    def _archive_session(self, session: ConversationSession) -> None:
        """Archive completed session for future analysis"""
        # In a real implementation, this would save to a database
        logger.info(f"Archiving session {session.session_id} with {len(session.conversation_history)} messages")
    
    def _infer_genre_from_elements(self, elements) -> Optional[str]:
        """Infer genre from discussed music elements"""
        # Simple genre inference based on elements discussed
        # This could be much more sophisticated
        element_names = [elem.value for elem in elements]
        
        if 'bass' in element_names and 'kick' in element_names:
            return 'electronic'
        
        return None
    
    def _assess_resolution_status(self, session: ConversationSession) -> str:
        """Assess whether user questions were resolved"""
        # This is a simplified assessment
        # In practice, this could use ML to analyze conversation sentiment and resolution
        
        user_messages = [msg for msg in session.conversation_history if msg.get('sender') == 'user']
        
        if len(user_messages) < 2:
            return 'insufficient_data'
        elif len(user_messages) > 5:
            return 'engaged_conversation'
        else:
            return 'basic_resolution'
    
    def _recommend_next_steps(self, session: ConversationSession) -> List[str]:
        """Recommend next steps based on conversation"""
        steps = []
        
        if not session.analysis_results:
            steps.append("Upload a track for detailed analysis")
        
        discussed_topics = self._extract_discussed_topics(session)
        
        if 'kick' in discussed_topics:
            steps.append("Apply kick improvements to your track")
        
        if 'mix' in discussed_topics:
            steps.append("Practice mixing techniques discussed")
        
        if session.skill_level == SkillLevel.BEGINNER:
            steps.append("Complete one full track using these concepts")
        
        return steps[:3]
    
    def _recommend_learning_focus(self, session: ConversationSession) -> List[str]:
        """Recommend learning focus areas"""
        focus_areas = []
        
        # Based on conversation topics and skill level
        if session.skill_level == SkillLevel.BEGINNER:
            focus_areas.extend(["Basic mixing", "Song structure", "Sound selection"])
        elif session.skill_level == SkillLevel.INTERMEDIATE:
            focus_areas.extend(["Advanced arrangement", "Creative sound design", "Genre mastery"])
        
        return focus_areas[:3]
    
    def _recommend_skill_development(self, session: ConversationSession) -> List[str]:
        """Recommend skill development activities"""
        activities = []
        
        progress = self._assess_user_progress(session)
        
        if progress.get('progressed_in_complexity'):
            activities.append("Continue exploring advanced concepts")
        else:
            activities.append("Practice fundamentals consistently")
        
        if session.analysis_results:
            activities.append("Analyze reference tracks in your genre")
        
        activities.append("Set specific production goals for next week")
        
        return activities[:3]
    
    def _get_frequent_question_types(self, session: ConversationSession) -> List[str]:
        """Get most frequent question types in this session"""
        # This would analyze the conversation history for patterns
        # For now, return placeholder
        return ["improvement_request", "technical_help", "analysis_question"]


# Example usage
if __name__ == "__main__":
    # Initialize conversation manager
    conv_manager = ConversationManager()
    
    # Start a conversation
    session = conv_manager.start_conversation(
        session_id="test_session_1",
        skill_level=SkillLevel.BEGINNER,
        current_genre="trap"
    )
    
    # Process some user inputs
    test_inputs = [
        "What's wrong with my kick drum?",
        "How can I make it punchier?",
        "What about the bass? Does it work with the kick?"
    ]
    
    for user_input in test_inputs:
        response = conv_manager.process_user_input("test_session_1", user_input)
        print(f"\nUser: {user_input}")
        print(f"Assistant: {response['primary_answer'][:100]}...")
    
    # Get conversation summary
    summary = conv_manager.get_conversation_summary("test_session_1")
    print(f"\nConversation Summary: {summary['session_info']}")
    
    # End conversation
    final_summary = conv_manager.end_conversation("test_session_1")
    print(f"Final Summary: {final_summary['session_info']}")
