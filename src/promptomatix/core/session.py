"""
Module for managing optimization sessions and their state.
"""

from datetime import datetime
from typing import List, Dict, Optional
from .feedback import Feedback
from ..utils.logging import SessionLogger
from ..core.config import Config, ModelProvider
from ..utils.paths import SESSIONS_DIR
import json
import os
from pathlib import Path
from enum import Enum

class OptimizationSession:
    """
    Manages the state and lifecycle of a prompt optimization session.
    
    Attributes:
        session_id (str): Unique identifier for the session
        initial_human_input (str): Original prompt from user
        updated_human_input (str): Current version of the prompt
        latest_optimized_prompt (str): Most recent optimized prompt
        config (Config): Session configuration
        created_at (datetime): Session creation timestamp
        logger (SessionLogger): Session-specific logger
        comprehensive_feedback (str): Comprehensive feedback for the session
        individual_feedbacks (List[Dict]): Individual feedback for the session
    """
    
    def __init__(self, session_id: str, initial_human_input: str, config: Config):
        """
        Initialize a new optimization session.
        
        Args:
            session_id (str): Unique identifier for the session
            initial_human_input (str): Original prompt from user
            config (Config): Configuration for the session
        """
        self.session_id = session_id
        self.initial_human_input = initial_human_input
        self.updated_human_input = initial_human_input
        self.latest_optimized_prompt = None
        self.latest_human_feedback: List[Feedback] = []
        self.config = config
        self.created_at = datetime.now()
        self.logger = SessionLogger(session_id)
        
        # Add feedback storage attributes
        self.comprehensive_feedback = None
        self.individual_feedbacks = []
        
        # Log session creation
        self.logger.add_entry("SESSION_START", {
            "action": "Session Created",
            "input": initial_human_input,
            "config": {
                "model": config.model_name,
                "task_type": config.task_type
            }
        })
    
    def add_feedback(self, feedback: Feedback) -> None:
        """Add a new feedback to the session."""
        self.latest_human_feedback.append(feedback)
        self.logger.add_entry("COMMENT_ADDED", {
            "feedback_id": feedback.id,
            "text": feedback.text,
            "feedback": feedback.feedback
        })
    
    def update_optimized_prompt(self, new_prompt: str) -> None:
        """Update the latest optimized prompt."""
        self.latest_optimized_prompt = new_prompt
        self.logger.add_entry("PROMPT_UPDATE", {
            "action": "Optimized Prompt Updated",
            "new_prompt": new_prompt
        })
    
    def update_human_input(self, new_input: str) -> None:
        """Update the human input prompt."""
        self.updated_human_input = new_input
        self.logger.add_entry("INPUT_UPDATE", {
            "action": "Human Input Updated",
            "new_input": new_input
        })
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary format."""
        return {
            'session_id': self.session_id,
            'initial_human_input': self.initial_human_input,
            'updated_human_input': self.updated_human_input,
            'latest_optimized_prompt': self.latest_optimized_prompt,
            'latest_human_feedback': [{
                'id': c.id,
                'text': c.text,
                'start_offset': c.start_offset,
                'end_offset': c.end_offset,
                'feedback': c.feedback,
                'created_at': c.created_at.isoformat()
            } for c in self.latest_human_feedback],
            'comprehensive_feedback': self.comprehensive_feedback,
            'individual_feedbacks': self.individual_feedbacks,
            'created_at': self.created_at.isoformat(),
            'config': self.config
        }

class SessionManager:
    """
    Manages multiple optimization sessions with persistence.
    """
    
    def __init__(self):
        self.sessions: Dict[str, OptimizationSession] = {}
        self.sessions_dir = SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def load_session_from_file(self, session_file_path: str) -> Optional[OptimizationSession]:
        """Load a specific session from a file path."""
        try:
            with open(session_file_path, 'r') as f:
                session_data = json.load(f)
                
                # Convert string values back to enums where needed
                config_data = session_data.get('config', {})
                if 'model_provider' in config_data:
                    config_data['model_provider'] = ModelProvider(config_data['model_provider'])
                
                config = Config(**config_data)
                session = OptimizationSession(
                    session_id=session_data['session_id'],
                    initial_human_input=session_data['initial_human_input'],
                    config=config
                )
                session.updated_human_input = session_data['updated_human_input']
                session.latest_optimized_prompt = session_data['latest_optimized_prompt']
                self.sessions[session.session_id] = session
                return session
        except Exception as e:
            print(f"Error loading session {session_file_path}: {e}")
            return None
    
    def _save_session(self, session: OptimizationSession):
        """Save session data to a file."""
        session_path = SESSIONS_DIR / f'{session.session_id}.json'
        
        # Ensure the sessions directory exists
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        session_data = session.to_dict()
        
        # Handle Config object serialization
        config_dict = {}
        config = session_data['config']
        
        # Convert Config object to dictionary
        if hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_') and not callable(v)}
        else:
            # If config is already a dict, use it directly
            config_dict = config
        
        # Handle nested objects in config
        for key, value in config_dict.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                # Handle basic types directly
                continue
            elif hasattr(value, '__dict__'):
                # For objects with __dict__, convert to dict
                config_dict[key] = {k: v for k, v in value.__dict__.items() 
                                  if not k.startswith('_') and not callable(v)}
            else:
                try:
                    # Try to serialize the value directly
                    json.dumps(value)  # Test if serializable
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    config_dict[key] = str(value)
        
        session_data['config'] = config_dict
        
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def create_session(self, session_id: str, initial_input: str, config: Config) -> OptimizationSession:
        """Create and store a new optimization session."""
        session = OptimizationSession(session_id, initial_input, config)
        self.sessions[session_id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[OptimizationSession]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    def update_session(self, session: OptimizationSession):
        """Update a session and persist changes."""
        self.sessions[session.session_id] = session
        self._save_session(session)
    
    def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        return [session.to_dict() for session in self.sessions.values()] 