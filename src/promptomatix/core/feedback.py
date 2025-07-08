"""
Module for handling feedback and annotations in the prompt optimization process.
"""

from datetime import datetime
from typing import List, Dict, Optional
import json

class Feedback:
    """
    Represents a feedback or annotation on a prompt.
    
    Attributes:
        id (str): Unique identifier for the feedback
        text (str): The text being given feedback on
        start_offset (int): Starting position of the feedback
        end_offset (int): Ending position of the feedback
        feedback (str): The actual feedback text
        prompt_id (str): ID of the prompt this feedback belongs to
        created_at (datetime): Timestamp when feedback was created
    """
    
    def __init__(self, text: str, start_offset: int, end_offset: int, 
                 feedback: str, prompt_id: Optional[str] = None):
        self.id = str(datetime.now().timestamp())
        self.text = text
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.feedback = feedback
        self.prompt_id = prompt_id
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert feedback to dictionary format."""
        return {
            "id": self.id,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "feedback": self.feedback,
            "prompt_id": self.prompt_id,
            "created_at": self.created_at.isoformat()
        }

class FeedbackStore:
    """
    Manages storage and retrieval of feedback.
    """
    
    def __init__(self):
        self.feedback: List[Feedback] = []
    
    def add_feedback(self, feedback: Feedback) -> Dict:
        """Add a new feedback to the store."""
        self.feedback.append(feedback)
        return feedback.to_dict()
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback in dictionary format."""
        return [feedback.to_dict() for feedback in self.feedback]
    
    def get_feedback_for_prompt(self, prompt_id: str) -> List[Dict]:
        """Get all feedback for a specific prompt."""
        return [feedback.to_dict() for feedback in self.feedback 
                if feedback.prompt_id == prompt_id]
    
    def analyze_feedback(self, prompt_id: Optional[str] = None) -> Dict:
        """Analyze feedback and provide insights."""
        relevant_feedback = [c for c in self.feedback 
                           if not prompt_id or c.prompt_id == prompt_id]
        
        return {
            "total_feedback": len(relevant_feedback),
            "feedback_sections": [{
                "text": feedback.text,
                "feedback": feedback.feedback,
                "position": {
                    "start": feedback.start_offset,
                    "end": feedback.end_offset
                }
            } for feedback in relevant_feedback]
        }
    
    def export_to_file(self, filename: str, prompt_id: Optional[str] = None):
        """Export feedback to a JSON file."""
        feedback_data = (self.get_feedback_for_prompt(prompt_id) 
                        if prompt_id else self.get_all_feedback())
        
        with open(filename, 'w') as f:
            json.dump(feedback_data, f, indent=2) 