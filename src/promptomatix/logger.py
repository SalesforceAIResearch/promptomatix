import logging
from datetime import datetime
from typing import Dict, Any
import json

class SessionLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_entries = []
        self.start_time = datetime.now()

    def add_entry(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.log_entries.append(entry)

    def format_log(self) -> str:
        """Format the log in a human-readable way"""
        output = []
        
        # Session Header
        output.append("="*80)
        output.append("PROMPT OPTIMIZATION SESSION LOG")
        output.append("="*80)
        output.append(f"Session ID: {self.session_id}")
        output.append(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Duration: {(datetime.now() - self.start_time).total_seconds():.2f} seconds")
        output.append("="*80 + "\n")

        # Process each log entry
        for entry in self.log_entries:
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
            details = entry["details"]
            
            # Event Header
            output.append(f"[{timestamp}] {entry['event_type']}")
            output.append("-"*40)
            
            # LLM Calls
            if entry["event_type"] == "LLM_CALL":
                output.append(f"Function: {details.get('function', 'Unknown')}")
                output.append(f"Stage: {details.get('stage', 'Unknown')}")
                
                if details.get('stage') == 'before':
                    output.append("\nPrompt to LLM:")
                    output.append(f"{details.get('prompt', 'N/A')}")
                    output.append("\nModel Configuration:")
                    output.append(f"  - Model: {details.get('model', 'N/A')}")
                    output.append(f"  - Temperature: {details.get('temperature', 'N/A')}")
                    output.append(f"  - Max Tokens: {details.get('max_tokens', 'N/A')}")
                
                if details.get('stage') == 'after':
                    output.append("\nLLM Response:")
                    output.append(f"{details.get('response', 'N/A')}")
            
            # Function Calls
            if entry["event_type"] == "FUNCTION_CALL":
                output.append(f"Function: {details.get('function', 'Unknown')}")
                
                if "input" in details:
                    output.append("\nFunction Input:")
                    if isinstance(details["input"], dict):
                        for key, value in details["input"].items():
                            output.append(f"  - {key}: {value}")
                    else:
                        output.append(f"  {details['input']}")
                
                if "output" in details:
                    output.append("\nFunction Output:")
                    if isinstance(details["output"], dict):
                        for key, value in details["output"].items():
                            output.append(f"  - {key}: {value}")
                    else:
                        output.append(f"  {details['output']}")
            
            # User Actions
            if "user_action" in details:
                output.append(f"User Action: {details['user_action']}")
            
            # API Calls
            if "api_endpoint" in details:
                output.append(f"API Endpoint: {details['api_endpoint']}")
                output.append(f"Method: {details.get('method', 'N/A')}")
            
            # Model Operations
            if "model_config" in details:
                output.append("\nModel Configuration:")
                for key, value in details["model_config"].items():
                    output.append(f"  - {key}: {value}")
            
            # Synthetic Data Generation
            if "synthetic_data" in details:
                output.append("\nSynthetic Data Generation:")
                output.append(f"  - Size: {details['synthetic_data'].get('size', 'N/A')}")
                output.append(f"  - Train/Test Split: {details['synthetic_data'].get('split_ratio', 'N/A')}")
            
            # Comments/Feedback
            if "comments" in details:
                output.append("\nUser Feedback:")
                for comment in details["comments"]:
                    output.append(f"  - Selected Text: \"{comment['text']}\"")
                    output.append(f"    Comment: \"{comment['comment']}\"")
            
            # Optimization Results
            if "optimization_results" in details:
                output.append("\nOptimization Results:")
                results = details["optimization_results"]
                if "metrics" in results:
                    output.append("  Metrics:")
                    output.append(f"    {results['metrics']}")
            
            # Errors
            if "error" in details:
                output.append("\nError:")
                output.append(f"  {details['error']}")
                if "traceback" in details:
                    output.append("  Traceback:")
                    output.append(f"  {details['traceback']}")
            
            # Section Separator
            output.append("\n" + "-"*80 + "\n")

        return "\n".join(output)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "entries": self.log_entries
        } 