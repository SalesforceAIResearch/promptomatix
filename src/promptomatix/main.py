"""
Main entry point for the promptomatix prompt optimization tool.
"""

import os
import sys
import json
import dspy
import nltk
import traceback
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import time
import ast
from openai import OpenAI, APITimeoutError, InternalServerError, RateLimitError, UnprocessableEntityError
import backoff
from dotenv import load_dotenv
import colorama
from colorama import Fore, Back, Style
from tqdm import tqdm
import threading
import queue

from .core.config import Config
from .core.optimizer import PromptOptimizer
from .core.session import SessionManager, OptimizationSession
from .core.feedback import Feedback, FeedbackStore
from .cli.parser import parse_args
from .utils.paths import SESSIONS_DIR

# Load environment variables from .env file
load_dotenv()

# Initialize global managers
session_manager = SessionManager()
feedback_store = FeedbackStore()

# Compatibility layer for the backend
class OptimizationSessionWrapper:
    """Wrapper class to maintain compatibility with the old API"""
    def __init__(self, session_manager):
        self.session_manager = session_manager
    
    def __getitem__(self, session_id):
        return self.session_manager.get_session(session_id)
    
    def __setitem__(self, session_id, session):
        # This won't be called directly, as sessions are managed through session_manager
        pass
    
    def __contains__(self, session_id):
        return self.session_manager.get_session(session_id) is not None
    
    def get(self, session_id, default=None):
        return self.session_manager.get_session(session_id) or default

# Create global instance for backward compatibility
optimization_sessions = OptimizationSessionWrapper(session_manager)


def process_input(**kwargs) -> Dict:
    """Process an initial optimization request."""
    session_id = kwargs.get('session_id', str(time.time()))
    session = None

    try:
        print("🚀 Starting Promptomatix optimization...")
        start_time = time.time()
        
        # Create config
        config = Config(**kwargs)
        config.session_id = session_id
        
        # Create session without saving
        session = session_manager.create_session(
            session_id=session_id,
            initial_input=config.task,
            config=config
        )
        
        # Initialize language model with configurable parameters
        lm = dspy.LM(
            config.model_name,
            api_key=config.model_api_key,
            api_base=config.model_api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        dspy.configure(lm=lm)
                
        # Create and run optimizer
        optimizer = PromptOptimizer(config)
        optimizer.lm = lm
        
        print("🎯 Running optimization...")
        result = optimizer.run(initial_flag=True)

        # Check if result contains an error
        if 'error' in result:
            print(f"❌ Optimization failed: {result['error']}")
            return result

        end_time = time.time()
        time_taken = round((end_time - start_time), 6)

        # Aggregate costs
        config_cost = getattr(config, 'llm_cost', 0)
        optimizer_cost = getattr(optimizer, 'llm_cost', 0)
        total_cost = config_cost + optimizer_cost
        if 'metrics' in result:
            result['metrics']['cost'] = total_cost
            result['metrics']['time_taken'] = time_taken
        else:
            result['metrics'] = {'cost': total_cost, 'time_taken': time_taken}
        
        # Add input and output fields to the result
        result['input_fields'] = config.input_fields
        result['output_fields'] = config.output_fields
        result['task_type'] = config.task_type

        # Update session with optimized prompt
        if isinstance(result.get('result'), str):
            session.update_optimized_prompt(result['result'])
                
        print("✅ Optimization completed successfully!")
        return result
            
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        
        print(f"❌ Optimization failed: {error_msg}")
        
        if session:
            session.logger.add_entry("ERROR", {
                "error": error_msg,
                "traceback": trace,
                "stage": "Initial Optimization"
            })
        
        return {
            'error': error_msg,
            'traceback': trace,
            'session_id': session_id if session_id else None
        }
    
def optimize_with_feedback(session_id: str) -> Dict:
    """
    Optimize prompt based on feedback for a given session.
    
    Args:
        session_id (str): Session identifier
        
    Returns:
        Dict: Optimization results
    """
    try:
        print("🔄 Optimizing with feedback...")
        start_time = time.time()

        session = session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Get the latest feedback for this session
        session_feedbacks = feedback_store.get_feedback_for_prompt(session_id)

        if not session_feedbacks:
            raise ValueError("No feedback found for this session")
        
        # Use the latest feedback
        latest_feedback = max(session_feedbacks, key=lambda x: x['created_at'])
        
        # Create feedback config
        feedback_config = Config(
            raw_input=f"Prompt: {session.latest_optimized_prompt}\n\nFeedback: {latest_feedback['feedback']}",
            original_raw_input=session.config.original_raw_input,
            synthetic_data_size=session.config.synthetic_data_size,
            train_ratio=session.config.train_ratio,
            task_type=session.config.task_type,
            model_name=session.config.model_name,
            model_provider=session.config.model_provider,
            model_api_key=session.config.model_api_key,
            model_api_base=session.config.model_api_base,
            dspy_module=session.config.dspy_module,
            session_id=session_id
        )
        
        # Initialize optimizer
        optimizer = PromptOptimizer(feedback_config)
        
        # Reset DSPy configuration for this thread
        dspy.settings.configure(reset=True)
        
        # Initialize language model
        lm = dspy.LM(
            feedback_config.model_name,
            api_key=feedback_config.model_api_key,
            api_base=feedback_config.model_api_base,
            temperature=feedback_config.temperature,
            max_tokens=feedback_config.max_tokens,
            cache=True
        )
        
        # Configure DSPy with the new LM instance
        dspy.configure(lm=lm)
        optimizer.lm = lm
        
        # Run optimization
        result = optimizer.run(initial_flag=False)

        # Check if result contains an error
        if 'error' in result:
            print(f"❌ Feedback optimization failed: {result['error']}")
            return result

        end_time = time.time()
        time_taken = round((end_time - start_time), 6)

        # Aggregate costs
        config_cost = getattr(feedback_config, 'llm_cost', 0)
        optimizer_cost = getattr(optimizer, 'llm_cost', 0)
        total_cost = config_cost + optimizer_cost
        if 'metrics' in result:
            result['metrics']['cost'] = total_cost
            result['metrics']['time_taken'] = time_taken
        else:
            result['metrics'] = {'cost': total_cost, 'time_taken': time_taken}

        # Add input and output fields to the result
        result['input_fields'] = feedback_config.input_fields
        result['output_fields'] = feedback_config.output_fields
        result['task_type'] = feedback_config.task_type

        # Update session with new optimized prompt if successful
        if isinstance(result.get('result'), str):
            session.update_optimized_prompt(result['result'])
        
        print("✅ Feedback optimization completed!")
        return result
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        
        print(f"❌ Feedback optimization failed: {error_msg}")
        
        if session:
            session.logger.add_entry("ERROR", {
                "error": error_msg,
                "traceback": trace,
                "stage": "Feedback Optimization"
            })
        
        return {
            'error': error_msg,
            'traceback': trace,
            'session_id': session_id,
            'result': None,
            'metrics': None
        }

def optimize_with_synthetic_feedback(session_id: str, synthetic_feedback: str) -> Dict:
    """
    Optimize prompt based on synthetic dataset feedback for a given session.
    
    Args:
        session_id (str): Session identifier
        synthetic_feedback (str): Feedback for synthetic dataset
        
    Returns:
        Dict: Optimization results
    """
    try:
        print("🤖 Optimizing with synthetic feedback...")
        start_time = time.time()

        session = session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Create feedback config with synthetic dataset feedback
        feedback_config = Config(
            raw_input=session.config.raw_input,
            original_raw_input=session.config.original_raw_input,
            synthetic_data_size=session.config.synthetic_data_size,
            train_ratio=session.config.train_ratio,
            task_type=session.config.task_type,
            model_name=session.config.model_name,
            model_provider=session.config.model_provider,
            model_api_key=session.config.model_api_key,
            model_api_base=session.config.model_api_base,
            dspy_module=session.config.dspy_module,
            session_id=session_id,
            synthetic_feedback=synthetic_feedback  # Add synthetic feedback to config
        )
        
        # Initialize optimizer
        optimizer = PromptOptimizer(feedback_config)
        
        # Create a new DSPy settings context for this thread
        with dspy.settings.context():
            # Initialize language model
            lm = dspy.LM(
                feedback_config.model_name,
                api_key=feedback_config.model_api_key,
                api_base=feedback_config.model_api_base,
                temperature=feedback_config.temperature,
                max_tokens=feedback_config.max_tokens,
                cache=True
            )
            
            # Configure DSPy with the new LM instance
            dspy.configure(lm=lm)
            optimizer.lm = lm
            
            # Run optimization
            result = optimizer.run(initial_flag=False)

            end_time = time.time()
            time_taken = round((end_time - start_time), 6)

            # Aggregate costs
            config_cost = getattr(feedback_config, 'llm_cost', 0)
            optimizer_cost = getattr(optimizer, 'llm_cost', 0)
            total_cost = config_cost + optimizer_cost
            if 'metrics' in result:
                result['metrics']['cost'] = total_cost
                result['metrics']['time_taken'] = time_taken
            else:
                result['metrics'] = {'cost': total_cost, 'time_taken': time_taken}

            # Update session with new optimized prompt if successful
            if isinstance(result.get('result'), str):
                session.update_optimized_prompt(result['result'])
            
            print("✅ Synthetic feedback optimization completed!")
            return result
            
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        
        print(f"❌ Synthetic feedback optimization failed: {error_msg}")
        
        if session:
            session.logger.add_entry("ERROR", {
                "error": error_msg,
                "traceback": trace,
                "stage": "Synthetic Data Feedback Optimization"
            })
        
        return {
            'error': error_msg,
            'traceback': trace,
            'session_id': session_id,
            'result': None,
            'metrics': None
        }

def save_feedback(text: str, start_offset: int, end_offset: int, 
                feedback: str, prompt_id: str) -> Dict:
    """
    Save a feedback for a prompt.
    
    Args:
        text (str): Text being feedbacked on
        start_offset (int): Feedback start position
        end_offset (int): Feedback end position
        feedback (str): Feedback text
        prompt_id (str): Associated prompt ID
        
    Returns:
        Dict: Saved feedback details
    """
    try:
        print("💾 Saving feedback...")

        new_feedback = Feedback(
            text=text,
            start_offset=start_offset,
            end_offset=end_offset,
            feedback=feedback,
            prompt_id=prompt_id
        )
        
        # Store feedback
        feedback_store.add_feedback(new_feedback)
        
        # Add to session if exists
        session = session_manager.get_session(prompt_id)
        if session:
            session.add_feedback(new_feedback)
        
        print("✅ Feedback saved successfully!")
        return new_feedback.to_dict()
        
    except Exception as e:
        print(f"❌ Failed to save feedback: {str(e)}")
        if prompt_id:
            session = session_manager.get_session(prompt_id)
            if session:
                session.logger.add_entry("ERROR", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                    "stage": "Feedback Addition"
            })
        raise

def load_session_from_file(session_file_path: str) -> Dict:
    """
    Load a session from a specific file path.
    
    Args:
        session_file_path (str): Path to the session file
        
    Returns:
        Dict: Session data or error information
    """
    try:
        print("📂 Loading session from file...")
        session = session_manager.load_session(session_file_path)
        if not session:
            print("❌ Failed to load session from file")
            return {
                'error': f'Failed to load session from {session_file_path}',
                'session_id': None
            }
        
        print("✅ Session loaded successfully!")
        return {
            'session_id': session.session_id,
            'initial_human_input': session.initial_human_input,
            'updated_human_input': session.updated_human_input,
            'latest_optimized_prompt': session.latest_optimized_prompt,
            'config': session.config.__dict__
        }
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        
        print(f"❌ Failed to load session: {error_msg}")
        return {
            'error': error_msg,
            'traceback': trace,
            'session_id': None
        }

def download_session(session_id: str, output_path: Optional[str] = None) -> Dict:
    """
    Download a session's data to a file.
    
    Args:
        session_id (str): The ID of the session to download
        output_path (str, optional): Path where to save the session file. 
                                   If not provided, saves in the sessions directory.
    
    Returns:
        Dict: Session data that was saved
    """
    try:
        print("📥 Downloading session...")
        session = session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Convert session to dictionary format
        session_data = session.to_dict()
        
        # Add additional metadata
        session_data.update({
            'timestamp': datetime.now().isoformat(),
            'versions': session_data.get('versions', []),
            'config': session_data.get('config', {}).__dict__
        })
        
        # Determine output path
        if not output_path:
            output_path = SESSIONS_DIR / f'session_{session_id}.json'
        else:
            output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        print(f"✅ Session saved to: {output_path}")
        return session_data
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"❌ Failed to download session: {error_msg}")
        return {
            'error': error_msg,
            'traceback': trace
        }

def upload_session(session_file_path: str) -> Dict:
    """
    Upload a session from a file.
    
    Args:
        session_file_path (str): Path to the session file to upload
    
    Returns:
        Dict: Loaded session data
    """
    try:
        print("📤 Uploading session...")
        # Load the session using the session manager
        session = session_manager.load_session_from_file(session_file_path)
        
        if not session:
            raise ValueError(f"Failed to load session from {session_file_path}")
        
        # Convert session to dictionary format
        session_data = session.to_dict()
        
        # Add additional metadata
        session_data.update({
            'timestamp': datetime.now().isoformat(),
            'versions': session_data.get('versions', []),
            'config': session_data.get('config', {}).__dict__
        })
        
        print(f"✅ Session uploaded successfully: {session.session_id}")
        return session_data
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"❌ Failed to upload session: {error_msg}")
        return {
            'error': error_msg,
            'traceback': trace
        }

def list_sessions() -> List[Dict]:
    """
    List all available sessions.
    
    Returns:
        List[Dict]: List of session metadata
    """
    try:
        print("📋 Listing sessions...")
        sessions = session_manager.list_sessions()
        print(f"✅ Found {len(sessions)} sessions")
        return [{
            'session_id': session['session_id'],
            'created_at': session['created_at'],
            'initial_input': session['initial_human_input'],
            'latest_optimized_prompt': session['latest_optimized_prompt']
        } for session in sessions]
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"❌ Failed to list sessions: {error_msg}")
        return []

def generate_feedback(
    optimized_prompt: str,
    input_fields: List[str],
    output_fields: List[str],
    model_name: str,
    model_api_key: str,
    model_api_base: str = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    synthetic_data: List[Dict] = None,
    session_id: str = None
) -> Dict:
    """
    Generate comprehensive feedback for an optimized prompt using synthetic data with explicit arguments.
    
    This function:
    1. Uses the provided optimized prompt and synthetic data
    2. For each synthetic data sample, uses generate_prompt_feedback_2 to create a feedback prompt
    3. Invokes an LLM with each feedback prompt to get individual feedback
    4. Collects all individual feedback and sends it to genrate_prompt_changes_prompt_2
    5. Returns the final comprehensive feedback
    
    Args:
        optimized_prompt (str): The optimized prompt to evaluate
        input_fields (List[str]): List of input field names to extract from synthetic data
        output_fields (List[str]): List of output field names to extract from synthetic data
        model_name (str): Name of the model to use for feedback generation
        model_api_key (str): API key for the model
        model_api_base (str, optional): API base URL for the model
        max_tokens (int): Maximum tokens for model responses
        temperature (float): Temperature setting for model responses
        synthetic_data (List[Dict], optional): Pre-generated synthetic data. If None, will generate new data
        session_id (str, optional): Session ID for logging purposes
        
    Returns:
        Dict: Generated feedback results including comprehensive feedback and individual sample feedback
    """
    try:
        print("🧠 Generating feedback...")
        start_time = time.time()
        
        if not optimized_prompt:
            raise ValueError("Optimized prompt is required")
        
        if not input_fields or not output_fields:
            raise ValueError("Input fields and output fields are required")
        
        # Initialize OpenAI client for feedback generation
        openai_client = OpenAI(api_key=model_api_key, base_url=model_api_base)
        
        # Import the feedback generation functions
        from promptomatix.core.prompts import generate_prompt_feedback_3, genrate_prompt_changes_prompt_2, generate_prompt_changes_prompt_3, generate_prompt_changes_prompt_4
        
        individual_feedbacks = []
        feedback_prompts = []
        
        # Process each synthetic data sample with fancy progress bar
        print("🔄 Processing synthetic data samples...")
        
        # Create a progress bar with custom styling
        pbar = tqdm(
            total=len(synthetic_data),
            desc="🧠 Generating feedback",
            unit="sample",
            ncols=100,
            bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green',
            leave=True
        )
        
        for i, sample in enumerate(synthetic_data):
            try:
                # Update progress bar description with current sample info
                pbar.set_description(f"🧠 Processing sample {i+1}/{len(synthetic_data)}")
                
                # Prepare user input (combine input fields)
                user_input = ""
                if isinstance(input_fields, (list, tuple)):
                    user_input = " ".join([str(sample.get(field, "")) for field in input_fields])
                else:
                    user_input = str(sample.get(input_fields, ""))
                
                # Get expected output
                expected_output = ""
                if isinstance(output_fields, (list, tuple)):
                    expected_output = " ".join([str(sample.get(field, "")) for field in output_fields])
                else:
                    expected_output = str(sample.get(output_fields, ""))
                
                # Generate AI system output using direct OpenAI API call instead of DSPy
                @backoff.on_exception(
                    backoff.expo,
                    (APITimeoutError, InternalServerError, RateLimitError, UnprocessableEntityError),
                    max_tries=3,
                    max_time=60
                )
                def get_ai_output(prompt):
                    response = openai_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content
                
                ai_system_output = get_ai_output(f"{optimized_prompt}\n\nInput: {user_input}")
                
                # Generate feedback prompt for this sample
                feedback_prompt = generate_prompt_feedback_3(
                    user_input=user_input,
                    ai_system_output=ai_system_output,
                    expected_output=expected_output,
                    prompts_used=optimized_prompt
                )
                
                feedback_prompts.append(feedback_prompt)
                
                # Get feedback from OpenAI API with retry logic
                @backoff.on_exception(
                    backoff.expo,
                    (APITimeoutError, InternalServerError, RateLimitError, UnprocessableEntityError),
                    max_tries=3,
                    max_time=60
                )
                def get_openai_feedback(prompt):
                    response = openai_client.chat.completions.create(
                        model="o3",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content
                
                feedback_response = get_openai_feedback(feedback_prompt)
                individual_feedbacks.append({
                    'sample_index': i,
                    'user_input': user_input,
                    'expected_output': expected_output,
                    'ai_output': ai_system_output,
                    'feedback': feedback_response
                })
                
                # Update progress bar after successful processing
                pbar.update(1)
                pbar.set_postfix({
                    'Success': len(individual_feedbacks),
                    'Failed': i + 1 - len(individual_feedbacks)
                })
                
            except Exception as e:
                # Update progress bar even on error
                pbar.update(1)
                pbar.set_postfix({
                    'Success': len(individual_feedbacks),
                    'Failed': i + 1 - len(individual_feedbacks)
                })
                
                # Log error if session is available
                if session_id:
                    session = session_manager.get_session(session_id)
                    if session:
                        session.logger.add_entry("ERROR", {
                            "error": f"Error processing sample {i}: {str(e)}",
                            "sample": sample,
                            "stage": "Individual Feedback Generation"
                        })
                continue
        
        pbar.close()
        
        if not individual_feedbacks:
            raise ValueError("No individual feedback was generated successfully")
        
        # Combine all individual feedback
        feedback_list = "\n###\n".join([
            f"Sample {fb['sample_index'] + 1}:\nUser Input: {fb['user_input']}\nExpected: {fb['expected_output']}\nAI Output: {fb['ai_output']}\nFeedback: {fb['feedback']}"
            for fb in individual_feedbacks
        ])
        
        # Generate comprehensive feedback using OpenAI API
        comprehensive_feedback_prompt = generate_prompt_changes_prompt_4(optimized_prompt, feedback_list)
        
        @backoff.on_exception(
            backoff.expo,
            (APITimeoutError, InternalServerError, RateLimitError, UnprocessableEntityError),
            max_tries=3,
            max_time=60
        )
        def get_comprehensive_feedback(prompt):
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        comprehensive_feedback = get_comprehensive_feedback(comprehensive_feedback_prompt)
        
        end_time = time.time()
        time_taken = round((end_time - start_time), 6)
        
        # Calculate costs - since we're not using DSPy, we'll need to track costs manually
        # For now, we'll set it to 0 since we don't have cost tracking for direct API calls
        total_cost = 0  # You may want to implement cost tracking for direct API calls
        
        result = {
            'session_id': session_id,
            'comprehensive_feedback': comprehensive_feedback,
            'individual_feedbacks': individual_feedbacks,
            'synthetic_data_used': len(synthetic_data),
            'metrics': {
                'time_taken': time_taken,
                'cost': total_cost,
                'samples_processed': len(individual_feedbacks)
            }
        }
        
        # Store the comprehensive feedback in the session for later use if session_id is provided
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                session.comprehensive_feedback = comprehensive_feedback
                session.individual_feedbacks = individual_feedbacks
        
        print(f"[DEBUG] Feedback generation time_taken (ms): {time_taken}")
        
        print("✅ Feedback generation completed!")
        return result
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        
        # Log error if session is available
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                session.logger.add_entry("ERROR", {
                    "error": error_msg,
                    "traceback": trace,
                    "stage": "Feedback Generation"
                })
        
        print(f"❌ Failed to generate feedback: {error_msg}")
        return {
            'error': error_msg,
            'traceback': trace,
            'session_id': session_id,
            'comprehensive_feedback': None,
            'individual_feedbacks': [],
            'metrics': None
        }

def display_fancy_result(result: Dict) -> None:
    """
    Display optimization results in a fancy, formatted way.
    
    Args:
        result (Dict): The result dictionary from process_input
    """
    # Try to import colorama, fallback to plain text if not available
    try:
        import colorama
        from colorama import Fore, Back, Style
        colorama.init()
        USE_COLORS = True
    except ImportError:
        # Fallback colors for systems without colorama
        class DummyColors:
            def __getattr__(self, name):
                return ""
        Fore = Back = Style = DummyColors()
        USE_COLORS = False
    
    from datetime import datetime
    
    # Check for errors first
    if 'error' in result:
        print(f"\n{Fore.RED}❌ Optimization Failed{Style.RESET_ALL}")
        print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        if 'traceback' in result:
            print(f"{Fore.YELLOW}Traceback: {result['traceback']}{Style.RESET_ALL}")
        return
    
    # Header
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{' PROMPTOMATIX OPTIMIZATION RESULTS':^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Session Info
    print(f"\n{Fore.BLUE}📋 Session Information{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Session ID: {Fore.YELLOW}{result.get('session_id', 'N/A')}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Backend: {Fore.YELLOW}{result.get('backend', 'N/A')}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Task Type: {Fore.YELLOW}{result.get('task_type', 'N/A')}{Style.RESET_ALL}")
    
    # Task Configuration
    print(f"\n{Fore.BLUE}⚙️  Task Configuration{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Input Fields: {Fore.YELLOW}{result.get('input_fields', 'N/A')}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Output Fields: {Fore.YELLOW}{result.get('output_fields', 'N/A')}{Style.RESET_ALL}")
    
    # Metrics
    metrics = result.get('metrics', {})
    if metrics:
        print(f"\n{Fore.BLUE}📊 Performance Metrics{Style.RESET_ALL}")
        
        # Scores
        if 'initial_prompt_score' in metrics and 'optimized_prompt_score' in metrics:
            initial_score = metrics['initial_prompt_score']
            optimized_score = metrics['optimized_prompt_score']
            improvement = optimized_score - initial_score
            
            print(f"{Fore.WHITE}Initial Score: {Fore.RED}{initial_score:.4f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Optimized Score: {Fore.GREEN}{optimized_score:.4f}{Style.RESET_ALL}")
            
            if improvement > 0:
                print(f"{Fore.WHITE}Improvement: {Fore.GREEN}+{improvement:.4f} ({improvement/initial_score*100:.1f}%){Style.RESET_ALL}")
            else:
                print(f"{Fore.WHITE}Change: {Fore.RED}{improvement:.4f} ({improvement/initial_score*100:.1f}%){Style.RESET_ALL}")
        
        # Cost and Time
        if 'cost' in metrics:
            print(f"{Fore.WHITE}Total Cost: {Fore.YELLOW}${metrics['cost']:.6f}{Style.RESET_ALL}")
        if 'time_taken' in metrics:
            print(f"{Fore.WHITE}Processing Time: {Fore.YELLOW}{metrics['time_taken']:.3f}s{Style.RESET_ALL}")
    
    # Prompts
    print(f"\n{Fore.BLUE} Prompt Comparison{Style.RESET_ALL}")
    
    # Original Prompt
    if 'initial_prompt' in result:
        print(f"\n{Fore.WHITE}Original Prompt:{Style.RESET_ALL}")
        print(f"{Fore.RED}{'─'*40}{Style.RESET_ALL}")
        print(f"{result['initial_prompt']}")
        print(f"{Fore.RED}{'─'*40}{Style.RESET_ALL}")
    
    # Optimized Prompt
    if 'result' in result:
        print(f"\n{Fore.WHITE}Optimized Prompt:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'─'*40}{Style.RESET_ALL}")
        print(f"{result['result']}")
        print(f"{Fore.GREEN}{'─'*40}{Style.RESET_ALL}")
    
    # Synthetic Data Summary
    synthetic_data = result.get('synthetic_data', [])
    if synthetic_data:
        print(f"\n{Fore.BLUE}📚 Synthetic Data Generated{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Total Samples: {Fore.YELLOW}{len(synthetic_data)}{Style.RESET_ALL}")
        
        # Show a few examples
        if len(synthetic_data) > 0:
            print(f"\n{Fore.WHITE}Sample Data:{Style.RESET_ALL}")
            for i, sample in enumerate(synthetic_data[:3]):  # Show first 3 samples
                print(f"{Fore.YELLOW}Sample {i+1}:{Style.RESET_ALL}")
                for key, value in sample.items():
                    print(f"  {Fore.WHITE}{key}:{Style.RESET_ALL} {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            if len(synthetic_data) > 3:
                print(f"{Fore.YELLOW}... and {len(synthetic_data) - 3} more samples{Style.RESET_ALL}")
    
    # Footer
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'✨ Optimization Complete!':^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Reset colorama
    colorama.deinit()

def main():
    """Main entry point for the CLI application."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Process optimization
        if args.get('raw_input') or args.get('huggingface_dataset_name'):
            result = process_input(**args)
            display_fancy_result(result)  # Replace the JSON dump with fancy display
            return
        
        # Handle feedback management commands
        if args.get('list_feedbacks'):
            print(json.dumps(feedback_store.get_all_feedbacks(), indent=2))
            return
            
        if args.get('analyze_feedbacks'):
            analysis = feedback_store.analyze_feedbacks(args.get('prompt_id'))
            print(json.dumps(analysis, indent=2))
            return
            
        if args.get('export_feedbacks'):
            feedback_store.export_to_file(
                args.get('export_feedbacks'),
                args.get('prompt_id')
            )
            return
        
        print("No valid command specified. Use --help for usage information.")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
