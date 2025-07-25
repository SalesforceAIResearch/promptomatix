# main.py API Documentation

This file documents the main entry point for the Promptomatix prompt optimization tool. It is the most important file in the system, orchestrating CLI usage, session management, feedback integration, optimization workflows, and user interaction.

---

## High-Level Flow and Stages

The `main.py` module is the central hub for all prompt optimization workflows in Promptomatix. It provides:

1. **CLI Entry Point**
   - Parses command-line arguments and dispatches to the appropriate workflow (optimization, feedback, session management).

2. **Session and Feedback Management**
   - Manages creation, retrieval, updating, and persistence of optimization sessions.
   - Handles user and synthetic feedback, storing and associating it with sessions.

3. **Prompt Optimization**
   - Orchestrates the full optimization pipeline, including configuration, LLM setup, optimizer execution, and result aggregation.
   - Supports optimization with initial input, user feedback, or synthetic feedback.

4. **Result Presentation**
   - Provides a rich, colorized CLI output for optimization results, including metrics, prompt comparison, and synthetic data samples.

5. **Session File Operations**
   - Supports loading, downloading, uploading, and listing sessions for reproducibility and sharing.

6. **Feedback Generation**
   - Automates the process of generating comprehensive feedback using LLMs and synthetic data, with robust error handling and progress reporting.

---

## Main Functions and Workflows

### process_input
```
def process_input(**kwargs) -> Dict
```
Processes an initial optimization request, creating a session, configuring the optimizer, running the optimization, and returning results. Handles all error logging and cost/time aggregation.

### optimize_with_feedback
```
def optimize_with_feedback(session_id: str) -> Dict
```
Optimizes a prompt based on the latest user feedback for a given session. Updates the session with new results and metrics.

### optimize_with_synthetic_feedback
```
def optimize_with_synthetic_feedback(session_id: str, synthetic_feedback: str) -> Dict
```
Optimizes a prompt using synthetic dataset feedback for a given session. Updates the session with new results and metrics.

### save_feedback
```
def save_feedback(text: str, start_offset: int, end_offset: int, feedback: str, prompt_id: str) -> Dict
```
Saves user feedback for a prompt, associates it with the session, and logs the action.

### load_session_from_file
```
def load_session_from_file(session_file_path: str) -> Dict
```
Loads a session from a file, returning session data or error information.

### download_session
```
def download_session(session_id: str, output_path: Optional[str] = None) -> Dict
```
Downloads a session's data to a file, including all metadata and configuration.

### upload_session
```
def upload_session(session_file_path: str) -> Dict
```
Uploads a session from a file, reconstructing the session in the system.

### list_sessions
```
def list_sessions() -> List[Dict]
```
Lists all available sessions with metadata.

### generate_feedback
```
def generate_feedback(optimized_prompt: str, input_fields: List[str], output_fields: List[str], model_name: str, model_api_key: str, model_api_base: str = None, max_tokens: int = 1000, temperature: float = 0.7, synthetic_data: List[Dict] = None, session_id: str = None) -> Dict
```
Generates comprehensive feedback for an optimized prompt using synthetic data and LLM calls. Handles batching, error handling, and progress reporting.

### display_fancy_result
```
def display_fancy_result(result: Dict) -> None
```
Displays optimization results in a rich, colorized, and user-friendly format in the CLI, including metrics, prompt comparison, and synthetic data samples.

### main
```
def main()
```
Main entry point for the CLI application. Parses arguments, dispatches to the correct workflow, and handles all top-level errors.

---

## Internal and Utility Methods

- **OptimizationSessionWrapper**
  ```
  class OptimizationSessionWrapper
  ```
  Compatibility layer for session management, providing dict-like access to sessions for backward compatibility.

- **Global Managers**
  - `session_manager`: Manages all optimization sessions.
  - `feedback_store`: Manages all feedback objects and associations.

- **Error Handling and Logging**
  - All major workflows include robust try/except blocks, logging errors to session logs and providing detailed tracebacks.

- **LLM Integration**
  - Uses DSPy and OpenAI APIs for optimization and feedback generation, with retry logic and exponential backoff for reliability.

- **Progress Reporting**
  - Uses tqdm for progress bars and colorama for colorized CLI output.

- **CLI Argument Parsing**
  - Uses a custom parser to support a wide range of commands, including optimization, feedback management, and session operations.

---

## CLI Usage and Workflows

- **Prompt Optimization**
  - Run with a raw prompt or dataset to optimize and display results.
- **Feedback Management**
  - Save, list, analyze, and export feedback for any session.
- **Session Management**
  - Download, upload, and list sessions for reproducibility and sharing.
- **Feedback Generation**
  - Generate comprehensive feedback using LLMs and synthetic data for iterative improvement.

---

## Best Practices and Notes

- Always check CLI output for errors and tracebacks for debugging.
- Use session download/upload to save and restore optimization states.
- Use feedback workflows to iteratively improve prompt quality.
- The main.py file is the orchestrator—refer to this documentation for understanding the full system flow.

--- 