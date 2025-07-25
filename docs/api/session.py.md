# session.py API Documentation

This file documents all classes, functions, and the high-level flow for `session.py` in detail.

---

## High-Level Flow and Stages

The `session.py` module manages the lifecycle and state of prompt optimization sessions, including session creation, feedback management, prompt updates, and persistent storage. It provides both single-session and multi-session management, with logging and serialization support.

1. **Session Initialization**
   - `OptimizationSession` is initialized with a session ID, initial human input, and a `Config` object.
   - Sets up logging and feedback storage.

2. **Feedback and Prompt Management**
   - Methods to add feedback, update prompts, and update human input.
   - All changes are logged for traceability.

3. **Session Serialization**
   - Sessions can be converted to dictionaries for storage or inspection.
   - `SessionManager` handles saving and loading sessions from disk.

4. **Session Management**
   - `SessionManager` creates, retrieves, updates, and lists multiple sessions.
   - Ensures persistence and directory management for session files.

---

## Class: OptimizationSession

```
class OptimizationSession
```
Manages the state and lifecycle of a prompt optimization session.

**Attributes**
- `session_id` (str): Unique identifier for the session
- `initial_human_input` (str): Original prompt from user
- `updated_human_input` (str): Current version of the prompt
- `latest_optimized_prompt` (str): Most recent optimized prompt
- `config` (Config): Session configuration
- `created_at` (datetime): Session creation timestamp
- `logger` (SessionLogger): Session-specific logger
- `comprehensive_feedback` (str): Comprehensive feedback for the session
- `individual_feedbacks` (List[Dict]): Individual feedback for the session

**Methods**
- `__init__(session_id: str, initial_human_input: str, config: Config)`: Initialize a new optimization session.
- `add_feedback(feedback: Feedback) -> None`: Add a new feedback to the session.
- `update_optimized_prompt(new_prompt: str) -> None`: Update the latest optimized prompt.
- `update_human_input(new_input: str) -> None`: Update the human input prompt.
- `to_dict() -> Dict`: Convert session to dictionary format.

---

## Class: SessionManager

```
class SessionManager
```
Manages multiple optimization sessions with persistence.

**Attributes**
- `sessions` (Dict[str, OptimizationSession]): Dictionary of session_id to session objects
- `sessions_dir` (Path): Directory for session files

**Methods**
- `__init__()`: Initialize the session manager and ensure session directory exists.
- `load_session_from_file(session_file_path: str) -> Optional[OptimizationSession]`: Load a specific session from a file path.
- `create_session(session_id: str, initial_input: str, config: Config) -> OptimizationSession`: Create and store a new optimization session.
- `get_session(session_id: str) -> Optional[OptimizationSession]`: Retrieve a session by ID.
- `update_session(session: OptimizationSession)`: Update a session and persist changes.
- `list_sessions() -> List[Dict]`: List all active sessions.

---

## Internal and Utility Methods

- **_save_session**
  ```
  def _save_session(self, session: OptimizationSession)
  ```
  Saves the session data to a file in the sessions directory, serializing the session and its configuration. Handles conversion of non-serializable objects and ensures directory existence.

- **to_dict (OptimizationSession)**
  ```
  def to_dict(self) -> Dict
  ```
  Converts the session object to a dictionary, including all feedback, prompt states, and configuration. Handles serialization of feedback and timestamps.

- **load_session_from_file**
  ```
  def load_session_from_file(self, session_file_path: str) -> Optional[OptimizationSession]
  ```
  Loads a session from a JSON file, reconstructing the session and its configuration. Handles enum and object deserialization.

- **add_feedback**
  ```
  def add_feedback(self, feedback: Feedback) -> None
  ```
  Adds a feedback object to the session and logs the action.

- **update_optimized_prompt**
  ```
  def update_optimized_prompt(self, new_prompt: str) -> None
  ```
  Updates the latest optimized prompt and logs the update.

- **update_human_input**
  ```
  def update_human_input(self, new_input: str) -> None
  ```
  Updates the human input prompt and logs the update.

- **create_session**
  ```
  def create_session(self, session_id: str, initial_input: str, config: Config) -> OptimizationSession
  ```
  Creates a new session, stores it in the manager, and persists it to disk.

- **get_session**
  ```
  def get_session(self, session_id: str) -> Optional[OptimizationSession]
  ```
  Retrieves a session by its unique ID from the manager's dictionary.

- **update_session**
  ```
  def update_session(self, session: OptimizationSession)
  ```
  Updates a session in the manager and saves it to disk.

- **list_sessions**
  ```
  def list_sessions(self) -> List[Dict]
  ```
  Returns a list of all active sessions as dictionaries.

--- 