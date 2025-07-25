# utils API Documentation

This file documents all utility modules in the `utils/` directory, including their high-level flow, classes, functions, and internal methods.

---

## High-Level Flow and Purpose

The `utils` package provides foundational utilities for the Promptomatix system, including path management, logging, and text parsing. These modules are used throughout the codebase to ensure consistent directory structure, robust logging, and reliable parsing of text and data structures.

---

## Module: paths.py

Manages application paths and directories, ensuring all required folders exist for logs and session data.

**Key Variables**
- `PROJECT_ROOT`: The root directory of the project.
- `LOGS_DIR`: Directory for all logs.
- `SESSIONS_DIR`: Directory for session data.
- `SESSION_LOGS_DIR`, `CONFIG_LOGS_DIR`, `OPTIMIZER_LOGS_DIR`: Subdirectories for different log types.

**Behavior**
- Automatically creates all necessary directories if they do not exist.

---

## Module: logging.py

Handles logging functionality for optimization sessions, including both application and DSPy-specific logs.

### Class: SessionLogger

```
class SessionLogger
```
Handles logging for individual optimization sessions.

**Attributes**
- `session_id` (str): ID of the session being logged
- `app_logger` (logging.Logger): Logger for application events
- `dspy_logger` (logging.Logger): Logger for DSPy-specific events

**Methods**
- `__init__(session_id: str)`: Initialize a new session logger, creating log files for the session.
- `add_entry(entry_type: str, data: Dict[str, Any]) -> None`: Add a new log entry (as JSON) to the application log. Errors are also logged at error level.

---

## Module: parsing.py

Provides utility functions for parsing and cleaning text data, especially dictionary-like strings from LLM outputs or user input.

**Functions**
- `parse_dict_strings(text: str) -> str`: Parse and clean dictionary strings from various formats, handling smart quotes and malformed JSON.
- `_clean_parsed_data(data: Union[List, Dict]) -> str`: Clean parsed JSON data, escaping problematic characters.
- `_manual_dict_parse(text: str) -> str`: Manually parse dictionary-like strings if JSON parsing fails.
- `_split_dict_pairs(content: str) -> List[str]`: Split dictionary content into key-value pairs, respecting quoted strings.
- `_process_dict_pairs(pairs: List[str]) -> Dict[str, str]`: Process key-value pairs into a clean dictionary.
- `_clean_dict_value(value: str) -> str`: Clean and format dictionary values, escaping special characters.

---

## Internal and Utility Methods

### paths.py
- Directory variables are initialized and created at import time, ensuring all required folders exist for logs and sessions.

### logging.py
- **SessionLogger.__init__**
  ```
  def __init__(self, session_id: str)
  ```
  Initializes loggers and creates log files for the session, one for application events and one for DSPy events.

- **SessionLogger.add_entry**
  ```
  def add_entry(self, entry_type: str, data: Dict[str, Any]) -> None
  ```
  Adds a log entry as a JSON object to the application log. Errors are also logged at error level.

### parsing.py
- **parse_dict_strings**
  ```
  def parse_dict_strings(text: str) -> str
  ```
  Parses and cleans dictionary strings, handling smart quotes and malformed JSON.

- **_clean_parsed_data**
  ```
  def _clean_parsed_data(data: Union[List, Dict]) -> str
  ```
  Cleans parsed JSON data, escaping problematic characters.

- **_manual_dict_parse**
  ```
  def _manual_dict_parse(text: str) -> str
  ```
  Manually parses dictionary-like strings if JSON parsing fails.

- **_split_dict_pairs**
  ```
  def _split_dict_pairs(content: str) -> List[str]
  ```
  Splits dictionary content into key-value pairs, respecting quoted strings.

- **_process_dict_pairs**
  ```
  def _process_dict_pairs(pairs: List[str]) -> Dict[str, str]
  ```
  Processes key-value pairs into a clean dictionary.

- **_clean_dict_value**
  ```
  def _clean_dict_value(value: str) -> str
  ```
  Cleans and formats dictionary values, escaping special characters.

--- 