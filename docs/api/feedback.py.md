# feedback.py API Documentation

This file documents all classes and functions for `feedback.py` in detail.

---

### Class: Feedback

```
class Feedback
```
Represents a feedback or annotation on a prompt.

**Attributes**
- `id` (str): Unique identifier for the feedback
- `text` (str): The text being given feedback on
- `start_offset` (int): Starting position of the feedback
- `end_offset` (int): Ending position of the feedback
- `feedback` (str): The actual feedback text
- `prompt_id` (str): ID of the prompt this feedback belongs to
- `created_at` (datetime): Timestamp when feedback was created

**Methods**
- `__init__(text: str, start_offset: int, end_offset: int, feedback: str, prompt_id: Optional[str] = None)`: Initializes a Feedback object.
- `to_dict() -> Dict`: Convert feedback to dictionary format.

---

### Class: FeedbackStore

```
class FeedbackStore
```
Manages storage and retrieval of feedback.

**Methods**
- `__init__()`: Initializes the feedback store.
- `add_feedback(feedback: Feedback) -> Dict`: Add a new feedback to the store.
- `get_all_feedback() -> List[Dict]`: Get all feedback in dictionary format.
- `get_feedback_for_prompt(prompt_id: str) -> List[Dict]`: Get all feedback for a specific prompt.
- `analyze_feedback(prompt_id: Optional[str] = None) -> Dict`: Analyze feedback and provide insights (total count, feedback sections, etc).
- `export_to_file(filename: str, prompt_id: Optional[str] = None)`: Export feedback to a JSON file (optionally filtered by prompt).

--- 