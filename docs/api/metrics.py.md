# metrics.py API Documentation

This file documents all classes, functions, and the high-level flow for `metrics.py` in detail.

---

## High-Level Flow and Stages

The `metrics.py` module provides a comprehensive suite of evaluation metrics for NLP tasks, including QA, classification, generation, summarization, translation, code, and more. It supports dynamic metric selection based on task type, detailed metric breakdowns, and advanced scoring using BERTScore, ROUGE, BLEU, and custom logic. The module also manages output field configuration and applies length penalties for prompt optimization.

1. **Configuration**
   - `MetricsManager.configure(output_fields)`: Sets up output fields for metric extraction.

2. **Metric Selection**
   - `get_metrics_for_task(task_type)`: Returns the appropriate metric function for the given task type.
   - `get_final_eval_metrics(task_type)`: Returns the final evaluation metric function for the task type.

3. **Metric Computation**
   - Each metric function computes a score (0-1) for a prediction/example pair, using BERTScore, ROUGE, BLEU, F1, accuracy, and other relevant metrics.
   - Length penalties are applied based on prompt length and lambda penalty.

4. **Detailed Analysis**
   - `get_detailed_metrics`: Provides a breakdown of all relevant metrics for analysis and debugging.

5. **Advanced Handling**
   - Automatic language detection for translation metrics.
   - Suppression of warnings and stderr for clean metric computation.

---

## Class: MetricsManager

```
class MetricsManager
```
Central manager for all evaluation metrics in prompt optimization and NLP workflows.

**Attributes**
- `_output_fields` (List[str]): Class-level storage for output field names used in metric extraction.

**Methods**
- `configure(output_fields: List[str]) -> None`: Configure the MetricsManager with output fields.
- `get_metrics_for_task(task_type: str) -> Callable`: Returns the metric function for the given task type.
- `get_final_eval_metrics(task_type: str) -> Callable`: Returns the final evaluation metric function for the task type.
- `get_detailed_metrics(task_type: str, example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> Dict[str, float]`: Returns a detailed breakdown of all metrics for analysis purposes.

---

## Internal and Utility Methods

- **_get_output_value**
  ```
  def _get_output_value(item: Any) -> str
  ```
  Helper method to extract the output value from an item using the configured output fields. Returns a concatenated string of all present output fields.

- **_qa_metrics**
  ```
  def _qa_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Evaluates QA predictions using BERTScore and exact match, with length penalty.

- **_classification_metrics**
  ```
  def _classification_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Computes accuracy for classification tasks, with length penalty.

- **_generation_metrics**
  ```
  def _generation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Computes metrics for text generation tasks using BERTScore for fluency, creativity, and similarity, with length penalty.

- **_summarization_metrics**
  ```
  def _summarization_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Computes metrics for summarization tasks using BERTScore, with length penalty.

- **_translation_metrics**
  ```
  def _translation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Evaluates translation predictions using BERTScore with automatic language detection, with length penalty.

- **_default_metrics**
  ```
  def _default_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Default metric (exact match) for unknown task types, with length penalty.

- **_final_eval_metrics**
  ```
  def _final_eval_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Final evaluation metric for generic tasks using BERTScore.

- **_qa_metrics_final_eval**
  ```
  def _qa_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Final evaluation for QA using BERTScore and exact match.

- **_classification_metrics_final_eval**
  ```
  def _classification_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Final evaluation for classification using accuracy.

- **_generation_metrics_final_eval**
  ```
  def _generation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Final evaluation for generation using BERTScore for fluency, creativity, and similarity.

- **_summarization_metrics_final_eval**
  ```
  def _summarization_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Final evaluation for summarization using BERTScore.

- **_translation_metrics_final_eval**
  ```
  def _translation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Final evaluation for translation using BERTScore with language detection.

- **_multi_label_classification_metrics** / **_multi_label_classification_metrics_final_eval**
  ```
  def _multi_label_classification_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _multi_label_classification_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for multi-label classification using F1 and Hamming similarity.

- **_information_extraction_metrics** / **_information_extraction_metrics_final_eval**
  ```
  def _information_extraction_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _information_extraction_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for information extraction using F1 and structure similarity.

- **_paraphrasing_metrics** / **_paraphrasing_metrics_final_eval**
  ```
  def _paraphrasing_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _paraphrasing_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for paraphrasing using semantic similarity and lexical diversity.

- **_conversation_metrics** / **_conversation_metrics_final_eval**
  ```
  def _conversation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _conversation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for conversation using response similarity and coherence.

- **_negotiation_metrics** / **_negotiation_metrics_final_eval**
  ```
  def _negotiation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _negotiation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for negotiation using response appropriateness and effectiveness.

- **_code_generation_metrics** / **_code_generation_metrics_final_eval**
  ```
  def _code_generation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _code_generation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for code generation using code similarity and quality.

- **_code_explanation_metrics** / **_code_explanation_metrics_final_eval**
  ```
  def _code_explanation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _code_explanation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for code explanation using explanation similarity and quality.

- **_code_completion_metrics** / **_code_completion_metrics_final_eval**
  ```
  def _code_completion_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _code_completion_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for code completion using completion similarity and quality.

- **_code_debugging_metrics** / **_code_debugging_metrics_final_eval**
  ```
  def _code_debugging_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _code_debugging_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for code debugging using debug similarity and quality.

- **_planning_metrics** / **_planning_metrics_final_eval**
  ```
  def _planning_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _planning_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for planning using plan similarity and quality.

- **_tool_use_metrics** / **_tool_use_metrics_final_eval**
  ```
  def _tool_use_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _tool_use_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for tool use using usage similarity and quality.

- **_decision_making_metrics** / **_decision_making_metrics_final_eval**
  ```
  def _decision_making_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _decision_making_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for decision making using decision similarity and quality.

- **_process_automation_metrics** / **_process_automation_metrics_final_eval**
  ```
  def _process_automation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _process_automation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for process automation using process similarity and quality.

- **_reasoning_metrics** / **_reasoning_metrics_final_eval**
  ```
  def _reasoning_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _reasoning_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for reasoning using reasoning similarity and quality.

- **_recommendation_metrics** / **_recommendation_metrics_final_eval**
  ```
  def _recommendation_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _recommendation_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for recommendation using recommendation similarity and quality.

- **_data_analysis_metrics** / **_data_analysis_metrics_final_eval**
  ```
  def _data_analysis_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  def _data_analysis_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float
  ```
  Metrics for data analysis using analysis similarity and quality.

--- 