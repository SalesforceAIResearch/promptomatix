# optimizer.py API Documentation

This file documents all classes, functions, and the high-level flow for `optimizer.py` in detail.

---

## High-Level Flow and Stages

The `optimizer.py` module orchestrates the entire prompt optimization process. Here is the high-level flow and what happens at each stage:

1. **Initialization**
   - The `PromptOptimizer` class is initialized with a `Config` object.
   - Sets up logging and determines the backend (`dspy` or `simple_meta_prompt`).

2. **Signature Creation**
   - `create_signature`: Dynamically creates a DSPy signature class based on input/output fields for the task.

3. **Synthetic Data Generation**
   - `generate_synthetic_data`: Generates synthetic training data in batches using LLMs, with validation and feedback loops.
   - `_prepare_sample_data`: Parses and prepares sample data for generation.
   - `_create_synthetic_data_prompt`: Crafts the prompt for LLM-based data generation.
   - `_validate_synthetic_data`: Validates each generated sample using LLM feedback.

4. **Backend Selection and Optimization Run**
   - `run`: Entry point for running the optimization. Dispatches to the selected backend.
   - `_run_dspy_backend`: Full DSPy-based optimization pipeline (signature, data, trainer, program, evaluation, compilation, results).
   - `_run_meta_prompt_backend`: Meta-prompt optimization using direct LLM API calls and meta-prompting strategies.

5. **Evaluation**
   - `_evaluate_prompt_meta_backend`: Evaluates a prompt by running it against synthetic data and scoring with metrics.
   - `get_eval_metrics` / `get_final_eval_metrics`: Retrieves the appropriate evaluation metrics for the task.

6. **LLM API Calls**
   - `_call_llm_api_directly`, `_call_openai_api`, `_call_anthropic_api`: Handles direct calls to LLM providers for prompt optimization and validation.

7. **Dataset Preparation**
   - `_prepare_dataset`, `_prepare_datasets`, `_prepare_full_validation_dataset`: Converts raw data into DSPy Example objects for training/validation.

8. **Trainer and Compilation**
   - `_initialize_trainer`: Sets up the DSPy trainer (e.g., MIPROv2).
   - `_compile_program`: Compiles the DSPy program using the trainer and datasets.

9. **Result Preparation**
   - `_prepare_results`: Assembles the final results, including optimized prompt, scores, synthetic data, and LLM cost.

10. **Logging**
    - `setup_optimizer_logger`: Sets up a dedicated logger for all optimization steps and results.

---

## Class: PromptOptimizer

```
class PromptOptimizer
```
Handles the optimization of prompts using either DSPy or meta-prompt backend.

**Attributes**
- `config` (Config): Configuration for optimization
- `lm`: Language model instance
- `llm_cost` (float): Accumulated LLM usage cost
- `backend` (str): Optimization backend ('dspy' or 'simple_meta_prompt')
- `logger`: Logger instance

**Methods**
- `__init__(config: Config)`: Initialize the optimizer with configuration.
- `create_signature(name: str, input_fields: List[str], output_fields: List[str]) -> Type[dspy.Signature]`: Create a DSPy signature for the optimization task.
- `generate_synthetic_data() -> List[Dict]`: Generate synthetic training data based on sample data in batches, with validation.
- `run(initial_flag: bool = True) -> Dict`: Run the optimization process using the configured backend.
- `get_eval_metrics()`: Get evaluation metrics for the task type.
- `get_final_eval_metrics()`: Get final evaluation metrics for the task type.

---

## Internal and Utility Methods

- **_parse_fields**
  ```
  def _parse_fields(fields: Union[List[str], str]) -> List[str]
  ```
  Parses field definitions from a string or list, ensuring a standardized list of field names for signature and data preparation.

- **_prepare_sample_data**
  ```
  def _prepare_sample_data(self) -> Dict
  ```
  Prepares and parses sample data for synthetic data generation, extracting input-output pairs and formatting them for prompt construction.

- **_create_synthetic_data_prompt**
  ```
  def _create_synthetic_data_prompt(self, sample_data: Dict, template: Dict, batch_size: int, feedback_section: str = "") -> str
  ```
  Generates a high-quality prompt for LLM-based synthetic data creation, incorporating sample data, templates, and optional feedback.

- **_clean_llm_response**
  ```
  def _clean_llm_response(self, response: str) -> str
  ```
  Cleans and formats the raw LLM response, removing extraneous text and ensuring consistency for downstream processing.

- **_run_dspy_backend**
  ```
  def _run_dspy_backend(self, initial_flag: bool = True) -> Dict
  ```
  Executes the full DSPy-based optimization pipeline, including signature creation, data preparation, training, evaluation, and result compilation.

- **_run_meta_prompt_backend**
  ```
  def _run_meta_prompt_backend(self, initial_flag: bool = True) -> Dict
  ```
  Runs the meta-prompt optimization backend using direct LLM API calls and meta-prompting strategies, returning optimized prompts and evaluation results.

- **_evaluate_prompt_meta_backend**
  ```
  def _evaluate_prompt_meta_backend(self, prompt: str) -> float
  ```
  Evaluates a prompt by running it against synthetic data and scoring it with the configured metrics, used primarily in the meta-prompt backend.

- **_create_test_input_from_sample**
  ```
  def _create_test_input_from_sample(self, sample: Dict) -> str
  ```
  Constructs a test input string from a sample data dictionary, used for prompt evaluation and LLM input formatting.

- **_create_prediction_object**
  ```
  def _create_prediction_object(self, prediction_text: str, sample: Dict) -> Dict
  ```
  Creates a prediction object with the expected structure for evaluation, combining the model's output with the original sample.

- **_call_llm_api_directly**
  ```
  def _call_llm_api_directly(self, prompt: str, model: str = "") -> str
  ```
  Calls the LLM API directly based on the configured provider, handling prompt submission and response retrieval.

- **_call_openai_api**
  ```
  def _call_openai_api(self, prompt: str, model: str = "") -> str
  ```
  Sends a prompt to the OpenAI API and returns the generated response.

- **_call_anthropic_api**
  ```
  def _call_anthropic_api(self, prompt: str) -> str
  ```
  Sends a prompt to the Anthropic API and returns the generated response.

- **_parse_input_fields**
  ```
  def _parse_input_fields(self) -> Union[str, List[str], Tuple[str, ...]]
  ```
  Parses input fields from the configuration, supporting flexible field definitions.

- **_prepare_dataset**
  ```
  def _prepare_dataset(self, data: List[Dict]) -> List[dspy.Example]
  ```
  Converts raw data into DSPy Example objects for training and validation.

- **_prepare_datasets**
  ```
  def _prepare_datasets(self)
  ```
  Prepares training and validation datasets, splitting and formatting data as needed for optimization.

- **_prepare_full_validation_dataset**
  ```
  def _prepare_full_validation_dataset(self)
  ```
  Prepares the full validation dataset if available, ensuring comprehensive evaluation.

- **_initialize_trainer**
  ```
  def _initialize_trainer(self)
  ```
  Sets up the DSPy trainer (e.g., MIPROv2) for optimization.

- **_compile_program**
  ```
  def _compile_program(self, trainer, program, trainset, validset)
  ```
  Compiles the DSPy program using the trainer and datasets, preparing it for evaluation and optimization.

- **_prepare_results**
  ```
  def _prepare_results(self, initial_prompt: str, optimized_prompt: str, initial_score: float, optimized_score: float) -> Dict
  ```
  Assembles the final results, including optimized prompt, scores, synthetic data, and LLM cost.

- **_validate_synthetic_data**
  ```
  def _validate_synthetic_data(self, data: Dict, task: str) -> Tuple[bool, str]
  ```
  Validates generated synthetic data for quality and consistency, returning a boolean and feedback message.

---

## Function: setup_optimizer_logger

```
def setup_optimizer_logger()
```
Set up dedicated logger for optimization steps and results. Creates a log file in the optimizer logs directory, storing all optimization steps and results in JSON Lines format.

--- 