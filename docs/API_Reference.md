# API Reference

This section documents the main classes and functions in Promptomatix.

## Core Classes
- `PromptOptimizer`
- `PromptSession`
- `PromptFeedback`

## Key Functions
- `optimize()`
- `evaluate()`
- `get_metrics()`

---
## config.py

### Class: ModelProvider

```
class ModelProvider(Enum)
```
Represents the available model providers for LLMs.

**Members**
- OPENAI: Use OpenAI models (e.g., GPT-4)
- ANTHROPIC: Use Anthropic models (e.g., Claude)
- DATABRICKS: Use Databricks models
- LOCAL: Use local models
- TOGETHERAI: Use TogetherAI models

---

### Class: Config

```
class Config
```
Configuration class for prompt optimization and evaluation workflows. This class manages all parameters for NLP task setup, model selection, data handling, and optimization. It supports both direct human input and HuggingFace datasets, and can automatically infer, validate, and populate missing configuration fields.

> **Note:** The config model (parameters prefixed with `config_`) acts as the "teacher" model. It is typically a more powerful or reliable model used for configuration, guidance, and generating synthetic data, while the main model (specified by `model_name` and `model_provider`) is the one being optimized or evaluated.

**Initialization Parameters**

- raw_input (str, required): The initial prompt to be optimized.
- raw_input_improvised (bool, optional): If the human input is unclear or sub-optimal, an LLM call is made to enhance it. The improved input is stored here.
- huggingface_dataset_name (str, optional): Name of a HuggingFace dataset (e.g., 'squad_v2'). For custom datasets, add them to DatasetConfig.
- original_raw_input (str, optional): Preserved copy of the initial human input.
- task_description (str, optional): Detailed description of the NLP task. Usually inferred internally; user input not required.
- sample_data (str, optional): Example input-output pairs for the task. Used as seed samples to create synthetic data. If not provided, generated internally using LLMs.
- output_format (str, optional): Expected output format (e.g., 'json', 'text', 'list').
- style_guide (str, optional): Guidelines for output formatting, tone, and style.
- constraints (str, optional): Task-specific constraints (e.g., max length).
- context (str, optional): Additional context or background information for the task.
- task_type (str, optional): Category of NLP task (e.g., 'classification', 'qa', 'generation').
- tools (list, optional): External tools or resources needed for the task.
- decouple_task_description_and_raw_input (bool, optional): If True, task description and human input are handled separately.
- model_name (str, optional): Name of the language model (e.g., 'gpt-4', 'claude-33.5').
- model_api_key (str, optional): API key for model access. Store securely in a .env file.
- model_api_base (str, optional): Base URL for API requests.
- model_provider (str, optional): Provider of the model (e.g., 'openai', 'anthropic').
- temperature (float, optional): Sampling temperature for generation. Default is 0.7.
- max_tokens (int, optional): Maximum tokens for model output. Default is 4000.
- config_max_tokens (int, optional): Max tokens for config LLM calls. Default is 14000.
- config_temperature (float, optional): Temperature for config LLM calls. Default is 0.7.
- config_model_name (str, optional): Model name for config LLM calls.
- config_model_provider (str, optional): Model provider for config LLM calls.
- config_model_api_key (str, optional): API key for config LLM calls.
- config_model_api_base (str, optional): API base for config LLM calls.
- synthetic_data_size (int, optional): Number of synthetic examples to generate. Default is 30. Used for training and testing.
- train_ratio (float, optional): Fraction of data to use for training. Default is 0.2.
- dspy_module (Any, optional): DSPy module configuration for task execution.
- input_fields (list, optional): Required input field names for structured data.
- output_fields (list, optional): Expected output field names for structured data.
- metrics (list, optional): Evaluation metrics to use (e.g., ['accuracy', 'f1']). If not evaluation metric is provided, then we infer it from the task description.
- trainer (str, optional): Training algorithm to use (e.g., 'MIPROv2').
- search_type (str, optional): Optimization strategy ('quick_search', 'moderate_search', 'heavy_search').
- backend (str, optional): Optimization backend ('dspy' or 'simple_meta_prompt'). Default is 'simple_meta_prompt'. Can be extended to other techniques as well.
- lambda_penalty (float, optional): Penalty value for metrics calculations. Default is 0.005.
- train_data (Any, optional): Training dataset.
- valid_data (Any, optional): Validation dataset.
- valid_data_full (Any, optional): Complete validation dataset.
- train_data_size (int, optional): Size of training dataset.
- valid_data_size (int, optional): Size of validation dataset.
- load_data_local (bool, optional): If True, load data from local files. Default is False.
- local_train_data_path (str, optional): Path to local training data file.
- local_test_data_path (str, optional): Path to local test data file.


**Configuration Flow**

The configuration process in the `Config` class is designed to be robust, automatic, and transparent. Here is a step-by-step overview of how configuration is populated and validated:

1. **Set Search Type Configuration**
   - The process begins by determining the search/optimization strategy (e.g., 'quick_search', 'moderate_search', 'heavy_search') using the internal `_set_search_type_config()` method. This sets up the overall approach for prompt optimization.

2. **Preserve Original Human Input**
   - If `original_raw_input` is not already set, it is initialized with the value of `raw_input`. This ensures the original user input is always available for reference, even if it is later modified or improved.

3. **Initialize the Language Model for Configuration**
   - The internal `_setup_model_config()` method is called to initialize the "teacher" (config) model. This model is typically more powerful and is used for all configuration, guidance, and synthetic data generation steps.

4. **Process Human Input and Feedback**
   - The human input is processed and improved (if necessary) using `_process_human_feedback()`. This may involve LLM calls to clarify, clean, or enhance the input, ensuring the system works with the best possible prompt.

5. **Develop Prompt Template Components**
   - The `_develop_prompt_template_components()` function extracts and constructs all necessary prompt components (such as templates, style guides, and constraints) from the processed input, ensuring the prompt is well-structured for optimization.

6. **Extract Task-Specific Configurations**
   - The following internal methods are called in order to extract all required task details:
     - `_extract_task_description()`: Determines the detailed task description, either from the dataset or the improved input.
     - `_extract_sample_data()`: Gathers or generates example input-output pairs (seed samples) for the task.
     - `_extract_task_type()`: Infers the type of NLP task (e.g., classification, QA, generation).
     - `_extract_fields()`: Identifies the required input and output fields for the task.
     - `_extract_tools()`: Extracts any external tools or resources needed for the task (if relevant).

7. **Set Training and Optimization Parameters**
   - The training algorithm is selected using `_set_trainer()`, and the DSPy module is chosen with `_set_dspy_module()`. These determine how the optimization and training will be performed.
   - Default values for `synthetic_data_size` (number of synthetic examples) and `train_ratio` (fraction of data for training) are set if not provided.

8. **Calculate Dataset Sizes**
   - The `_calculate_dataset_sizes()` method is called to determine the sizes of the training and validation datasets, ensuring proper data splits for training and evaluation.

9. **Track LLM Usage and Cost**
   - After all LLM calls and configuration steps, the total cost of LLM usage is calculated and stored in `llm_cost` for transparency and monitoring.

10. **Cleanup**
    - Temporary objects (such as the config model instance) are deleted to free up resources, and a final log message confirms successful configuration.

Throughout this process, all LLM interactions are logged, configuration values are validated as they are set, and default values are used when appropriate. The order of operations is carefully designed so that later steps can depend on the results of earlier ones (e.g., tools extraction depends on task type). If any required information is missing or inconsistent, a `ValueError` is raised to alert the user.

**Raises**

- ValueError: If configuration parameters are invalid, inconsistent, or missing required fields.

**Usage Example**

```python
config = Config(raw_input="Classify text into positive/negative sentiment")
config = Config(huggingface_dataset_name="squad_v2", model_name="gpt-4", model_provider="openai")
```

---