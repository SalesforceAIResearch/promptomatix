# How Do I Customize and Extend Promptomatix?

This guide covers how to customize and extend Promptomatix for your specific use cases, from simple configuration changes to advanced customizations.

## Table of Contents
- [Backend Selection](#backend-selection)
- [Custom Meta-Prompts](#custom-meta-prompts)
- [Custom Metrics](#custom-metrics)
- [Custom DSPy Modules](#custom-dspy-modules)
- [Adding New Task Types](#adding-new-task-types)
- [Custom Data Processing](#custom-data-processing)
- [Extending the CLI](#extending-the-cli)
- [Advanced Customizations](#advanced-customizations)

## Backend Selection

### What are Backends?

Promptomatix supports two optimization backends:

1. **Meta-Prompt Backend (`simple_meta_prompt`)**: Uses a meta-prompt to directly optimize your input prompt through iterative refinement. This is lightweight, fast, and requires minimal configuration.

2. **DSPy Backend (`dspy`)**: Uses the DSPy library for programmatic prompt compilation and optimization. This provides more control and advanced features like chain-of-thought reasoning, but requires more setup.

### How to Change Backends

#### Command Line Interface
```bash
# Use meta-prompt backend (default)
promtomatic --raw_input "Classify text sentiment" --backend "simple_meta_prompt"

# Use DSPy backend
promtomatic --raw_input "Classify text sentiment" --backend "dspy"
```

#### Python API
```python
from promptomatix.main import process_input

# Meta-prompt backend
result = process_input(
    human_input="Classify text sentiment",
    backend="simple_meta_prompt"
)

# DSPy backend
result = process_input(
    human_input="Classify text sentiment", 
    backend="dspy"
)
```

### Can I Add My Own Backend?

While you can technically add custom backends by modifying `config.py` and `optimizer.py`, we recommend using the existing backends with customizations:

- **For meta-prompt backend**: Customize the meta-prompt itself (see [Custom Meta-Prompts](#custom-meta-prompts))
- **For DSPy backend**: Configure DSPy components and modules (see [Custom DSPy Modules](#custom-dspy-modules))

## Custom Meta-Prompts

### Understanding Meta-Prompts

The meta-prompt backend uses a meta-prompt to optimize your input. The current meta-prompts are defined in `src/promptomatix/core/prompts.py`:

- `generate_meta_prompt()`: Basic meta-prompt for general optimization
- `generate_meta_prompt_2()`: Enhanced version with better structure
- `generate_meta_prompt_7()`: Latest version with improved instructions

### Creating Your Own Meta-Prompt

1. **Add your meta-prompt function** in `src/promptomatix/core/prompts.py`:

```python
def generate_my_custom_meta_prompt(initial_prompt: str) -> str:
    return f"""You are an expert prompt engineer. Your task is to optimize the following prompt for better performance.

ORIGINAL PROMPT:
{initial_prompt}

OPTIMIZATION GUIDELINES:
- Make the prompt more specific and actionable
- Add clear examples where helpful
- Ensure the output format is well-defined
- Remove ambiguity and vagueness
- Keep the prompt concise but comprehensive

OPTIMIZED PROMPT:
"""
```

2. **Update the optimizer** in `src/promptomatix/core/optimizer.py`:

```python
# In the _run_meta_prompt_backend method, replace:
meta_prompt = generate_meta_prompt_7(initial_prompt)

# With:
meta_prompt = generate_my_custom_meta_prompt(initial_prompt)
```

### Meta-Prompt Best Practices

- **Be specific about optimization goals**: Define what "better" means for your use case
- **Include examples**: Show the expected input/output format
- **Set constraints**: Specify length limits, tone requirements, etc.
- **Iterative refinement**: Design prompts that can be improved through multiple iterations

## Custom Metrics

### Understanding the Metrics System

Promptomatix uses a `MetricsManager` class that automatically selects appropriate metrics based on task type. Metrics are defined in `src/promptomatix/metrics/metrics.py`.

### Adding Custom Metrics

1. **Create a custom metric function**:

```python
def my_custom_metric(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
    """
    Custom metric for your specific task.
    
    Args:
        example: The ground truth example
        pred: The model prediction
        instructions: The prompt used
        trace: Execution trace (if available)
    
    Returns:
        float: Score between 0 and 1
    """
    try:
        # Extract values
        gold_output = MetricsManager._get_output_value(example)
        pred_output = MetricsManager._get_output_value(pred)
        
        # Your custom logic here
        # Example: Custom similarity score
        similarity_score = calculate_custom_similarity(pred_output, gold_output)
        
        # Apply length penalty (optional)
        if instructions:
            prompt_length = len(instructions.split())
            length_penalty = math.exp(-LambdaPenalty.get_value() * prompt_length)
            return similarity_score * length_penalty
        
        return similarity_score
        
    except Exception as e:
        print(f"Error in custom metric: {str(e)}")
        return 0.0
```

2. **Register your metric** in the `MetricsManager`:

```python
# In the get_metrics_for_task method, add:
metrics_map = {
    'qa': MetricsManager._qa_metrics,
    'classification': MetricsManager._classification_metrics,
    # ... existing metrics ...
    'my_custom_task': my_custom_metric,  # Add your metric
}
```

3. **Use your custom task type**:

```python
result = process_input(
    human_input="Your task description",
    task_type="my_custom_task"
)
```

### Custom Metrics Example

See `examples/scripts/custom_metrics.py` for a complete example of implementing custom metrics with quality scoring, cost efficiency, and performance tracking.

## Custom DSPy Modules

### Understanding DSPy Modules

DSPy modules define how the model should approach the task. Available modules include:

- `dspy.Predict`: Basic prediction
- `dspy.ChainOfThought`: Step-by-step reasoning
- `dspy.ProgramOfThought`: Code-based reasoning
- `dspy.ReAct`: Tool-using agent

### Using Different DSPy Modules

#### Command Line
```bash
promtomatic --raw_input "Solve math problems" --backend "dspy" --dspy_module "dspy.ChainOfThought"
```

#### Python API
```python
result = process_input(
    human_input="Solve math problems",
    backend="dspy",
    dspy_module="dspy.ChainOfThought"
)
```

### Creating Custom DSPy Modules

1. **Define your custom module**:

```python
import dspy

class CustomReasoningModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.ChainOfThought(signature)
    
    def forward(self, **kwargs):
        # Add custom preprocessing
        processed_kwargs = self.preprocess_inputs(kwargs)
        
        # Use the base predictor
        result = self.predictor(**processed_kwargs)
        
        # Add custom postprocessing
        return self.postprocess_output(result)
    
    def preprocess_inputs(self, inputs):
        # Your custom preprocessing logic
        return inputs
    
    def postprocess_output(self, output):
        # Your custom postprocessing logic
        return output
```

2. **Register your module** in `config.py`:

```python
# Add to DSPY_MODULE_MAP
DSPY_MODULE_MAP = {
    DSPyModules.PREDICT: dspy.Predict,
    DSPyModules.CHAIN_OF_THOUGHT: dspy.ChainOfThought,
    # ... existing modules ...
    'custom_reasoning': CustomReasoningModule,  # Add your module
}
```

## Adding New Task Types

### Understanding Task Types

Task types determine which metrics and processing logic are used. They're defined in the `MetricsManager` and `Config` classes.

### Adding a New Task Type

1. **Add metrics for your task type** in `metrics.py`:

```python
@staticmethod
def _my_new_task_metrics(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
    """Compute metrics for your new task type."""
    # Your metric implementation
    pass

@staticmethod
def _my_new_task_metrics_final_eval(example: Any, pred: Any, instructions: Any = None, trace: Any = None) -> float:
    """Final evaluation metrics for your new task type."""
    # Your final evaluation implementation
    pass
```

2. **Register the metrics** in both metric maps:

```python
# In get_metrics_for_task
metrics_map = {
    # ... existing metrics ...
    'my_new_task': MetricsManager._my_new_task_metrics,
}

# In get_final_eval_metrics
metrics_map = {
    # ... existing metrics ...
    'my_new_task': MetricsManager._my_new_task_metrics_final_eval,
}
```

3. **Add task-specific processing** in `config.py` if needed:

```python
# In the Config class, add task-specific logic
if self.task_type == 'my_new_task':
    # Custom processing for your task type
    pass
```

## Custom Data Processing

### Understanding Data Processing

Promptomatix processes data through several stages:
1. **Input extraction**: Parsing raw input to extract task description, sample data, etc.
2. **Synthetic data generation**: Creating training data from examples
3. **Dataset preparation**: Formatting data for optimization

### Customizing Data Processing

1. **Custom input extraction**:

```python
# In prompts.py, create custom extraction functions
def extract_my_custom_fields(human_input: str) -> str:
    """Extract custom fields from human input."""
    return f"""Extract the following information from this input:
    
    INPUT: {human_input}
    
    Please extract:
    - Custom field 1
    - Custom field 2
    - Custom field 3
    """
```

2. **Custom synthetic data generation**:

```python
def generate_my_custom_synthetic_data(task: str, batch_size: int, example_data: str, template: str) -> str:
    """Generate synthetic data with custom logic."""
    return f"""Generate {batch_size} synthetic examples for this task:
    
    TASK: {task}
    EXAMPLES: {example_data}
    TEMPLATE: {template}
    
    CUSTOM REQUIREMENTS:
    - Your specific requirements here
    - Additional constraints
    - Special formatting needs
    """
```

3. **Custom dataset processing**:

```python
# In config.py, add custom dataset processing
def _process_my_custom_dataset(self, train_dataset: Dataset, test_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Custom processing for your dataset type."""
    # Your custom processing logic
    return train_dataset, test_dataset
```

## Extending the CLI

### Adding New CLI Arguments

1. **Add arguments** in `src/promptomatix/cli/parser.py`:

```python
# Add to the appropriate argument group
custom_group = parser.add_argument_group('Custom Configuration')
custom_group.add_argument("--my_custom_param", type=str, 
                         help="Description of your custom parameter")
custom_group.add_argument("--my_custom_flag", action="store_true",
                         help="Description of your custom flag")
```

2. **Handle the arguments** in your processing logic:

```python
# In main.py or optimizer.py
if args.get('my_custom_param'):
    # Handle your custom parameter
    pass

if args.get('my_custom_flag'):
    # Handle your custom flag
    pass
```

### Adding New CLI Commands

1. **Create a new command function**:

```python
def my_custom_command(args):
    """Handle your custom command."""
    # Your command logic here
    pass
```

2. **Add command routing** in the main function:

```python
def main():
    args = parse_args()
    
    if args.get('my_custom_command'):
        return my_custom_command(args)
    elif args.get('raw_input'):
        # Existing logic
        pass
```

## Advanced Customizations

### Custom Model Providers

1. **Add new model provider** in `config.py`:

```python
class ModelProvider(Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    DATABRICKS = 'databricks'
    LOCAL = 'local'
    TOGETHERAI = 'togetherai'
    # ... existing providers ...
    MY_CUSTOM_PROVIDER = 'my_custom_provider'  # Add your provider
```

2. **Add provider configuration** in the `_setup_model_config` method:

```python
PROVIDER_CONFIGS = {
    # ... existing providers ...
    ModelProvider.MY_CUSTOM_PROVIDER: {
        'api_base': 'https://api.myprovider.com/v1',
        'env_key': 'MY_PROVIDER_API_KEY',
        'default_model': 'my-model-v1'
    }
}
```

3. **Implement API calls** in `optimizer.py`:

```python
def _call_my_custom_api(self, prompt: str, model: str = "") -> str:
    """Call your custom API."""
    # Your API implementation
    pass
```

4. **Update the LMManager** in `lm_manager.py`:

```python
SUPPORTED_PROVIDERS = {
    'openai': OpenAI,
    'anthropic': Anthropic,
    'cohere': Cohere,
    'my_custom_provider': MyCustomProvider,  # Add your provider class
}
```

### Custom Session Management

1. **Extend session functionality**:

```python
class CustomOptimizationSession(OptimizationSession):
    """Extended session with custom features."""
    
    def __init__(self, session_id: str, initial_human_input: str, config: Config):
        super().__init__(session_id, initial_human_input, config)
        self.custom_metadata = {}
        self.version_history = []
    
    def add_version(self, prompt: str, metadata: Dict = None):
        """Track prompt versions."""
        self.version_history.append({
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
    
    def get_version_history(self) -> List[Dict]:
        """Get all prompt versions."""
        return self.version_history
```

2. **Custom session manager**:

```python
class CustomSessionManager(SessionManager):
    """Extended session manager with custom features."""
    
    def __init__(self):
        super().__init__()
        self.session_analytics = {}
    
    def create_session(self, session_id: str, initial_input: str, config: Config) -> CustomOptimizationSession:
        """Create custom session."""
        session = CustomOptimizationSession(session_id, initial_input, config)
        self.sessions[session_id] = session
        self._save_session(session)
        return session
    
    def get_session_analytics(self, session_id: str) -> Dict:
        """Get analytics for a session."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'duration': (datetime.now() - session.created_at).total_seconds(),
            'feedback_count': len(session.latest_human_feedback),
            'version_count': len(getattr(session, 'version_history', [])),
            'optimization_count': len(getattr(session, 'optimization_history', []))
        }
```

### Custom Feedback Systems

1. **Extend feedback functionality**:

```python
class CustomFeedback(Feedback):
    """Extended feedback with additional features."""
    
    def __init__(self, text: str, start_offset: int, end_offset: int, 
                 feedback: str, prompt_id: Optional[str] = None,
                 feedback_type: str = "general", priority: str = "medium"):
        super().__init__(text, start_offset, end_offset, feedback, prompt_id)
        self.feedback_type = feedback_type  # "general", "format", "content", "style"
        self.priority = priority  # "low", "medium", "high", "critical"
        self.resolved = False
        self.resolution_notes = ""
    
    def mark_resolved(self, notes: str = ""):
        """Mark feedback as resolved."""
        self.resolved = True
        self.resolution_notes = notes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with custom fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'feedback_type': self.feedback_type,
            'priority': self.priority,
            'resolved': self.resolved,
            'resolution_notes': self.resolution_notes
        })
        return base_dict
```

2. **Custom feedback store**:

```python
class CustomFeedbackStore(FeedbackStore):
    """Extended feedback store with advanced features."""
    
    def __init__(self):
        super().__init__()
        self.feedback_categories = {}
    
    def add_feedback(self, feedback: CustomFeedback) -> Dict:
        """Add feedback with categorization."""
        result = super().add_feedback(feedback)
        
        # Categorize feedback
        if feedback.feedback_type not in self.feedback_categories:
            self.feedback_categories[feedback.feedback_type] = []
        self.feedback_categories[feedback.feedback_type].append(feedback.id)
        
        return result
    
    def get_feedback_by_type(self, feedback_type: str) -> List[Dict]:
        """Get feedback by type."""
        return [f.to_dict() for f in self.feedback 
                if hasattr(f, 'feedback_type') and f.feedback_type == feedback_type]
    
    def get_unresolved_feedback(self, prompt_id: Optional[str] = None) -> List[Dict]:
        """Get unresolved feedback."""
        return [f.to_dict() for f in self.feedback 
                if hasattr(f, 'resolved') and not f.resolved
                and (not prompt_id or f.prompt_id == prompt_id)]
    
    def get_feedback_summary(self, prompt_id: Optional[str] = None) -> Dict:
        """Get comprehensive feedback summary."""
        relevant_feedback = [f for f in self.feedback 
                           if not prompt_id or f.prompt_id == prompt_id]
        
        return {
            'total_feedback': len(relevant_feedback),
            'resolved_count': len([f for f in relevant_feedback 
                                 if hasattr(f, 'resolved') and f.resolved]),
            'unresolved_count': len([f for f in relevant_feedback 
                                   if hasattr(f, 'resolved') and not f.resolved]),
            'by_type': {
                feedback_type: len([f for f in relevant_feedback 
                                  if hasattr(f, 'feedback_type') and f.feedback_type == feedback_type])
                for feedback_type in ['general', 'format', 'content', 'style']
            },
            'by_priority': {
                priority: len([f for f in relevant_feedback 
                             if hasattr(f, 'priority') and f.priority == priority])
                for priority in ['low', 'medium', 'high', 'critical']
            }
        }
```

### Custom Logging and Monitoring

1. **Extend the logging system**:

```python
class CustomSessionLogger(SessionLogger):
    """Extended session logger with custom features."""
    
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.performance_metrics = {}
        self.error_tracking = []
    
    def add_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Track performance metrics."""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append({
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_error(self, error_type: str, error_message: str, context: Dict = None):
        """Track errors with context."""
        self.error_tracking.append({
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'average': sum(v['value'] for v in values) / len(values),
                    'min': min(v['value'] for v in values),
                    'max': max(v['value'] for v in values),
                    'unit': values[0].get('unit', '')
                }
        return summary
    
    def get_error_summary(self) -> Dict:
        """Get error summary."""
        error_counts = {}
        for error in self.error_tracking:
            error_type = error['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_tracking),
            'error_types': error_counts,
            'recent_errors': self.error_tracking[-10:]  # Last 10 errors
        }
```

2. **Add custom metrics tracking**:

```python
class CustomMetricsTracker:
    """Track custom metrics for your use case."""
    
    def __init__(self):
        self.metrics = []
        self.alert_thresholds = {}
    
    def add_metric(self, name: str, value: float, metadata: Dict = None):
        """Add a custom metric."""
        metric_entry = {
            'name': name,
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.metrics.append(metric_entry)
        
        # Check alerts
        if name in self.alert_thresholds:
            threshold = self.alert_thresholds[name]
            if value > threshold['max'] or value < threshold['min']:
                self._trigger_alert(name, value, threshold)
    
    def set_alert_threshold(self, metric_name: str, min_value: float = None, max_value: float = None):
        """Set alert thresholds for a metric."""
        self.alert_thresholds[metric_name] = {
            'min': min_value,
            'max': max_value
        }
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: Dict):
        """Trigger an alert when threshold is exceeded."""
        alert_message = f"Alert: {metric_name} = {value}"
        if threshold['max'] and value > threshold['max']:
            alert_message += f" (exceeds max: {threshold['max']})"
        if threshold['min'] and value < threshold['min']:
            alert_message += f" (below min: {threshold['min']})"
        
        # Log alert or send notification
        print(f"🚨 {alert_message}")
    
    def get_performance_summary(self) -> Dict:
        """Get summary of custom metrics."""
        if not self.metrics:
            return {}
        
        # Group by metric name
        metric_groups = {}
        for metric in self.metrics:
            name = metric['name']
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric['value'])
        
        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                'count': len(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'recent': values[-5:] if len(values) >= 5 else values  # Last 5 values
            }
        
        return summary
```

### Custom Data Processing Pipelines

1. **Custom input preprocessing**:

```python
class CustomDataPreprocessor:
    """Custom data preprocessing pipeline."""
    
    def __init__(self):
        self.preprocessing_steps = []
    
    def add_preprocessing_step(self, step_name: str, step_function: Callable):
        """Add a preprocessing step."""
        self.preprocessing_steps.append({
            'name': step_name,
            'function': step_function
        })
    
    def preprocess_input(self, raw_input: str) -> str:
        """Apply all preprocessing steps."""
        processed_input = raw_input
        
        for step in self.preprocessing_steps:
            try:
                processed_input = step['function'](processed_input)
            except Exception as e:
                print(f"Error in preprocessing step '{step['name']}': {e}")
                continue
        
        return processed_input
    
    def preprocess_synthetic_data(self, data: List[Dict]) -> List[Dict]:
        """Preprocess synthetic data."""
        processed_data = []
        
        for item in data:
            processed_item = item.copy()
            
            # Apply preprocessing to input fields
            for field in processed_item:
                if isinstance(processed_item[field], str):
                    processed_item[field] = self.preprocess_input(processed_item[field])
            
            processed_data.append(processed_item)
        
        return processed_data
```

2. **Custom data validation**:

```python
class CustomDataValidator:
    """Custom data validation system."""
    
    def __init__(self):
        self.validation_rules = {}
    
    def add_validation_rule(self, field_name: str, rule_function: Callable, error_message: str):
        """Add a validation rule for a field."""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        self.validation_rules[field_name].append({
            'function': rule_function,
            'error_message': error_message
        })
    
    def validate_data(self, data: List[Dict]) -> Dict:
        """Validate data against all rules."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        for i, item in enumerate(data):
            for field_name, rules in self.validation_rules.items():
                if field_name in item:
                    field_value = item[field_name]
                    
                    for rule in rules:
                        try:
                            if not rule['function'](field_value):
                                validation_results['errors'].append({
                                    'item_index': i,
                                    'field': field_name,
                                    'value': field_value,
                                    'error': rule['error_message']
                                })
                                validation_results['valid'] = False
                        except Exception as e:
                            validation_results['warnings'].append({
                                'item_index': i,
                                'field': field_name,
                                'error': f"Validation rule failed: {e}"
                            })
        
        return validation_results
```

### Custom Evaluation Strategies

1. **Create custom evaluation logic**:

```python
class CustomEvaluator:
    """Custom evaluation strategy for prompts."""
    
    def __init__(self):
        self.evaluation_metrics = {}
        self.custom_scoring = {}
    
    def add_evaluation_metric(self, metric_name: str, metric_function: Callable):
        """Add a custom evaluation metric."""
        self.evaluation_metrics[metric_name] = metric_function
    
    def add_custom_scoring(self, scoring_name: str, scoring_function: Callable, weight: float = 1.0):
        """Add custom scoring with weight."""
        self.custom_scoring[scoring_name] = {
            'function': scoring_function,
            'weight': weight
        }
    
    def evaluate_prompt(self, prompt: str, test_data: List[Dict], 
                       model_responses: List[str]) -> Dict:
        """Evaluate a prompt using custom metrics."""
        results = {
            'prompt': prompt,
            'metrics': {},
            'custom_scores': {},
            'overall_score': 0.0
        }
        
        # Calculate standard metrics
        for metric_name, metric_function in self.evaluation_metrics.items():
            try:
                results['metrics'][metric_name] = metric_function(
                    prompt, test_data, model_responses
                )
            except Exception as e:
                results['metrics'][metric_name] = 0.0
                print(f"Error calculating metric '{metric_name}': {e}")
        
        # Calculate custom scores
        total_weight = 0.0
        weighted_sum = 0.0
        
        for scoring_name, scoring_config in self.custom_scoring.items():
            try:
                score = scoring_config['function'](prompt, test_data, model_responses)
                results['custom_scores'][scoring_name] = score
                weighted_sum += score * scoring_config['weight']
                total_weight += scoring_config['weight']
            except Exception as e:
                results['custom_scores'][scoring_name] = 0.0
                print(f"Error calculating custom score '{scoring_name}': {e}")
        
        # Calculate overall score
        if total_weight > 0:
            results['overall_score'] = weighted_sum / total_weight
        
        return results
```

2. **Integrate with the optimizer**:

```python
# In the optimizer, replace or extend evaluation calls
class CustomPromptOptimizer(PromptOptimizer):
    """Extended optimizer with custom evaluation."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.custom_evaluator = CustomEvaluator()
        self.setup_custom_evaluation()
    
    def setup_custom_evaluation(self):
        """Setup custom evaluation metrics."""
        # Add custom metrics
        self.custom_evaluator.add_evaluation_metric(
            'prompt_clarity',
            self._calculate_prompt_clarity
        )
        
        self.custom_evaluator.add_custom_scoring(
            'domain_specific_score',
            self._calculate_domain_score,
            weight=0.3
        )
    
    def _calculate_prompt_clarity(self, prompt: str, test_data: List[Dict], 
                                 model_responses: List[str]) -> float:
        """Calculate prompt clarity score."""
        # Your custom clarity calculation
        clarity_indicators = ['specific', 'clear', 'detailed', 'precise']
        score = sum(1 for indicator in clarity_indicators if indicator in prompt.lower())
        return min(score / len(clarity_indicators), 1.0)
    
    def _calculate_domain_score(self, prompt: str, test_data: List[Dict], 
                               model_responses: List[str]) -> float:
        """Calculate domain-specific score."""
        # Your domain-specific scoring logic
        return 0.8  # Example score
    
    def _evaluate_prompt_meta_backend(self, prompt: str) -> float:
        """Override evaluation with custom logic."""
        if hasattr(self, 'use_custom_evaluation') and self.use_custom_evaluation:
            # Use custom evaluation
            test_data = self.config.train_data + (self.config.valid_data or [])
            model_responses = self._get_model_responses(prompt, test_data)
            
            evaluation_result = self.custom_evaluator.evaluate_prompt(
                prompt, test_data, model_responses
            )
            
            return evaluation_result['overall_score']
        else:
            # Use default evaluation
            return super()._evaluate_prompt_meta_backend(prompt)
    
    def _get_model_responses(self, prompt: str, test_data: List[Dict]) -> List[str]:
        """Get model responses for test data."""
        responses = []
        for sample in test_data:
            try:
                test_input = self._create_test_input_from_sample(sample)
                full_prompt = f"{prompt}\n\n{test_input}"
                response = self._call_llm_api_directly(full_prompt)
                responses.append(response)
            except Exception as e:
                responses.append("")
        return responses
```

### Custom Evaluation Strategies

1. **Create custom evaluation logic**:

```python
def my_custom_evaluation_strategy(self, prompt: str, data: List[Dict]) -> float:
    """Custom evaluation strategy."""
    # Your evaluation logic
    pass
```

2. **Integrate with the optimizer**:

```python
# In the optimizer, replace or extend evaluation calls
if self.use_custom_evaluation:
    score = self.my_custom_evaluation_strategy(prompt, data)
else:
    score = self._evaluate_prompt_meta_backend(prompt)
```

### Custom Logging and Monitoring

1. **Extend the logging system**:

```python
# In logger.py, add custom loggers
def setup_custom_logger():
    """Setup custom logging for your extensions."""
    # Your custom logging setup
    pass
```

2. **Add custom metrics tracking**:

```python
class CustomMetricsTracker:
    """Track custom metrics for your use case."""
    
    def __init__(self):
        self.metrics = []
    
    def add_metric(self, name: str, value: float, metadata: Dict = None):
        """Add a custom metric."""
        self.metrics.append({
            'name': name,
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        })
    
    def get_summary(self) -> Dict:
        """Get summary of custom metrics."""
        # Your summary logic
        pass
```

## Best Practices for Customization

### 1. Start Simple
- Begin with existing backends and customize prompts/meta-prompts
- Use the CLI arguments for basic configuration
- Test thoroughly before adding complex customizations

### 2. Maintain Compatibility
- Keep your customizations compatible with the existing API
- Follow the established patterns in the codebase
- Document your changes clearly

### 3. Test Your Customizations
- Create test cases for your custom functionality
- Use the existing examples as templates
- Validate that your changes don't break existing features

### 4. Performance Considerations
- Monitor the impact of your customizations on performance
- Use efficient algorithms for custom metrics
- Consider caching for expensive operations

### 5. Documentation
- Document your customizations clearly
- Provide examples of how to use your extensions
- Update this guide if you add significant new features

## Troubleshooting Customizations

### Common Issues

1. **Backend not found**: Ensure your backend name is correctly registered in the config
2. **Metrics not working**: Check that your metric function returns a float between 0 and 1
3. **Custom prompts not used**: Verify that you've updated the correct function calls in the optimizer
4. **CLI arguments not recognized**: Make sure you've added arguments to the parser and handled them in the main logic

### Debugging Tips

1. **Enable verbose logging**: Set log levels to DEBUG to see detailed execution flow
2. **Use the session system**: Check session logs for detailed information about optimization steps
3. **Test incrementally**: Add customizations one at a time and test each step
4. **Use the examples**: Reference the existing examples for patterns and best practices

## Getting Help

If you encounter issues with customization:

1. **Check the examples**: Look at `examples/scripts/` for working implementations
2. **Review the API documentation**: See `docs/api/` for detailed function documentation
3. **Examine the source code**: The core modules are well-documented and can serve as examples
4. **Create minimal test cases**: Isolate the issue in a simple example

## Additional Customization Points

### Custom Path and Directory Management

1. **Customize application paths** in `utils/paths.py`:

```python
# Add custom directories
CUSTOM_DATA_DIR = PROJECT_ROOT / 'custom_data'
CUSTOM_MODELS_DIR = PROJECT_ROOT / 'custom_models'
CUSTOM_EXPORTS_DIR = PROJECT_ROOT / 'exports'

# Create custom directories
CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
```

2. **Custom path configuration**:

```python
class CustomPathManager:
    """Manage custom paths for your extensions."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.custom_dirs = {}
    
    def add_custom_directory(self, name: str, relative_path: str):
        """Add a custom directory."""
        full_path = self.base_dir / relative_path
        full_path.mkdir(parents=True, exist_ok=True)
        self.custom_dirs[name] = full_path
    
    def get_custom_path(self, name: str) -> Path:
        """Get a custom directory path."""
        return self.custom_dirs.get(name, self.base_dir)
    
    def list_custom_directories(self) -> Dict[str, Path]:
        """List all custom directories."""
        return self.custom_dirs.copy()
```

### Custom Parsing and Text Processing

1. **Extend parsing utilities** in `utils/parsing.py`:

```python
class CustomTextParser:
    """Custom text parsing utilities."""
    
    def __init__(self):
        self.parsing_rules = []
    
    def add_parsing_rule(self, rule_name: str, rule_function: Callable):
        """Add a custom parsing rule."""
        self.parsing_rules.append({
            'name': rule_name,
            'function': rule_function
        })
    
    def parse_text(self, text: str) -> Dict:
        """Apply all parsing rules to text."""
        result = {'original_text': text}
        
        for rule in self.parsing_rules:
            try:
                rule_result = rule['function'](text)
                result[rule['name']] = rule_result
            except Exception as e:
                result[rule['name']] = {'error': str(e)}
        
        return result
    
    def extract_structured_data(self, text: str) -> Dict:
        """Extract structured data from text."""
        # Custom structured data extraction
        return {
            'entities': self._extract_entities(text),
            'sentiment': self._analyze_sentiment(text),
            'keywords': self._extract_keywords(text)
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Your entity extraction logic
        return []
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        # Your sentiment analysis logic
        return "neutral"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Your keyword extraction logic
        return []
```

### Custom Configuration Management

1. **Extend configuration system**:

```python
class CustomConfig(Config):
    """Extended configuration with custom features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_settings = {}
        self.config_validation_rules = []
    
    def add_custom_setting(self, key: str, value: Any, description: str = ""):
        """Add a custom configuration setting."""
        self.custom_settings[key] = {
            'value': value,
            'description': description,
            'added_at': datetime.now().isoformat()
        }
    
    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Get a custom setting value."""
        return self.custom_settings.get(key, {}).get('value', default)
    
    def add_validation_rule(self, rule_function: Callable, error_message: str):
        """Add a custom validation rule."""
        self.config_validation_rules.append({
            'function': rule_function,
            'error_message': error_message
        })
    
    def validate_custom_settings(self) -> List[str]:
        """Validate custom settings."""
        errors = []
        
        for rule in self.config_validation_rules:
            try:
                if not rule['function'](self):
                    errors.append(rule['error_message'])
            except Exception as e:
                errors.append(f"Validation rule failed: {e}")
        
        return errors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary including custom settings."""
        base_dict = super().to_dict()
        base_dict['custom_settings'] = self.custom_settings
        return base_dict
```

### Custom CLI Extensions

1. **Add custom CLI commands**:

```python
def add_custom_cli_commands(parser: ArgumentParser):
    """Add custom CLI commands to the parser."""
    
    # Custom command group
    custom_group = parser.add_argument_group('Custom Commands')
    
    # Custom optimization command
    custom_group.add_argument("--custom_optimize", action="store_true",
                             help="Run custom optimization workflow")
    custom_group.add_argument("--optimization_strategy", type=str,
                             choices=['aggressive', 'conservative', 'balanced'],
                             help="Custom optimization strategy")
    
    # Custom analysis command
    custom_group.add_argument("--analyze_sessions", action="store_true",
                             help="Analyze all sessions for patterns")
    custom_group.add_argument("--export_analysis", type=str,
                             help="Export analysis results to file")
    
    # Custom data management
    custom_group.add_argument("--import_custom_data", type=str,
                             help="Import custom training data")
    custom_group.add_argument("--export_custom_data", type=str,
                             help="Export custom data to file")
    
    # Custom model management
    custom_group.add_argument("--custom_model_path", type=str,
                             help="Path to custom model configuration")
    custom_group.add_argument("--validate_custom_model", action="store_true",
                             help="Validate custom model configuration")

def handle_custom_commands(args: Dict) -> bool:
    """Handle custom CLI commands."""
    
    if args.get('custom_optimize'):
        return run_custom_optimization(args)
    
    if args.get('analyze_sessions'):
        return analyze_all_sessions(args)
    
    if args.get('import_custom_data'):
        return import_custom_data(args['import_custom_data'])
    
    if args.get('export_custom_data'):
        return export_custom_data(args['export_custom_data'])
    
    if args.get('validate_custom_model'):
        return validate_custom_model(args.get('custom_model_path'))
    
    return False

def run_custom_optimization(args: Dict) -> bool:
    """Run custom optimization workflow."""
    strategy = args.get('optimization_strategy', 'balanced')
    
    # Your custom optimization logic
    print(f"Running custom optimization with strategy: {strategy}")
    
    # Example custom workflow
    config = {
        'raw_input': args.get('raw_input'),
        'optimization_strategy': strategy,
        'custom_settings': {
            'max_iterations': 10 if strategy == 'aggressive' else 5,
            'quality_threshold': 0.9 if strategy == 'conservative' else 0.7
        }
    }
    
    # Run optimization with custom settings
    result = process_input(**config)
    print("Custom optimization completed!")
    return True
```

### Custom Integration Points

1. **Webhook integration**:

```python
class WebhookManager:
    """Manage webhook integrations for external systems."""
    
    def __init__(self):
        self.webhooks = {}
        self.webhook_history = []
    
    def register_webhook(self, event_type: str, url: str, headers: Dict = None):
        """Register a webhook for an event type."""
        if event_type not in self.webhooks:
            self.webhooks[event_type] = []
        
        self.webhooks[event_type].append({
            'url': url,
            'headers': headers or {},
            'enabled': True
        })
    
    def trigger_webhook(self, event_type: str, data: Dict):
        """Trigger webhooks for an event type."""
        if event_type not in self.webhooks:
            return
        
        for webhook in self.webhooks[event_type]:
            if webhook['enabled']:
                try:
                    self._send_webhook(webhook, data)
                except Exception as e:
                    print(f"Webhook failed: {e}")
    
    def _send_webhook(self, webhook: Dict, data: Dict):
        """Send webhook request."""
        import requests
        
        response = requests.post(
            webhook['url'],
            json=data,
            headers=webhook['headers'],
            timeout=10
        )
        
        # Log webhook history
        self.webhook_history.append({
            'url': webhook['url'],
            'event_type': data.get('event_type'),
            'status_code': response.status_code,
            'timestamp': datetime.now().isoformat()
        })
```

2. **Database integration**:

```python
class DatabaseManager:
    """Manage database integration for persistent storage."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self):
        """Establish database connection."""
        # Your database connection logic
        pass
    
    def save_session(self, session_data: Dict):
        """Save session to database."""
        # Your database save logic
        pass
    
    def load_session(self, session_id: str) -> Dict:
        """Load session from database."""
        # Your database load logic
        return {}
    
    def save_metrics(self, metrics_data: Dict):
        """Save metrics to database."""
        # Your metrics save logic
        pass
    
    def get_analytics(self, filters: Dict = None) -> Dict:
        """Get analytics from database."""
        # Your analytics query logic
        return {}
```

### Custom Plugin System

1. **Plugin architecture**:

```python
class PluginManager:
    """Manage custom plugins for Promptomatix."""
    
    def __init__(self):
        self.plugins = {}
        self.plugin_hooks = {}
    
    def register_plugin(self, plugin_name: str, plugin_class: Type):
        """Register a plugin."""
        self.plugins[plugin_name] = plugin_class()
    
    def add_hook(self, hook_name: str, plugin_name: str, hook_function: Callable):
        """Add a hook for a plugin."""
        if hook_name not in self.plugin_hooks:
            self.plugin_hooks[hook_name] = []
        
        self.plugin_hooks[hook_name].append({
            'plugin': plugin_name,
            'function': hook_function
        })
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all hooks for a given hook name."""
        if hook_name not in self.plugin_hooks:
            return
        
        results = []
        for hook in self.plugin_hooks[hook_name]:
            try:
                result = hook['function'](*args, **kwargs)
                results.append({
                    'plugin': hook['plugin'],
                    'result': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'plugin': hook['plugin'],
                    'error': str(e),
                    'success': False
                })
        
        return results

# Example plugin
class CustomOptimizationPlugin:
    """Example custom optimization plugin."""
    
    def __init__(self):
        self.name = "CustomOptimizationPlugin"
    
    def pre_optimization_hook(self, config: Config) -> Config:
        """Hook called before optimization starts."""
        # Modify config for custom optimization
        config.add_custom_setting('plugin_optimization', True)
        return config
    
    def post_optimization_hook(self, result: Dict) -> Dict:
        """Hook called after optimization completes."""
        # Add custom analysis to result
        result['plugin_analysis'] = self.analyze_result(result)
        return result
    
    def analyze_result(self, result: Dict) -> Dict:
        """Custom analysis of optimization result."""
        return {
            'custom_score': self.calculate_custom_score(result),
            'improvement_potential': self.assess_improvement_potential(result)
        }
    
    def calculate_custom_score(self, result: Dict) -> float:
        """Calculate custom score for result."""
        # Your custom scoring logic
        return 0.85
    
    def assess_improvement_potential(self, result: Dict) -> str:
        """Assess potential for further improvement."""
        # Your assessment logic
        return "high"
```

## Integration Examples

### Example 1: Custom Domain-Specific Optimizer

```python
class DomainSpecificOptimizer:
    """Custom optimizer for specific domains."""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.domain_rules = self._load_domain_rules(domain)
        self.custom_metrics = self._setup_domain_metrics(domain)
    
    def _load_domain_rules(self, domain: str) -> Dict:
        """Load domain-specific optimization rules."""
        rules = {
            'medical': {
                'precision_required': True,
                'safety_constraints': True,
                'regulatory_compliance': True
            },
            'legal': {
                'accuracy_required': True,
                'citation_needed': True,
                'formal_tone': True
            },
            'technical': {
                'code_quality': True,
                'documentation_required': True,
                'performance_focus': True
            }
        }
        return rules.get(domain, {})
    
    def _setup_domain_metrics(self, domain: str) -> Dict:
        """Setup domain-specific metrics."""
        metrics = {
            'medical': ['accuracy', 'safety_score', 'compliance_check'],
            'legal': ['accuracy', 'citation_quality', 'formal_tone_score'],
            'technical': ['code_quality', 'documentation_score', 'performance_metric']
        }
        return metrics.get(domain, [])
    
    def optimize_for_domain(self, prompt: str, task_type: str) -> str:
        """Optimize prompt for specific domain."""
        # Apply domain-specific optimization
        optimized_prompt = prompt
        
        if self.domain_rules.get('precision_required'):
            optimized_prompt = self._add_precision_requirements(optimized_prompt)
        
        if self.domain_rules.get('safety_constraints'):
            optimized_prompt = self._add_safety_constraints(optimized_prompt)
        
        if self.domain_rules.get('formal_tone'):
            optimized_prompt = self._ensure_formal_tone(optimized_prompt)
        
        return optimized_prompt
    
    def _add_precision_requirements(self, prompt: str) -> str:
        """Add precision requirements to prompt."""
        return prompt + "\n\nPlease provide precise, accurate, and well-supported responses."
    
    def _add_safety_constraints(self, prompt: str) -> str:
        """Add safety constraints to prompt."""
        return prompt + "\n\nEnsure all responses prioritize safety and follow established guidelines."
    
    def _ensure_formal_tone(self, prompt: str) -> str:
        """Ensure formal tone in prompt."""
        return prompt + "\n\nMaintain a formal, professional tone throughout your response."
```

### Example 2: Custom Workflow Integration

```python
class CustomWorkflowManager:
    """Manage custom optimization workflows."""
    
    def __init__(self):
        self.workflows = {}
        self.workflow_history = []
    
    def register_workflow(self, name: str, workflow_steps: List[Callable]):
        """Register a custom workflow."""
        self.workflows[name] = workflow_steps
    
    def execute_workflow(self, workflow_name: str, initial_data: Dict) -> Dict:
        """Execute a custom workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        current_data = initial_data.copy()
        workflow_steps = self.workflows[workflow_name]
        
        for i, step in enumerate(workflow_steps):
            try:
                step_result = step(current_data)
                current_data.update(step_result)
                
                # Log workflow progress
                self.workflow_history.append({
                    'workflow': workflow_name,
                    'step': i,
                    'step_name': step.__name__,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                # Log workflow error
                self.workflow_history.append({
                    'workflow': workflow_name,
                    'step': i,
                    'step_name': step.__name__,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                raise
        
        return current_data

# Example workflow steps
def step_1_analyze_input(data: Dict) -> Dict:
    """Step 1: Analyze input requirements."""
    # Your analysis logic
    return {'analysis_complete': True, 'requirements': ['clarity', 'specificity']}

def step_2_generate_variants(data: Dict) -> Dict:
    """Step 2: Generate prompt variants."""
    # Your variant generation logic
    return {'variants': ['variant1', 'variant2', 'variant3']}

def step_3_evaluate_variants(data: Dict) -> Dict:
    """Step 3: Evaluate all variants."""
    # Your evaluation logic
    return {'best_variant': 'variant2', 'scores': [0.8, 0.9, 0.7]}

# Register workflow
workflow_manager = CustomWorkflowManager()
workflow_manager.register_workflow(
    'comprehensive_optimization',
    [step_1_analyze_input, step_2_generate_variants, step_3_evaluate_variants]
)
```

Remember: The existing backends and configurations should handle most use cases. Only add customizations when absolutely necessary for your specific requirements.
