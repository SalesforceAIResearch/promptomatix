# Promptomatix Examples

This directory contains comprehensive examples demonstrating how to use the Promptomatix library for LLM prompt optimization.

## 📁 Directory Structure

```
examples/
├── notebooks/           # Jupyter notebooks for interactive examples
│   ├── 01_basic_usage.ipynb          # Simple prompt optimization workflow
│   ├── 02_prompt_optimization.ipynb  # Advanced optimization techniques
│   ├── 03_metrics_evaluation.ipynb   # Evaluation and metrics analysis
│   └── 04_advanced_features.ipynb    # Advanced features and customization
├── scripts/            # Python scripts for automation
│   ├── basic_example.py              # Basic usage script
│   ├── batch_processing.py           # Batch processing multiple prompts
│   └── custom_metrics.py             # Custom metrics implementation
├── data/              # Sample data for examples
│   ├── sample_prompts.txt            # Sample prompts for testing
│   └── test_data.json                # Test datasets
└── README.md          # This file
```

## 🚀 Quick Start

1. **For Beginners**: Start with `notebooks/01_basic_usage.ipynb`
2. **For Researchers**: Explore `notebooks/03_metrics_evaluation.ipynb`
3. **For Engineers**: Check out `scripts/batch_processing.py`

## 📚 Example Categories

### Basic Usage (`01_basic_usage.ipynb`)
- Simple prompt optimization
- Feedback generation
- Optimization with feedback
- Perfect for non-technical users

### Advanced Optimization (`02_prompt_optimization.ipynb`)
- Different optimization backends
- Custom task types
- Synthetic data generation
- For intermediate users

### Metrics & Evaluation (`03_metrics_evaluation.ipynb`)
- Performance metrics analysis
- Cost tracking
- Quality assessment
- For researchers and data scientists

### Advanced Features (`04_advanced_features.ipynb`)
- Custom configurations
- Batch processing
- Integration examples
- For advanced users and developers

## 🔧 Prerequisites

1. Install Promptomatix:
   ```bash
   ./install.sh
   source promptomatix_env/bin/activate
   ```

2. Set up API keys:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   ```

3. Install Jupyter (for notebooks):
   ```bash
   pip install jupyter
   ```

## 📖 Usage Examples

### Command Line
```bash
# Basic optimization
promtomatic --raw_input "Classify text sentiment"

# With custom model
promtomatic --raw_input "Summarize text" --model_name "gpt-4" --temperature 0.3
```

### Python API
```python
from promtomatic import process_input, generate_feedback, optimize_with_feedback

# Optimize a prompt
result = process_input(
    raw_input="Classify text sentiment",
    model_name="gpt-3.5-turbo",
    task_type="classification"
)

# Generate feedback
feedback = generate_feedback(
    optimized_prompt=result['result'],
    input_fields=result['input_fields'],
    output_fields=result['output_fields'],
    model_name="gpt-3.5-turbo",
    model_api_key="your_api_key"
)

# Optimize with feedback
improved_result = optimize_with_feedback(result['session_id'])
```

## 🎯 Target Audience

- **Researchers**: Focus on metrics, evaluation, and experimental design
- **Engineers**: Use scripts for automation and integration
- **Non-technical Users**: Start with basic usage notebook
- **Data Scientists**: Explore advanced features and custom metrics

## 📝 Contributing

When adding new examples:
1. Follow the existing naming convention
2. Include clear documentation
3. Test with different user skill levels
4. Add appropriate error handling
5. Include cost and performance considerations

## 🔗 Related Documentation

- [Main README](../README.md) - Library overview and installation
- [API Documentation](../docs/api.md) - Complete API reference
- [CLI Guide](../docs/cli.md) - Command-line interface guide 