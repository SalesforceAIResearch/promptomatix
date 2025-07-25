# Promptomatix

## Overview
Promptomatix is an AI-driven framework for automating and optimizing large language model (LLM) prompts. It provides a structured, cost-effective, and high-quality approach to prompt engineering, leveraging DSPy and advanced optimization techniques. Ideal for both researchers and developers.

## Key Features
- Zero-configuration intelligence: automatic task analysis and prompt setup
- Automated synthetic data generation for training and testing
- Task-specific optimization and metrics
- Real-time human feedback integration
- Comprehensive session management and logging
- Framework-agnostic: supports OpenAI, Anthropic, Cohere, and more
- Flexible CLI and Python API interfaces

## Architecture
- **Input Processing**: Analyzes user input and task requirements
- **Synthetic Data Generation**: Creates tailored datasets
- **Optimization Engine**: Iteratively improves prompts
- **Evaluation System**: Assesses performance with task-specific metrics
- **Feedback Integration**: Incorporates human feedback
- **Session Management**: Tracks progress and logs

## Quick Install
```bash
# Clone the repository
git clone https://github.com/airesearch-emu/promptomatix.git
cd promptomatix
./install.sh
```

## Environment Setup
```bash
# Activate the environment
source promptomatix_env/bin/activate

# Set your API keys
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Example Usage
### CLI
```bash
python -m src.promptomatix.main --raw_input "Classify text sentiment into positive or negative"
```

### Python API
```python
from promptomatix import process_input
result = process_input(
    raw_input="Classify text sentiment",
    model_name="gpt-3.5-turbo",
    task_type="classification"
)
```

## Citation
If you use Promptomatix in your research, please cite:
```bibtex
@misc{murthy2025promptomatixautomaticpromptoptimization,
  title={Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models},
  author={Rithesh Murthy and Ming Zhu and Liangwei Yang and Jielin Qiu and Juntao Tan and Shelby Heinecke and Caiming Xiong and Silvio Savarese and Huan Wang},
  year={2025},
  eprint={2507.14241},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2507.14241},
}
``` 