<div align="center">
  <img src="images/logo1.png" alt="Promptomatix Logo" width="400"/>
  
  <h1>Promptomatix</h1>
  <h3>A Powerful Framework for LLM Prompt Optimization</h3>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/License-Apache-green.svg" alt="License">
</div>

<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-example-usage">Examples</a> |
  <a href="#-key-features">Features</a> |
  <a href="#-api-documentation">API</a> |
  <a href="#-cli-usage">CLI</a>
</p>

## 📋 Overview

Promptomatix is an AI-driven framework designed to automate and optimize large language model (LLM) prompts. It provides a structured approach to prompt optimization, ensuring consistency, cost-effectiveness, and high-quality outputs while reducing the trial-and-error typically associated with manual prompt engineering.

The framework leverages the power of DSPy and advanced optimization techniques to iteratively refine prompts based on task requirements, synthetic data, and user feedback. Whether you're a researcher exploring LLM capabilities or a developer building production applications, Promptomatix provides a comprehensive solution for prompt optimization.

## 🏗️ Architecture

<div align="center">
  <a href="images/architecture1.pdf" target="_blank">
    <img src="images/architecture1_quality.png" alt="Promptomatix Architecture" width="1200"/>
  </a>
</div>

The Promptomatix architecture consists of several key components:

- **Input Processing**: Analyzes raw user input to determine task type and requirements
- **Synthetic Data Generation**: Creates training and testing datasets tailored to the specific task
- **Optimization Engine**: Uses DSPy or meta-prompt backends to iteratively improve prompts
- **Evaluation System**: Assesses prompt performance using task-specific metrics
- **Feedback Integration**: Incorporates human feedback for continuous improvement
- **Session Management**: Tracks optimization progress and maintains detailed logs

### 🌟 Key Features

- **Zero-Configuration Intelligence**: Automatically analyzes tasks, selects techniques, and configures prompts
- **Automated Dataset Generation**: Creates synthetic training and testing data tailored to your specific domain
- **Task-Specific Optimization**: Selects the appropriate DSPy module and metrics based on task type
- **Real-Time Human Feedback**: Incorporates user feedback for iterative prompt refinement
- **Comprehensive Session Management**: Tracks optimization progress and maintains detailed logs
- **Framework Agnostic Design**: Supports multiple LLM providers (OpenAI, Anthropic, Cohere)
- **CLI and API Interfaces**: Flexible usage through command-line or REST API

## ⚙️ Installation

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/airesearch-emu/promptomatix.git
cd promptomatix

# Install with one command
./install.sh
```

The installer will:
- ✅ Check Python 3 installation
- ✅ Create a virtual environment (`promptomatix_env`)
- ✅ Initialize git submodules (DSPy)
- ✅ Install all dependencies

### 🔧 Activate the Environment

**Important**: You need to activate the virtual environment each time you use Promptomatix:

```bash
# Activate the environment
source promptomatix_env/bin/activate

# You'll see (promptomatix_env) in your prompt when activated
```

### 🔑 Set Up API Keys

```bash
# Set your API keys
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Or create a .env file
cp .env.example .env
# Edit .env with your actual API keys
```

### 🚀 Test Installation

```bash
# Test the installation
python -m src.promptomatix.main --raw_input "Given a questions about human anatomy answer it in two words" --model_name "gpt-3.5-turbo" --backend "simple_meta_prompt" --synthetic_data_size 10 --model_provider "openai"
```

### 💡 Pro Tips

**Auto-activation**: Add this to your `~/.bashrc` or `~/.zshrc`:
```bash
alias promptomatix='source promptomatix_env/bin/activate && promptomatix'
```

**Deactivate when done**:
```bash
deactivate
```

## 🚀 Example Usage

### Interactive Notebooks

The best way to learn Promptomatix is through our comprehensive Jupyter notebooks:

```bash
# Navigate to examples
cd examples/notebooks

# Start with basic usage
jupyter notebook 01_basic_usage.ipynb
```

**Notebook Guide:**
- **`01_basic_usage.ipynb`** - Simple prompt optimization workflow (start here!)
- **`02_prompt_optimization.ipynb`** - Advanced optimization techniques
- **`03_metrics_evaluation.ipynb`** - Evaluation and metrics analysis
- **`04_advanced_features.ipynb`** - Advanced features and customization

### Command Line Examples

```bash
# Basic optimization
python -m src.promptomatix.main --raw_input "Classify text sentiment into positive or negative"

# With custom model and parameters
python -m src.promptomatix.main --raw_input "Summarize this article" \
  --model_name "gpt-4" \
  --temperature 0.3 \
  --task_type "summarization"
# Advanced configuration
python -m src.promptomatix.main --raw_input "Given a questions about human anatomy answer it in two words" \
  --model_name "gpt-3.5-turbo" \
  --backend "simple_meta_prompt" \
  --synthetic_data_size 10 \
  --model_provider "openai"
```

### Python API Examples

```python
from promptomatix import process_input, generate_feedback, optimize_with_feedback

# Basic optimization
result = process_input(
    raw_input="Classify text sentiment",
    model_name="gpt-3.5-turbo",
    task_type="classification"
)

# Generate feedback for improvement
feedback = generate_feedback(
    optimized_prompt=result['result'],
    input_fields=result['input_fields'],
    output_fields=result['output_fields'],
    model_name="gpt-3.5-turbo"
)

# Optimize with feedback
improved_result = optimize_with_feedback(result['session_id'])
```
#### 📁 Project Structure

```
promptomatix/
├── images/                # Project images and logos
├── libs/                  # External libraries or submodules (e.g., DSPy)
├── logs/                  # Log files
├── promptomatix_env/      # Python virtual environment
├── sessions/              # Saved optimization sessions
├── dist/                  # Distribution files (if any)
├── build/                 # Build artifacts (if any)
├── examples/              # Example notebooks and scripts
├── src/
│   └── promptomatix/      # Core Python package
│       ├── cli/
│       ├── core/
│       ├── metrics/
│       ├── utils/
│       ├── __init__.py
│       ├── main.py
│       ├── lm_manager.py
│       └── logger.py
├── .env.example
├── .gitignore
├── .gitmodules
├── .python-version
├── CODEOWNERS
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE.txt
├── README.md
├── SECURITY.md
├── how_to_license.md
├── install.sh
├── requirements.txt
├── setup.py
```
---

<p align="center">
  <b>Promptomatix: Optimizing LLM prompts, so you don't have to.</b>
</p>
