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
- ✅ Set up the `promptomatix` command

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

### 🌐 Web Application

Promptomatix includes a modern web interface for interactive prompt optimization. The web app provides a user-friendly way to optimize prompts with real-time feedback and session management.

#### 🚀 Starting the Web App

**Prerequisites**: Make sure you have activated the virtual environment and set up your API keys.

```bash
# Activate the environment (if not already activated)
source promptomatix_env/bin/activate

# Set your API keys
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

**Start the Backend Server**:
```bash
# Navigate to the backend directory and start the Flask API server
cd src/backend
python api.py
```

The backend server will start on `http://localhost:5000`.

**Start the Frontend Development Server**:
```bash
# In a new terminal, navigate to the frontend directory
cd src/frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the next available port).

#### 🎯 Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:5173`
2. **Configure your settings** using the Config button in the top-right corner:
   - Set your API keys
   - Choose your preferred model (GPT-3.5, GPT-4, Claude, etc.)
   - Configure optimization parameters
3. **Enter your prompt** in the main input area
4. **Click "Optimize"** to start the optimization process
5. **Provide feedback** on the optimized prompt to improve it further
6. **Download sessions** to save your work for later

#### 🌟 Web App Features

- **Real-time Optimization**: Watch as your prompts are optimized in real-time
- **Interactive Feedback**: Click on any part of the prompt to provide specific feedback
- **Session Management**: Save and load optimization sessions
- **Configuration Panel**: Easy access to all optimization settings
- **Progress Tracking**: Monitor optimization progress and metrics
- **Export Options**: Download optimized prompts and session data

#### 🛠️ Development Mode

For developers who want to modify the web app:

```bash
# Backend development (with auto-reload)
cd src/backend
python api.py  # Flask runs in debug mode by default

# Frontend development (with hot reload)
cd src/frontend
npm run dev    # Vite provides hot module replacement
```

#### 📁 Project Structure

```
src/
├── backend/
│   └── api.py              # Flask API server
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.jsx         # Main app component
│   │   └── main.jsx        # App entry point
│   ├── package.json        # Frontend dependencies
│   └── vite.config.js      # Vite configuration
└── promptomatix/            # Core Python package
```

#### 🔍 Troubleshooting

**Port Already in Use**:
```bash
# Check what's using the port
lsof -i :5000  # For backend
lsof -i :5173  # For frontend

# Kill the process or use different ports
```

**API Key Issues**:
```bash
# Verify your API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Or check the .env file
cat .env
```

**Frontend Build Issues**:
```bash
# Clear node modules and reinstall
cd src/frontend
rm -rf node_modules package-lock.json
npm install
```

## 🛣️ Supported Features

### Task Types
- **Classification**: Text categorization tasks
- **Question Answering**: Extracting answers from contexts
- **Generation**: Creative text generation
- **Summarization**: Condensing longer texts
- **Translation**: Converting between languages
- **Reasoning**: Step-by-step problem solving

### LLM Providers
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Cohere
- Custom providers via LM Manager

### Optimization Techniques
- DSPy-based optimization
- MIPROv2 trainer
- Task-specific metrics
- Human feedback integration
- Synthetic data generation

## 🤝 Contributing

[TODO]

## 📄 License

[TODO]

## 📧 Contact

[TODO]

---

<p align="center">
  <b>Promptomatix: Optimizing LLM prompts, so you don't have to.</b>
</p>
