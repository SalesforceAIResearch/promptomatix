# Promptomatix Codespace Setup

This devcontainer configuration provides a fully configured Python development environment for Promptomatix.

## 🚀 Getting Started

When you open this repository in GitHub Codespaces or a local VS Code devcontainer, the environment will automatically:

1. **Install Python 3.11** - Matching the project's `.python-version`
2. **Initialize Git Submodules** - Download the DSPy library
3. **Create Virtual Environment** - Set up `promptomatix_env`
4. **Install Dependencies** - Install all requirements from `requirements.txt`
5. **Install Promptomatix** - Set up the package in development mode

## 📦 What's Included

### Base Image
- **Python 3.11** on Debian Bookworm
- **Git** with latest version
- **Zsh** with Oh My Zsh for better terminal experience

### VS Code Extensions
- **Python Development**
  - Python extension (ms-python.python)
  - Pylance for IntelliSense
  - Black formatter
  - Flake8 linting
  - isort for import sorting

- **Jupyter Notebooks**
  - Jupyter extension
  - Jupyter Keymap
  - Jupyter Renderers

- **Git & Collaboration**
  - GitHub Pull Requests
  - GitLens
  - Git History
  - Git Graph

- **General Development**
  - GitHub Copilot
  - IntelliCode
  - Path IntelliSense
  - Code Spell Checker
  - YAML support
  - TOML support

### Port Forwarding
- **Port 5000**: Flask API Server (auto-notify)
- **Port 8080**: Jupyter Server (auto-notify)

## 🔧 Usage

### First Time Setup

1. **Open in Codespace/Devcontainer**
   - The environment will build automatically (this may take a few minutes)

2. **Activate Virtual Environment**
   ```bash
   source promptomatix_env/bin/activate
   ```

3. **Set Up API Keys**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit with your actual API keys
   # You can use any editor (nano, vim, or VS Code)
   nano .env
   ```

4. **Test Installation**
   ```bash
   python -m src.promptomatix.main --raw_input "Classify sentiment" \
     --model_name "gpt-3.5-turbo" \
     --backend "simple_meta_prompt" \
     --synthetic_data_size 10 \
     --model_provider "openai"
   ```

### Working with Jupyter Notebooks

The examples directory contains helpful Jupyter notebooks:

```bash
# Navigate to examples
cd examples/notebooks

# Start with basic usage
jupyter notebook 01_basic_usage.ipynb
```

### Running the Flask API

```bash
# Activate environment if not already activated
source promptomatix_env/bin/activate

# Run the API server (if available)
python -m flask run --host=0.0.0.0 --port=5000
```

## 📝 Important Notes

### Virtual Environment
The virtual environment is located at `/workspaces/promptomatix/promptomatix_env/`. 

**Remember to activate it in each new terminal:**
```bash
source promptomatix_env/bin/activate
```

You'll see `(promptomatix_env)` in your terminal prompt when activated.

### API Keys
Never commit your `.env` file with actual API keys. It's already in `.gitignore`.

Required API keys:
- `OPENAI_API_KEY` - For OpenAI models
- `ANTHROPIC_API_KEY` - For Anthropic models (optional)

### Git Configuration
The devcontainer automatically marks the workspace as a safe directory for Git operations.

## 🔍 Troubleshooting

### Virtual Environment Not Found
If the virtual environment isn't created automatically:
```bash
bash .devcontainer/setup.sh
```

Or manually:
```bash
python3 -m venv promptomatix_env
source promptomatix_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Git Submodules Not Initialized
```bash
git submodule update --init --recursive
```

### Dependencies Not Installed
```bash
source promptomatix_env/bin/activate
pip install -r requirements.txt
if [ -d "libs/dspy" ]; then pip install -e libs/dspy/; fi
pip install -e .
```

## 📚 Additional Resources

- [Promptomatix Documentation](../docs/README.md)
- [API Reference](../docs/API_Reference.md)
- [Getting Started Guide](../docs/Getting_Started.md)
- [Examples](../examples/README.md)

## 🆘 Support

For issues or questions:
- Check the [main README](../README.md)
- Review the [Contributing Guide](../CONTRIBUTING.md)
- Contact: rithesh.murthy@salesforce.com
