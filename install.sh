#!/bin/bash

echo "🚀 Installing Promptomatix..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

# Check if we're already in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
else
    echo "📦 Creating virtual environment..."
    
    # Create virtual environment
    python3 -m venv promptomatix_env
    
    # Activate virtual environment
    echo " Activating virtual environment..."
    source promptomatix_env/bin/activate
    
    echo "✅ Virtual environment created and activated!"
fi

# Check if git submodules are initialized
if [ ! -d "libs/dspy" ]; then
    echo "📦 Initializing git submodules..."
    git submodule update --init --recursive
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install DSPy from submodule first
echo "📦 Installing DSPy from submodule..."
if [ -d "libs/dspy" ]; then
    pip install -e libs/dspy/
    echo "✅ DSPy installed successfully!"
else
    echo "⚠️  DSPy submodule not found, installing from PyPI..."
    pip install dspy>=2.6.0
fi

# Install the package with ALL dependencies
echo "📦 Installing Promptomatix with all dependencies..."
python setup.py install

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📝 IMPORTANT: You need to activate the virtual environment each time you use Promptomatix"
echo ""
echo "🔧 To activate the environment:"
echo "   source promptomatix_env/bin/activate"
echo ""
echo "🔧 To deactivate the environment:"
echo "   deactivate"
echo ""
echo "🔑 Set up your API keys:"
echo "   1. Copy the sample environment file:"
echo "      cp .env.example .env"
echo "   2. Edit .env and add your API keys:"
echo "      nano .env  # or use any text editor"
echo "   3. Make sure to replace 'your_key_here' with your actual API keys"
echo ""
echo " Quick start:"
echo "   1. Activate: source promptomatix_env/bin/activate"
echo "   2. Set up .env file (see above)"
echo "   3. Test: promptomatix --raw_input 'Classify sentiment'"
echo ""
echo "💡 Pro tip: Add this to your ~/.bashrc or ~/.zshrc to activate automatically:"
echo "   alias promptomatix='source promptomatix_env/bin/activate && promptomatix'"
echo ""
echo "📚 For more help: promptomatix --help" 