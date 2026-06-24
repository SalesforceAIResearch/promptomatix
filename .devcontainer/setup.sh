#!/bin/bash
# Promptomatix Codespace Setup Script
# This script runs automatically when the devcontainer is created

set -e  # Exit on error

echo "🚀 Setting up Promptomatix development environment..."

# Initialize git submodules
echo "📦 Initializing git submodules..."
git submodule update --init --recursive

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv promptomatix_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source promptomatix_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install DSPy from submodule if available
if [ -d "libs/dspy" ]; then
    echo "📦 Installing DSPy from submodule..."
    pip install -e libs/dspy/
else
    echo "⚠️  DSPy submodule not found, will be installed via requirements.txt"
fi

# Install Promptomatix in editable mode
echo "📦 Installing Promptomatix in editable mode..."
pip install -e .

echo ""
echo "✅ Promptomatix environment ready!"
echo ""
echo "📝 Next steps:"
echo "  1. Activate the environment: source promptomatix_env/bin/activate"
echo "  2. Copy .env.example to .env: cp .env.example .env"
echo "  3. Edit .env with your API keys"
echo "  4. Start developing!"
echo ""
