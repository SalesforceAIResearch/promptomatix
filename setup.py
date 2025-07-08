#!/usr/bin/env python3
"""
Complete setup script for Promptomatix with all dependencies.
Run: python setup.py install
"""

import os
import subprocess
import sys
from setuptools import setup, find_packages

def download_nltk_data():
    """Download required NLTK data during installation."""
    try:
        import nltk
        print("📥 Downloading NLTK data...")
        
        # Download required NLTK data
        required_data = [
            'tokenizers/punkt',
            'taggers/averaged_perceptron_tagger', 
            'corpora/wordnet'
        ]
        
        for data_path in required_data:
            try:
                nltk.data.find(data_path)
                print(f"  ✓ {data_path} already available")
            except LookupError:
                print(f"  📥 Downloading {data_path}...")
                nltk.download(data_path.split('/')[0], quiet=True)
                print(f"  ✓ Downloaded {data_path}")
        
        print("✅ NLTK data download complete!")
        
    except ImportError:
        print("⚠️  NLTK not available during setup, will download at runtime")
    except Exception as e:
        print(f"⚠️  Error downloading NLTK data: {e}")

def main():
    # Download NLTK data during installation
    download_nltk_data()
    
    # Install Promptomatix with ALL dependencies
    setup(
        name="promtomatic",
        version="0.1.0",
        description="A Powerful Framework for LLM Prompt Optimization",
        author="Rithesh Murthy",
        author_email="rithesh.murthy@salesforce.com",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=[
            # Core dependencies
            "dspy>=2.6.0",
            "openai>=1.0.0", 
            "anthropic>=0.18.0",
            "nltk>=3.8",
            "requests>=2.25.0",
            "flask>=2.0.0",
            "pydantic>=2.0.0",
            "python-dotenv>=0.19.0",
            
            # Metrics and evaluation (corrected package names)
            "rouge-score>=0.1.2",  # This installs as rouge_score
            "rouge>=1.0.0",
            "bert-score>=0.3.13",
            "langdetect>=1.0.9",
            
            # Data processing
            "datasets>=2.14.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            
            # Utilities
            "backoff>=2.2.0",
            "pyyaml>=6.0",
            "colorama>=0.4.6",
            "ujson>=5.0.0",
            "litellm",
        ],
        python_requires=">=3.8",
        entry_points={
            "console_scripts": [
                "promtomatic=promtomatic.cli.parser:main",
            ],
        },
        include_package_data=True,
        package_data={
            "promtomatic": ["*.txt", "*.md", "*.yml", "*.yaml"],
        },
    )
    
    print("\n🎉 Promptomatix installed successfully!")
    print("\nQuick start:")
    print("  1. cp .env.example .env")
    print("  2. Edit .env with your API keys")
    print("  3. source promptomatix_env/bin/activate")
    print("  4. promtomatic --raw_input 'Classify sentiment'")

if __name__ == "__main__":
    main() 
