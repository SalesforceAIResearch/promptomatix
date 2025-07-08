"""
Promptomatix: A Powerful Framework for LLM Prompt Optimization.

A comprehensive tool for automating and optimizing large language model (LLM) prompts
using DSPy and advanced optimization techniques.
"""

__version__ = "0.1.0"
__author__ = "Rithesh Murthy"

# Import main components for easy access
from .main import process_input, optimize_with_feedback, save_feedback

__all__ = [
    "__version__",
    "__author__",
    "process_input",
    "optimize_with_feedback", 
    "save_feedback",
]
