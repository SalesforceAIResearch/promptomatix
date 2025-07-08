"""
Module for managing application paths and directories.
"""

import os
from pathlib import Path

# Get the project root directory (where the repository is cloned)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Define base directories relative to project root
LOGS_DIR = PROJECT_ROOT / 'logs'
SESSIONS_DIR = PROJECT_ROOT / 'sessions'

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Define subdirectories
SESSION_LOGS_DIR = LOGS_DIR / 'sessions'
CONFIG_LOGS_DIR = LOGS_DIR / 'config'
OPTIMIZER_LOGS_DIR = LOGS_DIR / 'optimizer'

# Create subdirectories
SESSION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_LOGS_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZER_LOGS_DIR.mkdir(parents=True, exist_ok=True) 