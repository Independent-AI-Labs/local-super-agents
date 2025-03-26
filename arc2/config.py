"""Configuration settings for the ARC solver application."""

import os
from pathlib import Path

# API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDRNmcVDMvVENa2MDY9FfKvnZuIUxYHEQs")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", 'models/gemini-2.0-flash-lite')

# Path Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = os.environ.get("ARC_DATA_DIR", BASE_DIR / "data")
TRAINING_DATA_DIR = Path(DATA_DIR) / "training"
RESULTS_DIR = os.environ.get("ARC_RESULTS_DIR", BASE_DIR / "results")
CACHE_DIR = os.environ.get("ARC_CACHE_DIR", BASE_DIR / "cache")

# Cache Configuration
USE_DESCRIPTION_CACHE = True

# Grid and Object Configuration
BACKGROUND_VALUE = 0  # Value representing the background in grid
MIN_OBJECT_SIZE = 1   # Minimum number of cells for a valid object

# Create directories if they don't exist
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
