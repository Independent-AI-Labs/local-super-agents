"""
Configuration settings for VibeCheck.

This module defines all configuration values for the VibeCheck application.
"""

import os

# Base paths
VIBECHECK_DIR = '.vibecheck'
PROJECT_FILE = os.path.join(VIBECHECK_DIR, 'project.json')

# Architecture management
ARCHITECTURE_DOCS_DIR = os.path.join(VIBECHECK_DIR, 'architecture', 'docs')
ARCHITECTURE_DIAGRAMS_DIR = os.path.join(VIBECHECK_DIR, 'architecture', 'diagrams')
ARCHITECTURE_ANALYSIS_DIR = os.path.join(VIBECHECK_DIR, 'architecture', 'analysis')
ARCHITECTURE_SCOPE_FILE = os.path.join(VIBECHECK_DIR, 'architecture', 'scope.json')

# Implementation tracking
IMPLEMENTATION_DIR = os.path.join(VIBECHECK_DIR, 'implementation')
IMPLEMENTATION_PROGRESS_FILE = os.path.join(VIBECHECK_DIR, 'implementation', 'progress.json')
IMPLEMENTATION_SECURITY_DIR = os.path.join(VIBECHECK_DIR, 'implementation', 'security')

# Environment management
ENVIRONMENT_DIR = os.path.join(VIBECHECK_DIR, 'environment')
ENVIRONMENT_CONFIG_FILE = os.path.join(VIBECHECK_DIR, 'environment', 'config.json')

# Test management
TESTS_DIR = os.path.join(VIBECHECK_DIR, 'tests')
TEST_RESULTS_FILE = os.path.join(VIBECHECK_DIR, 'tests', 'results.json')

# Cache configuration
CACHE_DIR = os.path.join(VIBECHECK_DIR, 'cache')
CACHE_TTL = 86400  # Default cache TTL in seconds (1 day)

# File operations
DEFAULT_INCLUDE_EXTENSIONS = [
    '.py', '.js', '.java', '.cpp', '.c', '.go', '.ts', '.rb', '.php', '.cs',
    '.html', '.css', '.json', '.md', '.yaml', '.yml'
]
DEFAULT_EXCLUDE_DIRS = [
    '.git', 'node_modules', 'venv', 'env', '__pycache__', 'build', 'dist', 'target'
]

# File tracking
MAX_FILE_SIZE_FOR_ANALYSIS = 1024 * 1024  # 1MB
FILE_CHANGE_SCAN_INTERVAL = 1.0  # How often to scan for file changes (in seconds)

# LLM integration settings
# Set the model to qwen2.5-coder:14b for all integrations
MODEL_FOR_ARCH_ANALYSIS = "qwen2.5-coder:14b"
MODEL_FOR_SECURITY = "qwen2.5-coder:14b"
MODEL_FOR_IMPLEMENTATION = "qwen2.5-coder:14b"
MODEL_FOR_TEST = "qwen2.5-coder:14b"
MODEL_FOR_ENVIRONMENT = "qwen2.5-coder:14b"

# TODO Move away.
PROMPT_FOR_SECURITY = """
You are a cybersecurity expert. Analyze the following security vulnerabilities
and provide:
1. Assessment of severity and impact
2. Root causes of the vulnerabilities
3. Recommended fixes and mitigations
4. Best practices to prevent similar issues
"""

# Web integration settings
WEB_SEARCH_ENABLED = True
WEB_SEARCH_DEPTH = 3
WEB_SEARCH_TIMEOUT = 30  # seconds

# Betterscan integration settings
BETTERSCAN_ENABLED = False  # Disabled by default, enable if installed