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

# LLM prompt templates
PROMPT_FOR_ARCH_ANALYSIS = """
You are a software architecture expert. Analyze the following architecture document 
and provide insights about:
1. Overall architecture quality and completeness
2. Component relationships and dependencies
3. Potential issues or areas for improvement
4. Suggestions for implementation
"""

PROMPT_FOR_SECURITY = """
You are a cybersecurity expert. Analyze the following security vulnerabilities
and provide:
1. Assessment of severity and impact
2. Root causes of the vulnerabilities
3. Recommended fixes and mitigations
4. Best practices to prevent similar issues
"""

PROMPT_FOR_IMPLEMENTATION = """
You are a senior software engineer. Analyze the following implementation code
and provide:
1. Assessment of code quality and completeness
2. Adherence to the architectural design
3. Potential optimizations and improvements
4. Suggestions for testing
"""

PROMPT_FOR_TEST = """
You are a quality assurance expert. Analyze the following test results
and provide:
1. Assessment of test coverage and quality
2. Potential gaps in testing
3. Suggestions for additional tests
4. Root causes of test failures
"""

PROMPT_FOR_ENVIRONMENT = """
You are a DevOps expert. Analyze the following environment configuration
and provide:
1. Assessment of completeness and correctness
2. Potential issues or bottlenecks
3. Security considerations
4. Suggestions for improvement
"""

# Web integration settings
WEB_SEARCH_ENABLED = True
WEB_SEARCH_DEPTH = 3
WEB_SEARCH_TIMEOUT = 30  # seconds

# Betterscan integration settings
BETTERSCAN_ENABLED = False  # Disabled by default, enable if installed