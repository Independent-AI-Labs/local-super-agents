"""
Global configuration settings for VibeCheck.
"""

import os
from pathlib import Path

# Directory structure
VIBECHECK_DIR = ".vibecheck"
ARCHITECTURE_DIR = os.path.join(VIBECHECK_DIR, "architecture")
ARCHITECTURE_DOCS_DIR = os.path.join(ARCHITECTURE_DIR, "docs")
ARCHITECTURE_DIAGRAMS_DIR = os.path.join(ARCHITECTURE_DIR, "diagrams")
IMPLEMENTATION_DIR = os.path.join(VIBECHECK_DIR, "implementation")
IMPLEMENTATION_SECURITY_DIR = os.path.join(IMPLEMENTATION_DIR, "security")
ENVIRONMENT_DIR = os.path.join(VIBECHECK_DIR, "environment")
TESTS_DIR = os.path.join(VIBECHECK_DIR, "tests")
CACHE_DIR = os.path.join(VIBECHECK_DIR, "cache")

# File paths
PROJECT_FILE = os.path.join(VIBECHECK_DIR, "project.json")
IMPLEMENTATION_PROGRESS_FILE = os.path.join(IMPLEMENTATION_DIR, "progress.json")
ENVIRONMENT_CONFIG_FILE = os.path.join(ENVIRONMENT_DIR, "config.json")
TEST_RESULTS_FILE = os.path.join(TESTS_DIR, "results.json")

# LLM Integration
MODEL_FOR_ARCH_ANALYSIS = "qwen2.5-coder:14b"
PROMPT_FOR_ARCH_ANALYSIS = """
Analyze the following software architecture document and identify key components, 
relationships, and potential issues. Your analysis should focus on:

1. Main components and their responsibilities
2. Interfaces between components
3. Data flow patterns
4. Potential bottlenecks or design issues
5. Security considerations
"""

MODEL_FOR_SECURITY = "qwen2.5-coder:14b"
PROMPT_FOR_SECURITY = """
Analyze these security vulnerabilities and provide insights on the underlying issues, 
severity, and recommended fixes. Focus on:

1. Root causes of each vulnerability
2. Potential exploit scenarios
3. Business impact if exploited
4. Recommended mitigation strategies
5. Long-term architectural changes to prevent similar issues
"""

MODEL_FOR_CODE_ANALYSIS = "qwen2.5-coder:14b"
PROMPT_FOR_CODE_ANALYSIS = """
Analyze the provided code for quality, maintainability, and adherence to software 
engineering best practices. Focus on:

1. Code structure and organization
2. Error handling and edge cases
3. Performance considerations
4. Adherence to design patterns
5. Potential bugs or issues
"""

MODEL_FOR_TEST_GENERATION = "codellama-34b"
PROMPT_FOR_TEST_GENERATION = """
Generate comprehensive test cases for the provided code. Include:

1. Unit tests for individual functions
2. Integration tests for component interactions
3. Edge case scenarios
4. Boundary value tests
5. Error handling tests
"""

# Default file extensions to include in analysis
DEFAULT_INCLUDE_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".cs"
]

# Default directories to exclude from analysis
DEFAULT_EXCLUDE_DIRS = [
    "node_modules", "venv", ".env", ".git",
    "dist", "build", "__pycache__", ".pytest_cache",
    VIBECHECK_DIR
]

# Cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_AGE_DAYS = 7    # Maximum age of cache entries in days
