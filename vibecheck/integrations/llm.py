"""
LLM integration for VibeCheck.

This module provides functions for integrating with LLM services
to enhance architecture analysis, security analysis, and other tasks.
"""

from typing import Dict, Optional

from vibecheck.config import (
    MODEL_FOR_SECURITY,
    PROMPT_FOR_SECURITY
)
from integration.net.ollama.ollama_api import prompt_model


def analyze_with_llm(content: str, model: str, system_prompt: str) -> str:
    """
    Analyze content using an LLM.

    Args:
        content (str): The content to analyze
        model (str): The LLM model to use
        system_prompt (str): The system prompt to use

    Returns:
        str: The LLM's response
    """
    try:
        response = prompt_model(
            content,
            model=model,
            system_prompt=system_prompt
        )
        return response
    except Exception as e:
        print(f"Error analyzing with LLM: {e}")
        return f"LLM analysis failed: {str(e)}"

def analyze_security_vulnerabilities(vulnerabilities_text: str) -> str:
    """
    Analyze security vulnerabilities with an LLM.

    Args:
        vulnerabilities_text (str): Description of the vulnerabilities

    Returns:
        str: Analysis and recommendations for the vulnerabilities
    """
    return analyze_with_llm(
        vulnerabilities_text,
        MODEL_FOR_SECURITY,
        PROMPT_FOR_SECURITY
    )
