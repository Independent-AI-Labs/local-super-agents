"""
LLM integration for VibeCheck.

This module provides functions for integrating with LLM services
to enhance architecture analysis, security analysis, and other tasks.
"""

from typing import Dict, Optional

from vibecheck.config import (
    MODEL_FOR_ARCH_ANALYSIS,
    PROMPT_FOR_ARCH_ANALYSIS,
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


def analyze_architecture(architecture_doc: str) -> str:
    """
    Analyze an architecture document with an LLM.

    Args:
        architecture_doc (str): The architecture document to analyze

    Returns:
        str: Analysis of the architecture
    """
    return analyze_with_llm(
        architecture_doc,
        MODEL_FOR_ARCH_ANALYSIS,
        PROMPT_FOR_ARCH_ANALYSIS
    )


def generate_diagrams_from_architecture(architecture_doc: str) -> Dict[str, str]:
    """
    Generate architectural diagrams using an LLM.

    Args:
        architecture_doc (str): The architecture document to generate diagrams from

    Returns:
        Dict[str, str]: A dictionary of diagram types to SVG content
    """
    prompt = f"""
    Based on the following architecture document, generate SVG diagrams for:
    1. Module diagram - showing the high-level components/modules and their relationships
    2. Dataflow diagram - showing how data flows between components
    3. Security diagram - showing security boundaries and potential attack vectors

    The diagrams should be in SVG format. 
    
    Architecture document:
    {architecture_doc}
    
    Please return the response in the following format:
    <diagram type="module">
    <svg>...</svg>
    </diagram>
    
    <diagram type="dataflow">
    <svg>...</svg>
    </diagram>
    
    <diagram type="security">
    <svg>...</svg>
    </diagram>
    """
    
    response = analyze_with_llm(
        prompt,
        MODEL_FOR_ARCH_ANALYSIS,
        "Generate SVG diagrams based on the architecture document."
    )
    
    # Parse the response to extract the diagrams
    diagrams = {}
    import re
    
    diagram_pattern = r'<diagram type="([^"]+)">\s*(<svg>.*?</svg>)\s*</diagram>'
    for match in re.finditer(diagram_pattern, response, re.DOTALL):
        diagram_type, svg_content = match.groups()
        diagrams[diagram_type] = svg_content
    
    # If the LLM didn't format the response correctly, provide default placeholders
    required_diagrams = ["module", "dataflow", "security"]
    for diagram_type in required_diagrams:
        if diagram_type not in diagrams:
            diagrams[diagram_type] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300"><text x="50" y="50" font-family="Arial" font-size="16">Placeholder {diagram_type} diagram</text></svg>'
    
    return diagrams


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
