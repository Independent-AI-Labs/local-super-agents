"""
Web integration for VibeCheck.

This module provides functions for integrating with web services
to research security patterns, best practices, and common issues.
"""

from typing import Dict, List, Tuple

from integration.net.www.chrome.chrome_surfer import search_web, CHROME_DRIVER_POOL


def research_security_patterns(component_name: str) -> Tuple[List[Dict], List[str]]:
    """
    Research security patterns for a given component.

    Args:
        component_name (str): The name of the component to research

    Returns:
        Tuple[List[Dict], List[str]]: A tuple containing filtered data and discovered patterns
    """
    search_terms = [
        f"{component_name} security best practices",
        f"{component_name} common vulnerabilities",
        f"{component_name} security checklist"
    ]
    
    semantic_patterns = [
        "vulnerability",
        "CVE",
        "security issue",
        "best practice",
        "secure coding",
        "mitigation",
        "attack vector",
        "exploit"
    ]
    
    instructions = f"Find recent security best practices and common vulnerabilities for {component_name}"
    
    try:
        filtered_data, discovered_patterns = search_web(
            search_terms,
            semantic_patterns,
            instructions
        )
        return filtered_data, discovered_patterns
    except Exception as e:
        print(f"Error researching security patterns: {e}")
        return [], []


def research_implementation_patterns(component_name: str) -> Tuple[List[Dict], List[str]]:
    """
    Research implementation patterns for a given component.

    Args:
        component_name (str): The name of the component to research

    Returns:
        Tuple[List[Dict], List[str]]: A tuple containing filtered data and discovered patterns
    """
    search_terms = [
        f"{component_name} design patterns",
        f"{component_name} implementation examples",
        f"{component_name} best practices",
        f"{component_name} architecture"
    ]
    
    semantic_patterns = [
        "pattern",
        "implementation",
        "architecture",
        "design",
        "example",
        "tutorial",
        "guide",
        "sample code"
    ]
    
    instructions = f"Find implementation patterns and best practices for {component_name}"
    
    try:
        filtered_data, discovered_patterns = search_web(
            search_terms,
            semantic_patterns,
            instructions
        )
        return filtered_data, discovered_patterns
    except Exception as e:
        print(f"Error researching implementation patterns: {e}")
        return [], []


def check_dependency_security(dependency_name: str, version: str) -> Tuple[bool, List[Dict]]:
    """
    Check security vulnerabilities for a given dependency.

    Args:
        dependency_name (str): The name of the dependency
        version (str): The version of the dependency

    Returns:
        Tuple[bool, List[Dict]]: A tuple containing a boolean indicating if the dependency is secure,
        and a list of vulnerability details
    """
    search_terms = [
        f"{dependency_name} {version} vulnerabilities",
        f"{dependency_name} {version} CVE",
        f"{dependency_name} {version} security advisory"
    ]
    
    semantic_patterns = [
        "vulnerability",
        "CVE",
        "security advisory",
        "exploit",
        "patch",
        "security update",
        "fixed in version"
    ]
    
    instructions = f"Find security vulnerabilities for {dependency_name} version {version}"
    
    try:
        filtered_data, _ = search_web(
            search_terms,
            semantic_patterns,
            instructions
        )
        
        # If we found vulnerabilities, the dependency isn't secure
        is_secure = len(filtered_data) == 0
        
        return is_secure, filtered_data
    except Exception as e:
        print(f"Error checking dependency security: {e}")
        # Assume not secure if there was an error
        return False, []


def cleanup_web_drivers():
    """
    Clean up web drivers when shutting down the application.
    """
    try:
        for driver in CHROME_DRIVER_POOL:
            try:
                driver.quit()
            except Exception as e:
                print(f"Error quitting driver: {e}")
    except Exception as e:
        print(f"Error cleaning up web drivers: {e}")
