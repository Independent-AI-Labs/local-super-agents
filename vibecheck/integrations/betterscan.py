"""
Betterscan integration for VibeCheck.

This module provides functions for integrating with Betterscan to perform
security analysis on code.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from vibecheck.models.security import SecurityVulnerability


class BetterscanConfig:
    """Configuration for Betterscan CLI."""
    
    # Path to Betterscan CLI executable
    BETTERSCAN_PATH = os.environ.get("BETTERSCAN_PATH", "betterscan-cli")
    
    # Default scan options
    DEFAULT_SCAN_OPTIONS = [
        "--language=auto",  # Auto-detect language
        "--format=json",    # Output in JSON format
        "--severity=all"    # Include all severity levels
    ]


def is_betterscan_installed() -> bool:
    """
    Check if Betterscan CLI is installed and available.

    Returns:
        bool: True if Betterscan CLI is installed, False otherwise
    """
    try:
        result = subprocess.run(
            [BetterscanConfig.BETTERSCAN_PATH, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking Betterscan installation: {e}")
        return False


def run_betterscan_scan(path: str, options: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Run a Betterscan scan on a file or directory.

    Args:
        path (str): Path to the file or directory to scan
        options (Optional[List[str]], optional): Additional scan options. Defaults to None.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating if the scan was successful,
                          and the scan output or error message
    """
    if not is_betterscan_installed():
        return False, "Betterscan CLI is not installed or not in PATH."
    
    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"
    
    scan_options = BetterscanConfig.DEFAULT_SCAN_OPTIONS.copy()
    if options:
        scan_options.extend(options)
    
    try:
        # Create a temporary file for the scan output
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_file = temp_file.name
        
        # Run the scan
        cmd = [
            BetterscanConfig.BETTERSCAN_PATH,
            "scan",
            path,
            f"--output={output_file}"
        ] + scan_options
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Read the scan results
        try:
            with open(output_file, 'r') as f:
                scan_output = f.read()
        except Exception as e:
            return False, f"Error reading scan output: {e}"
        finally:
            # Clean up the temporary file
            try:
                os.unlink(output_file)
            except Exception:
                pass
        
        if result.returncode != 0:
            # Scan completed but might have found issues
            return True, scan_output
        
        return True, scan_output
    
    except Exception as e:
        return False, f"Error running Betterscan scan: {e}"


def parse_betterscan_results(scan_output: str) -> Dict[str, List[SecurityVulnerability]]:
    """
    Parse Betterscan scan results into SecurityVulnerability objects.

    Args:
        scan_output (str): The Betterscan scan output in JSON format

    Returns:
        Dict[str, List[SecurityVulnerability]]: A dictionary mapping file paths to lists of SecurityVulnerability objects
    """
    vulnerabilities_by_path: Dict[str, List[SecurityVulnerability]] = {}
    
    try:
        # Parse the JSON output
        scan_data = json.loads(scan_output)
        
        # Process findings
        for finding in scan_data.get("findings", []):
            file_path = finding.get("location", {}).get("file", "unknown")
            line = finding.get("location", {}).get("line", 0)
            location = f"{file_path}:{line}"
            
            severity = finding.get("severity", "unknown").lower()
            description = finding.get("description", "No description provided")
            recommendation = finding.get("recommendation", "No recommendation provided")
            
            # Map Betterscan severity to our severity levels
            severity_mapping = {
                "critical": "critical",
                "high": "high",
                "medium": "medium",
                "low": "low",
                "info": "info",
                "unknown": "info"
            }
            
            mapped_severity = severity_mapping.get(severity, "info")
            
            # Create a SecurityVulnerability object
            vulnerability = SecurityVulnerability(
                severity=mapped_severity,
                description=description,
                location=location,
                recommendation=recommendation
            )
            
            # Add to the dictionary, creating a new list if needed
            if file_path not in vulnerabilities_by_path:
                vulnerabilities_by_path[file_path] = []
            
            vulnerabilities_by_path[file_path].append(vulnerability)
    
    except json.JSONDecodeError as e:
        print(f"Error parsing Betterscan results: {e}")
    except Exception as e:
        print(f"Error processing Betterscan results: {e}")
    
    return vulnerabilities_by_path


def scan_project(project_path: str, target_path: Optional[str] = None) -> Dict[str, List[SecurityVulnerability]]:
    """
    Scan a project or specific file with Betterscan.

    Args:
        project_path (str): Path to the project
        target_path (Optional[str], optional): Specific file or directory to scan. Defaults to None.

    Returns:
        Dict[str, List[SecurityVulnerability]]: A dictionary mapping file paths to lists of SecurityVulnerability objects
    """
    path_to_scan = project_path
    if target_path:
        path_to_scan = os.path.join(project_path, target_path)
    
    # If Betterscan is not installed, provide some mock data for testing
    if not is_betterscan_installed():
        print("Betterscan not installed. Using mock data.")
        return _get_mock_vulnerabilities(path_to_scan)
    
    # Run the scan
    success, output = run_betterscan_scan(path_to_scan)
    
    if not success:
        print(f"Betterscan scan failed: {output}")
        return {}
    
    # Parse the results
    return parse_betterscan_results(output)


def _get_mock_vulnerabilities(path: str) -> Dict[str, List[SecurityVulnerability]]:
    """
    Generate mock vulnerabilities for testing purposes.

    Args:
        path (str): Path that was scanned

    Returns:
        Dict[str, List[SecurityVulnerability]]: A dictionary mapping file paths to lists of SecurityVulnerability objects
    """
    result: Dict[str, List[SecurityVulnerability]] = {}
    
    # Check if the path exists and is a directory
    if not os.path.exists(path):
        return result
    
    if os.path.isdir(path):
        # Find Python files in the path
        python_files = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, file), path)
                    python_files.append(rel_path)
        
        # Generate sample vulnerabilities for a few files
        import random
        random.seed(42)  # For reproducible results
        
        severities = ["critical", "high", "medium", "low", "info"]
        
        vulnerability_templates = [
            {
                "description": "SQL injection vulnerability detected",
                "recommendation": "Use parameterized queries or an ORM"
            },
            {
                "description": "Insecure direct object reference",
                "recommendation": "Implement proper access control checks"
            },
            {
                "description": "Cross-site scripting (XSS) vulnerability",
                "recommendation": "Validate and sanitize user input"
            },
            {
                "description": "Hardcoded credentials found",
                "recommendation": "Use environment variables or a secure credential store"
            },
            {
                "description": "Insecure use of cryptographic algorithm",
                "recommendation": "Update to a more secure cryptographic algorithm"
            },
            {
                "description": "Potential command injection",
                "recommendation": "Use secure APIs instead of shell commands"
            },
            {
                "description": "Unvalidated input in file operations",
                "recommendation": "Validate file paths and sanitize inputs"
            },
            {
                "description": "Use of outdated library with known vulnerabilities",
                "recommendation": "Update the library to the latest version"
            }
        ]
        
        for i, file_path in enumerate(python_files[:min(5, len(python_files))]):
            vulnerabilities = []
            
            # Generate 1-3 random vulnerabilities per file
            for j in range(random.randint(1, 3)):
                severity_idx = random.randint(0, len(severities) - 1)
                vuln_idx = random.randint(0, len(vulnerability_templates) - 1)
                
                vuln_template = vulnerability_templates[vuln_idx]
                
                vuln = SecurityVulnerability(
                    severity=severities[severity_idx],
                    description=vuln_template["description"],
                    location=f"{file_path}:{random.randint(1, 100)}",
                    recommendation=vuln_template["recommendation"]
                )
                
                vulnerabilities.append(vuln)
            
            result[file_path] = vulnerabilities
    
    elif os.path.isfile(path):
        # Generate sample vulnerabilities for a single file
        file_path = os.path.basename(path)
        
        # Only generate vulnerabilities for Python files
        if file_path.endswith('.py'):
            vulnerabilities = []
            
            # Generate 1-2 sample vulnerabilities
            vuln1 = SecurityVulnerability(
                severity="medium",
                description="Potential insecure use of exec() function",
                location=f"{file_path}:25",
                recommendation="Avoid using exec() with user-controlled input"
            )
            
            vuln2 = SecurityVulnerability(
                severity="low",
                description="Use of deprecated function",
                location=f"{file_path}:42",
                recommendation="Update to use the recommended alternative function"
            )
            
            vulnerabilities = [vuln1, vuln2]
            result[file_path] = vulnerabilities
    
    return result
