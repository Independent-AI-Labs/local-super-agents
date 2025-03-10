#!/usr/bin/env python3
"""
KGML Test Results Viewer Utilities

This module contains utility functions for loading, processing, and visualizing KGML test data.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLViewer")

# Default base directory for test logs
DEFAULT_BASE_DIR = "kgml_test_logs"


# ========== Data Loading Functions ==========

def find_test_runs(base_dir: str = DEFAULT_BASE_DIR) -> List[Dict[str, Any]]:
    """
    Find all test runs in the specified directory.

    Args:
        base_dir: Base directory for test logs

    Returns:
        List of dictionaries containing run information
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Base directory not found: {base_dir}")
        return []

    runs = []
    for run_dir in sorted(base_path.iterdir(), key=lambda p: p.name, reverse=True):
        if not run_dir.is_dir():
            continue

        # Try to load summary.txt first (new format)
        summary_file = run_dir / "summary.txt"
        stats_file = run_dir / "stats.json"

        run_info = None

        # Try the new format first (summary.txt)
        if summary_file.exists():
            try:
                run_info = _extract_run_info_from_summary(summary_file, run_dir)
            except Exception as e:
                logger.error(f"Error reading summary for {run_dir}: {e}")

        # Fall back to stats.json if available
        if run_info is None and stats_file.exists():
            try:
                run_info = _extract_run_info_from_stats(stats_file, run_dir)
            except Exception as e:
                logger.error(f"Error reading stats for {run_dir}: {e}")

        # If we have run info, add it to our list
        if run_info:
            runs.append(run_info)

    return runs


def _extract_run_info_from_summary(summary_file: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract run information from summary.txt file (new format)"""
    with open(summary_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse basic info
    run_id = run_dir.name
    model_name = "unknown"
    start_time_str = "Unknown"
    completed_time_str = "Unknown"
    total_tests = 0
    successful_tests = 0

    for i, line in enumerate(lines):
        if line.startswith("Test Run:"):
            run_id = line.split(":", 1)[1].strip()
        elif line.startswith("Model:"):
            model_name = line.split(":", 1)[1].strip()
        elif line.startswith("Started:"):
            start_time_str = line.split(":", 1)[1].strip()
        elif line.startswith("Completed:"):
            completed_time_str = line.split(":", 1)[1].strip()
        elif line.startswith("Tests:"):
            try:
                total_tests = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Successful:"):
            try:
                successful_part = line.split(":", 1)[1].strip()
                # Extract number before the percentage
                successful_tests = int(successful_part.split(" ")[0])
            except (ValueError, IndexError):
                pass

    # Calculate success rate
    success_rate = 0
    if total_tests > 0:
        success_rate = (successful_tests / total_tests) * 100

    # Format the dates for display
    try:
        # Try to parse as ISO format first
        from datetime import datetime
        start_time = datetime.fromisoformat(start_time_str)
        formatted_date = start_time.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        # Fall back to original string if parsing fails
        formatted_date = start_time_str

    # Count valid responses by checking test directories
    valid_responses = 0
    total_responses = 0

    for test_dir in run_dir.iterdir():
        if not test_dir.is_dir() or test_dir.name.startswith("__"):
            continue

        # Check each iteration directory for response files
        for iter_dir in test_dir.iterdir():
            if not iter_dir.is_dir() or not iter_dir.name.startswith("iteration_"):
                continue

            total_responses += 1

            # Check if response was valid
            meta_file = iter_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        import json
                        metadata = json.load(f)
                    if metadata.get("is_valid", False):
                        valid_responses += 1
                except Exception:
                    pass

    return {
        "run_id": run_id,
        "model_name": model_name,
        "formatted_date": formatted_date,
        "tests": total_tests,
        "valid_total": f"{valid_responses}/{total_responses}",
        "success_rate": f"{success_rate:.1f}%",
        "success_rate_value": success_rate,  # For sorting
        "avg_time": 0,  # Not directly available from summary
        "path": str(run_dir)
    }


def _extract_run_info_from_stats(stats_file: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract run information from stats.json file (old format)"""
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Calculate success rate safely
    total_responses = max(1, stats.get("total_responses", 0))
    valid_responses = stats.get("valid_responses", 0)
    success_rate = (valid_responses / total_responses) * 100

    # Format date for display
    start_time_str = stats.get("start_time", "")
    try:
        start_time = datetime.fromisoformat(start_time_str) if start_time_str else None
        formatted_date = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "Unknown"
    except:
        formatted_date = "Unknown"

    # Get model name
    model_name = stats.get("model_name", "unknown")

    # Extract key information
    return {
        "run_id": stats.get("run_id", run_dir.name),
        "model_name": model_name,
        "formatted_date": formatted_date,
        "tests": stats.get("total_tests", 0),
        "valid_total": f"{valid_responses}/{stats.get('total_responses', 0)}",
        "success_rate": f"{success_rate:.1f}%",
        "success_rate_value": success_rate,  # For sorting
        "avg_time": stats.get("avg_response_time", 0) or 0,
        "path": str(run_dir)
    }


def load_run_data(run_path: str) -> Dict[str, Any]:
    """
    Load the data for a specific test run.

    Args:
        run_path: Path to the test run directory

    Returns:
        Dictionary containing run statistics and test data
    """
    run_dir = Path(run_path)
    if not run_dir.exists():
        return {"error": f"Run directory not found: {run_path}"}

    # Try to load from summary.txt (new format) first, then fall back to stats.json
    summary_file = run_dir / "summary.txt"
    stats_file = run_dir / "stats.json"

    run_stats = None

    if summary_file.exists():
        try:
            run_stats = _load_run_stats_from_summary(summary_file)
        except Exception as e:
            logger.error(f"Error loading run data from summary: {e}")

    if run_stats is None and stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                run_stats = json.load(f)
        except Exception as e:
            logger.error(f"Error loading run data from stats: {e}")
            return {"error": f"Error loading run data: {e}"}

    if run_stats is None:
        return {"error": "No valid run data found"}

    # Load test summaries
    tests = []

    # First try to get test names from the summary file
    test_names = []
    if summary_file.exists():
        test_names = _extract_test_names_from_summary(summary_file)

    # If we couldn't get test names from summary, scan the directory
    if not test_names:
        for test_dir in run_dir.iterdir():
            if test_dir.is_dir() and not test_dir.name.startswith("__"):
                test_names.append(test_dir.name)

    # Now process each test
    for test_name in test_names:
        test_dir = run_dir / test_name

        # Check if test directory exists
        if not test_dir.exists():
            continue

        # Load test info from test summary if available
        test_summary_file = test_dir / "summary.txt"
        if test_summary_file.exists():
            test_info = _load_test_info_from_summary(test_summary_file, test_name, test_dir)
            if test_info:
                tests.append(test_info)
                continue

        # Fall back to counting iterations manually
        iteration_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]

        # Try to get metadata from the test
        metadata_file = test_dir / "metadata.txt"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
            except Exception as e:
                logger.error(f"Error reading metadata for {test_dir}: {e}")

        # Count valid responses
        valid_responses = 0
        total_responses = len(iteration_dirs)

        for iter_dir in iteration_dirs:
            meta_file = iter_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        iter_meta = json.load(f)
                    if iter_meta.get("is_valid", False):
                        valid_responses += 1
                except Exception:
                    pass

        # Build test summary
        tests.append({
            "name": test_name,
            "status": "⚠️ UNKNOWN",  # Default status
            "iterations": len(iteration_dirs),
            "valid_total": f"{valid_responses}/{total_responses}",
            "avg_time": 0,  # Not directly available
            "goal_reached": None,
            "iterations_to_goal": None,
            "metadata": metadata,
            "path": str(test_dir),
            "iteration_count": len(iteration_dirs)
        })

    # Return combined data
    return {
        "stats": run_stats,
        "tests": tests,
        "path": run_path
    }


def _load_run_stats_from_summary(summary_file: Path) -> Dict[str, Any]:
    """Load run statistics from summary.txt file"""
    with open(summary_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    stats = {
        "run_id": "",
        "model_name": "",
        "start_time": "",
        "end_time": "",
        "total_tests": 0,
        "total_prompts": 0,
        "total_responses": 0,
        "valid_responses": 0,
        "invalid_responses": 0,
        "syntax_errors": 0,
        "execution_errors": 0
    }

    for line in lines:
        if line.startswith("Test Run:"):
            stats["run_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Model:"):
            stats["model_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Started:"):
            stats["start_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Completed:"):
            stats["end_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Tests:"):
            try:
                stats["total_tests"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Successful:"):
            try:
                successful_part = line.split(":", 1)[1].strip()
                # Extract number before the percentage
                stats["valid_responses"] = int(successful_part.split(" ")[0])
            except (ValueError, IndexError):
                pass

    # Set total_responses based on counting test directories
    summary_dir = summary_file.parent
    response_count = 0

    for test_dir in summary_dir.iterdir():
        if test_dir.is_dir() and not test_dir.name.startswith("__"):
            for iter_dir in test_dir.iterdir():
                if iter_dir.is_dir() and iter_dir.name.startswith("iteration_"):
                    response_count += 1

    stats["total_responses"] = response_count
    stats["invalid_responses"] = response_count - stats["valid_responses"]

    return stats


def _extract_test_names_from_summary(summary_file: Path) -> List[str]:
    """Extract test names from the summary.txt file"""
    test_names = []
    in_test_results_section = False

    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            # Look for the "Test Results:" marker
            if line.strip() == "Test Results:":
                in_test_results_section = True
                continue

            if in_test_results_section and line.strip().startswith("  "):
                # Lines in the format "  test_name: ✅/❌ (Iterations: X)"
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    test_name = parts[0].strip()
                    test_names.append(test_name)

    return test_names


def _load_test_info_from_summary(summary_file: Path, test_name: str, test_dir: Path) -> Optional[Dict[str, Any]]:
    """Load test information from the test summary.txt file"""
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        goal_reached = None
        iterations_to_goal = None
        valid_responses = 0
        total_responses = 0
        avg_time = 0
        total_time = 0
        time_count = 0

        for line in lines:
            if line.startswith("Goal Reached:"):
                goal_str = line.split(":", 1)[1].strip().lower()
                if "true" in goal_str:
                    goal_reached = True
                elif "false" in goal_str:
                    goal_reached = False

            elif line.startswith("Iterations to Goal:"):
                try:
                    iterations_to_goal = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line.strip().startswith("  ") and "Response Valid:" in line:
                total_responses += 1
                if "Response Valid: ✓" in line:
                    valid_responses += 1

                # Try to extract response time if present
                time_match = re.search(r'\((\d+\.\d+)s\)', line)
                if time_match:
                    try:
                        response_time = float(time_match.group(1))
                        total_time += response_time
                        time_count += 1
                    except ValueError:
                        pass

        # If we couldn't extract valid responses from summary, count from iteration directories
        if total_responses == 0:
            # Count iterations and check metadata files
            iteration_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
            total_responses = len(iteration_dirs)

            for iter_dir in iteration_dirs:
                meta_file = iter_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            import json
                            metadata = json.load(f)
                            if metadata.get("is_valid", False):
                                valid_responses += 1

                            # Collect response time info
                            response_time = metadata.get("response_time")
                            if response_time is not None and isinstance(response_time, (int, float)) and response_time > 0:
                                total_time += response_time
                                time_count += 1
                    except Exception as e:
                        logger.error(f"Error reading metadata for {iter_dir}: {e}")

        # Calculate average time
        if time_count > 0:
            avg_time = total_time / time_count

        # Load metadata if available
        metadata_file = test_dir / "metadata.txt"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
            except Exception as e:
                logger.error(f"Error reading metadata for {test_dir}: {e}")

        # Set status based on goal_reached
        if goal_reached is True:
            status = "✅ PASSED"
        elif goal_reached is False:
            status = "🛑 FAILED"
        else:
            status = "⚠️ UNKNOWN"

        return {
            "name": test_name,
            "status": status,
            "iterations": len([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]),
            "valid_total": f"{valid_responses}/{total_responses}",
            "avg_time": avg_time,
            "goal_reached": goal_reached,
            "iterations_to_goal": iterations_to_goal,
            "metadata": metadata,
            "path": str(test_dir),
            "iteration_count": total_responses
        }

    except Exception as e:
        logger.error(f"Error loading test summary for {test_name}: {e}")
        return None


def load_test_iterations(test_path: str) -> List[Dict[str, Any]]:
    """
    Load all iterations for a specific test.

    Args:
        test_path: Path to the test directory

    Returns:
        List of dictionaries containing iteration data
    """
    test_dir = Path(test_path)
    if not test_dir.exists():
        return []

    iterations = []
    for iter_dir in sorted(test_dir.iterdir(), key=lambda p: p.name):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iteration_"):
            continue

        # Try to load from the metadata.json file
        meta_file = iter_dir / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    import json
                    metadata = json.load(f)

                # Extract iteration number
                try:
                    iteration_num = int(iter_dir.name.split("_")[-1])
                except:
                    iteration_num = 0

                # Check for request and response files
                has_request = (iter_dir / "request.kgml").exists()
                has_response = (iter_dir / "response.kgml").exists()

                # Format response time properly
                response_time = metadata.get('response_time', 0)
                if response_time is not None and response_time > 0:
                    response_time_str = f"{response_time:.2f}s"
                else:
                    response_time_str = "N/A"

                # Set iteration status
                if metadata.get("is_valid", False):
                    if metadata.get("execution_success", False):
                        status = "✅ SUCCESSFUL EXECUTION"
                    else:
                        status = "⚠️ FAILED EXECUTION"
                else:
                    status = "🛑 INVALID KGML"

                iter_info = {
                    "iteration": iteration_num,
                    "status": status,
                    "response_time": response_time_str,
                    "response_time_value": response_time or 0,  # For sorting
                    "is_valid": "Yes" if metadata.get("is_valid", False) else "No",
                    "processing_success": "Yes" if metadata.get("execution_success", False) else "No",
                    "path": str(iter_dir)
                }
                iterations.append(iter_info)
                continue
            except Exception as e:
                logger.error(f"Error reading metadata for {iter_dir}: {e}")

        # Fall back to checking for files if metadata.json doesn't exist or has errors
        has_request = (iter_dir / "request.kgml").exists()
        has_response = (iter_dir / "response.kgml").exists()

        if not (has_request or has_response):
            continue

        # Create basic metadata if file doesn't exist
        try:
            iteration_num = int(iter_dir.name.split("_")[-1])

            # Check execution result
            exec_result_file = iter_dir / "execution_result.txt"
            execution_success = None
            response_time = "N/A"

            if exec_result_file.exists():
                with open(exec_result_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    for line in f:
                        if line.startswith("Success:"):
                            exec_success_str = line.split(":", 1)[1].strip().lower()
                            execution_success = "true" in exec_success_str
                        elif line.startswith("Execution Time:"):
                            try:
                                time_val = float(line.split(":", 1)[1].strip().rstrip("s"))
                                response_time = f"{time_val:.2f}s"
                            except:
                                pass

            # Determine status
            if has_response:
                if execution_success is True:
                    status = "✅ SUCCESSFUL EXECUTION"
                elif execution_success is False:
                    status = "⚠️ FAILED EXECUTION"
                else:
                    status = "⚠️ UNKNOWN"
            else:
                status = "🛑 INVALID KGML"

            iterations.append({
                "iteration": iteration_num,
                "status": status,
                "response_time": response_time,
                "response_time_value": 0,
                "is_valid": "Yes" if has_response else "No",
                "processing_success": "Yes" if execution_success is True else "No" if execution_success is False else "Unknown",
                "path": str(iter_dir)
            })

        except Exception as e:
            logger.error(f"Error processing iteration {iter_dir}: {e}")
            continue

    # Sort by iteration number
    return sorted(iterations, key=lambda x: x["iteration"])


def load_iteration_details(iteration_path: str) -> Dict[str, Any]:
    """
    Load details for a specific test iteration.

    Args:
        iteration_path: Path to the iteration directory

    Returns:
        Dictionary containing iteration details
    """
    iter_dir = Path(iteration_path)
    if not iter_dir.exists():
        return {"error": f"Iteration directory not found: {iteration_path}"}

    # Load metadata if available
    meta_file = iter_dir / "metadata.json"
    metadata = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")

    # Load request
    request = ""
    request_file = iter_dir / "request.kgml"
    if request_file.exists():
        try:
            with open(request_file, "r", encoding="utf-8") as f:
                request = f.read()
        except Exception as e:
            logger.error(f"Error reading request file: {e}")

    # Load response
    response = ""
    response_file = iter_dir / "response.kgml"
    if response_file.exists():
        try:
            with open(response_file, "r", encoding="utf-8") as f:
                response = f.read()
        except Exception as e:
            logger.error(f"Error reading response file: {e}")

    # Load processing result
    processing_result = {}

    # First try the JSON processing result (newer format)
    result_file = iter_dir / "processing_result.json"
    if result_file.exists():
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                processing_result = json.load(f)
        except Exception as e:
            logger.error(f"Error reading processing result: {e}")

    # Fall back to execution_result.txt (older format)
    if not processing_result:
        exec_result_file = iter_dir / "execution_result.txt"
        if exec_result_file.exists():
            try:
                with open(exec_result_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse the execution result text
                result = {
                    "success": False,
                    "execution_log": []
                }

                for line in content.split("\n"):
                    if line.startswith("Success:"):
                        result["success"] = "True" in line
                    elif line.startswith("Error:"):
                        result["error"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Execution Time:"):
                        try:
                            result["execution_time"] = float(line.split(":", 1)[1].strip().rstrip("s"))
                        except:
                            pass
                    elif line.startswith("- "):
                        result["execution_log"].append(line[2:])

                processing_result = result
            except Exception as e:
                logger.error(f"Error reading execution result: {e}")

    # Format response time safely
    response_time = metadata.get('response_time_seconds', metadata.get('response_time', 0))
    response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"

    # Processing status
    processing_success = metadata.get('processing_success', metadata.get('execution_success'))
    if processing_success is not None:
        exec_status = "✅ Success" if processing_success else "🛑 Failed"
    else:
        exec_status = "⚠️ Unknown"

    return {
        "metadata": metadata,
        "request": request,
        "response": response,
        "processing_result": processing_result,
        "response_time_str": response_time_str,
        "is_valid": metadata.get("is_valid", False),
        "has_syntax_errors": metadata.get("has_syntax_errors", False),
        "processing_status": exec_status,
        "path": iteration_path
    }


# ========== Visualization Functions ==========

def create_test_result_chart(tests: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a pie chart showing test result distribution.

    Args:
        tests: List of test data dictionaries

    Returns:
        Plotly figure
    """
    # Count results by status
    passed = sum(1 for t in tests if t.get("goal_reached") is True)
    failed = sum(1 for t in tests if t.get("goal_reached") is False)
    unknown = sum(1 for t in tests if t.get("goal_reached") is None)

    labels = ["Passed", "Failed", "Unknown"]
    values = [passed, failed, unknown]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]  # Green, Red, Gray

    # Filter out zero values
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for l, v, c in zip(labels, values, colors):
        if v > 0:
            filtered_labels.append(l)
            filtered_values.append(v)
            filtered_colors.append(c)

    # If no data, add a placeholder
    if not filtered_values:
        filtered_labels = ["No Data"]
        filtered_values = [1]
        filtered_colors = ["#E0E0E0"]  # Light gray

    fig = go.Figure(data=[
        go.Pie(
            labels=filtered_labels,
            values=filtered_values,
            marker_colors=filtered_colors,
            textinfo="value+percent",
            hole=0.4,  # Create a donut chart
            textfont=dict(size=14)
        )
    ])

    fig.update_layout(
        title=dict(
            text="Test Results Distribution",
            font=dict(size=20)
        ),
        template="plotly_white",
        height=400
    )

    return fig


# ========== Summary Generation Functions ==========

def generate_run_summary(stats: Dict[str, Any]) -> str:
    """
    Generate a detailed run summary

    Args:
        stats: Run statistics dictionary

    Returns:
        Markdown formatted run summary
    """
    # Calculate summary stats safely
    total_responses = max(1, stats.get("total_responses", 1))
    valid_responses = stats.get("valid_responses", 0)
    invalid_responses = stats.get("invalid_responses", 0)
    syntax_errors = stats.get("syntax_errors", 0)
    processing_errors = stats.get("processing_errors", 0)

    # Format dates and duration
    start_time_str = stats.get("start_time")
    end_time_str = stats.get("end_time")
    model_name = stats.get("model_name", "unknown")

    start_time = None
    end_time = None
    duration = "Unknown"

    if start_time_str:
        try:
            start_time = datetime.fromisoformat(start_time_str)
        except:
            start_time = None

    if end_time_str:
        try:
            end_time = datetime.fromisoformat(end_time_str)
        except:
            end_time = None

    if start_time and end_time:
        duration = str(end_time - start_time)

    run_summary = f"""
# Run Summary: {stats.get('run_id', 'Unknown')}

**Model:** {model_name}  
**Start Time:** {start_time_str if start_time_str else 'Unknown'}  
**End Time:** {end_time_str if end_time_str else 'In progress'}  
**Duration:** {duration}

**Total Tests:** {stats.get('total_tests', 0)}  
**Total Prompts:** {stats.get('total_prompts', 0)}  
**Total Responses:** {total_responses}

**Valid Responses:** {valid_responses} ({(valid_responses / total_responses * 100):.1f}%)  
**Invalid Responses:** {invalid_responses} ({(invalid_responses / total_responses * 100):.1f}%)  
"""

    # Add syntax and execution errors if available
    if syntax_errors or processing_errors:
        run_summary += f"**Syntax Errors:** {syntax_errors} ({(syntax_errors / total_responses * 100):.1f}%)  \n"
        run_summary += f"**Execution Errors:** {processing_errors} ({(processing_errors / total_responses * 100):.1f}%)  \n"

    # Add response time statistics if available
    avg_response_time = stats.get("avg_response_time")
    min_response_time = stats.get("min_response_time")
    max_response_time = stats.get("max_response_time")

    if avg_response_time is not None:
        run_summary += f"""
**Response Time Statistics:**  
**Average:** {avg_response_time:.2f}s  
"""

        if min_response_time is not None:
            run_summary += f"**Minimum:** {min_response_time:.2f}s  "

        if max_response_time is not None:
            run_summary += f"**Maximum:** {max_response_time:.2f}s"

    return run_summary


def generate_test_summary(test_data: Dict[str, Any]) -> str:
    """
    Generate a detailed test summary

    Args:
        test_data: Test data dictionary

    Returns:
        Markdown formatted test summary
    """
    # Extract goal status from the status string
    goal_status = "⚠️ UNKNOWN"
    if "PASSED" in test_data["status"]:
        goal_status = "✅ REACHED"
    elif "FAILED" in test_data["status"]:
        goal_status = "🛑 FAILED"

    test_summary = f"""
# Test Summary: {test_data['name']}

**Goal:** {goal_status}  
**Valid Responses:** {test_data.get('valid_total', '0/0')}  
**Total Iterations:** {test_data.get('iteration_count', 0)}
"""

    # Add iterations to goal if available
    if test_data.get("iterations_to_goal") is not None:
        test_summary += f"**Iterations to Goal:** {test_data['iterations_to_goal']}\n"

    # Add response time if available
    if test_data.get("avg_time") is not None and test_data.get("avg_time") > 0:
        test_summary += f"\n**Average Response Time:** {test_data.get('avg_time', 0):.2f}s"

    # Add metadata if available
    metadata = test_data.get('metadata', {})
    if metadata:
        test_summary += "\n\n### Metadata\n"

        if 'problem_id' in metadata:
            test_summary += f"**Problem ID:** {metadata['problem_id']}  \n"

        if 'difficulty' in metadata:
            test_summary += f"**Difficulty:** {metadata['difficulty']}  \n"

        if 'description' in metadata:
            test_summary += f"**Description:** {metadata['description']}  \n"

    return test_summary


def create_response_time_chart(tests: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a response time chart for the tests.

    Args:
        tests: List of test data dictionaries

    Returns:
        Plotly figure
    """
    # Filter out tests with no response time
    filtered_tests = [t for t in tests if t.get("avg_time", 0) is not None]

    if not filtered_tests:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No response time data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Sort tests by response time for better visualization
    sorted_tests = sorted(filtered_tests, key=lambda t: t.get("avg_time", 0), reverse=True)
    test_names = [t.get("name", "Unknown") for t in sorted_tests]
    avg_times = [t.get("avg_time", 0) for t in sorted_tests]

    # Truncate long test names
    truncated_names = []
    for name in test_names:
        if len(name) > 25:
            truncated_names.append(name[:22] + "...")
        else:
            truncated_names.append(name)

    fig = go.Figure(data=[
        go.Bar(
            x=avg_times,
            y=truncated_names,
            orientation='h',  # Horizontal bars
            marker_color='#3F51B5',  # Indigo
            text=[f"{time:.2f}s" for time in avg_times],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title=dict(
            text="Average Response Times by Test",
            font=dict(size=20)
        ),
        xaxis_title="Response Time (seconds)",
        yaxis_title="Test",
        template="plotly_white",
        height=max(400, len(filtered_tests) * 40),  # Adjust height based on number of tests
        margin=dict(l=200, r=20, t=70, b=70)  # Increase left margin for test names
    )

    return fig


def create_processing_log_chart(processing_result: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization of the processing log.

    Args:
        processing_result: Processing result dictionary

    Returns:
        Plotly figure
    """
    # Check for execution_log in the processing result
    processing_log = processing_result.get("execution_log", [])
    if not processing_log:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No processing log data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Extract commands and success status
    commands = []
    status = []

    # Handle different log formats
    for entry in processing_log:
        # Check if entry is a string or a dict
        if isinstance(entry, str):
            # Simple string log entry
            commands.append(entry)
            status.append(True)  # Assume success for string logs
        else:
            # Dictionary format log entry
            cmd_type = entry.get("command_type", "Unknown")
            details = entry.get("details", {})

            # Try to create a descriptive command name
            cmd_desc = cmd_type
            if isinstance(details, dict):
                if "entity_type" in details and "uid" in details:
                    cmd_desc = f"{cmd_type} {details['entity_type']} {details['uid']}"
                elif "type" in details and "uid" in details:
                    cmd_desc = f"{cmd_type} {details['type']} {details['uid']}"

            commands.append(cmd_desc)
            status.append(entry.get("success", True))

    # Create color map based on status
    colors = ["#4CAF50" if s else "#F44336" for s in status]  # Green or Red

    fig = go.Figure()

    for i, (cmd, success, color) in enumerate(zip(commands, status, colors)):
        fig.add_trace(go.Bar(
            x=[1],
            y=[i],
            orientation='h',
            width=0.8,
            marker_color=color,
            name=f"Step {i + 1}",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Step {i + 1}: {cmd} ({'Success' if success else 'Failed'})"
        ))

    # Add text annotations for each command
    for i, cmd in enumerate(commands):
        # Truncate long command names for display
        if len(cmd) > 30:
            display_cmd = cmd[:27] + "..."
        else:
            display_cmd = cmd

        fig.add_annotation(
            x=0.5,
            y=i,
            text=display_cmd,
            showarrow=False,
            font=dict(color="white", size=12),
            xanchor="center"
        )

    fig.update_layout(
        title=dict(
            text="Processing Log Flow",
            font=dict(size=20)
        ),
        template="plotly_white",
        showlegend=False,
        height=max(400, len(commands) * 30),  # Adjust height based on number of commands
        yaxis=dict(
            autorange="reversed",  # Reverse the y-axis to show steps from top to bottom
            showticklabels=False,  # Hide y-axis labels
            zeroline=False
        ),
        xaxis=dict(
            showticklabels=False,  # Hide x-axis labels
            zeroline=False,
            range=[0, 1]  # Fix the range
        ),
        margin=dict(l=10, r=10, t=70, b=20)
    )

    return fig
