#!/usr/bin/env python3
"""
KGML Test Results Gradio Viewer - Fixed Version

An interactive web interface for exploring KGML test runs, viewing request-response pairs,
and analyzing test statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import gradio as gr
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLViewer")

# Default base directory for test logs
DEFAULT_BASE_DIR = "kgml_test_logs"


# ========== Utility Functions ==========

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

        stats_file = run_dir / "stats.json"
        if not stats_file.exists():
            continue

        try:
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
            run_info = {
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
            runs.append(run_info)
        except Exception as e:
            logger.error(f"Error reading stats for {run_dir}: {e}")

    return runs


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

    stats_file = run_dir / "stats.json"
    if not stats_file.exists():
        return {"error": f"Stats file not found for run: {run_path}"}

    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Load test summaries
        tests = []
        for test_name, test_stats in stats.get("tests", {}).items():
            safe_name = "".join(c if c.isalnum() else "_" for c in test_name)
            test_dir = run_dir / safe_name

            # Check if test directory exists
            if not test_dir.exists():
                continue

            # Count iterations by checking iteration directories
            iteration_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]

            # Goal reached status
            goal_reached = test_stats.get("goal_reached")
            status = "✅ PASSED" if goal_reached else "❌ FAILED" if goal_reached is not None else "⚠️ UNKNOWN"

            # Safe handling of response counts
            valid_responses = test_stats.get("valid_responses", 0)
            total_responses = test_stats.get("responses", 0)
            valid_total = f"{valid_responses}/{total_responses}"

            # Safe handling of times
            avg_response_time = test_stats.get("avg_response_time", 0)
            if avg_response_time is None:
                avg_response_time = 0

            test_summary = {
                "name": test_name,
                "status": status,
                "iterations": test_stats.get("iterations", 0),
                "valid_total": valid_total,
                "avg_time": avg_response_time,
                "goal_reached": goal_reached,
                "iterations_to_goal": test_stats.get("iterations_to_goal"),
                "metadata": test_stats.get("metadata", {}),
                "path": str(test_dir),
                "iteration_count": len(iteration_dirs)
            }
            tests.append(test_summary)

        # Return combined data
        return {
            "stats": stats,
            "tests": tests,
            "path": run_path
        }
    except Exception as e:
        logger.error(f"Error loading run data: {e}")
        return {"error": f"Error loading run data: {e}"}


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

        meta_file = iter_dir / "metadata.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Check for request, response, and execution result files
            has_request = (iter_dir / "request.kgml").exists()
            has_response = (iter_dir / "response.kgml").exists()
            has_execution_result = (iter_dir / "execution_result.json").exists()

            # Format response time safely
            response_time = metadata.get('response_time_seconds', 0)
            response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"

            # Set iteration status
            if metadata.get("is_valid", False):
                if metadata.get("execution_success", False):
                    status = "✅ Success"
                else:
                    status = "⚠️ Valid but failed execution"
            else:
                status = "❌ Invalid KGML"

            # Extract iteration number
            try:
                iteration_num = int(iter_dir.name.split("_")[-1])
            except:
                iteration_num = 0

            iter_info = {
                "iteration": iteration_num,
                "status": status,
                "response_time": response_time_str,
                "response_time_value": response_time or 0,  # For sorting
                "is_valid": "Yes" if metadata.get("is_valid", False) else "No",
                "execution_success": "Yes" if metadata.get("execution_success", False) else "No",
                "path": str(iter_dir)
            }
            iterations.append(iter_info)
        except Exception as e:
            logger.error(f"Error reading metadata for {iter_dir}: {e}")

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

    meta_file = iter_dir / "metadata.json"
    if not meta_file.exists():
        return {"error": f"Metadata file not found for iteration: {iteration_path}"}

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load request
        request = ""
        request_file = iter_dir / "request.kgml"
        if request_file.exists():
            with open(request_file, "r", encoding="utf-8") as f:
                request = f.read()

        # Load response
        response = ""
        response_file = iter_dir / "response.kgml"
        if response_file.exists():
            with open(response_file, "r", encoding="utf-8") as f:
                response = f.read()

        # Load execution result
        execution_result = {}
        result_file = iter_dir / "execution_result.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                execution_result = json.load(f)

        # Format response time safely
        response_time = metadata.get('response_time_seconds')
        response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"

        # Execution status
        execution_success = metadata.get('execution_success')
        if execution_success is not None:
            exec_status = "✅ Success" if execution_success else "❌ Failed"
        else:
            exec_status = "⚠️ Unknown"

        return {
            "metadata": metadata,
            "request": request,
            "response": response,
            "execution_result": execution_result,
            "response_time_str": response_time_str,
            "is_valid": metadata.get("is_valid", False),
            "has_syntax_errors": metadata.get("has_syntax_errors", False),
            "execution_status": exec_status,
            "path": iteration_path
        }
    except Exception as e:
        logger.error(f"Error loading iteration details: {e}")
        return {"error": f"Error loading iteration details: {e}"}


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


def create_execution_log_chart(execution_result: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization of the execution log.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Plotly figure
    """
    execution_log = execution_result.get("execution_log", [])
    if not execution_log:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No execution log data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Extract commands and success status
    commands = []
    status = []
    for entry in execution_log:
        cmd_type = entry.get("command_type", "Unknown")
        details = entry.get("details", {})

        # Try to create a descriptive command name
        cmd_desc = cmd_type
        if isinstance(details, dict):
            if "entity_type" in details and "uid" in details:
                cmd_desc = f"{cmd_type} {details['entity_type']} {details['uid']}"

        commands.append(cmd_desc)
        status.append(entry.get("success", False))

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
            text="Execution Log Flow",
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


def generate_run_summary(stats: Dict[str, Any]) -> str:
    """Generate a detailed run summary"""

    # Calculate summary stats safely
    total_responses = max(1, stats.get("total_responses", 1))
    valid_responses = stats.get("valid_responses", 0)
    invalid_responses = stats.get("invalid_responses", 0)
    syntax_errors = stats.get("syntax_errors", 0)
    execution_errors = stats.get("execution_errors", 0)

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
**Syntax Errors:** {syntax_errors} ({(syntax_errors / total_responses * 100):.1f}%)  
**Execution Errors:** {execution_errors} ({(execution_errors / total_responses * 100):.1f}%)
    """

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
    """Generate a detailed test summary"""

    # Extract goal status from the status string
    goal_status = "⚠️ UNKNOWN"
    if "PASSED" in test_data["status"]:
        goal_status = "✅ REACHED"
    elif "FAILED" in test_data["status"]:
        goal_status = "❌ FAILED"

    test_summary = f"""
# Test Summary: {test_data['name']}

**Goal:** {goal_status}  
**Valid Responses:** {test_data.get('valid_total', '0/0')}
"""

    if test_data.get("avg_time") is not None:
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


# ========== UI Event Handlers ==========

def refresh_runs() -> Tuple[List[List], List[Dict], str]:
    """
    Refresh the list of test runs

    Returns:
        Tuple containing visible data for the table, full run data, and any error messages
    """
    try:
        runs = find_test_runs(DEFAULT_BASE_DIR)
        logger.info(f"Found {len(runs)} test runs")

        if not runs:
            return [], [], "No test runs found in directory. Check that data exists in the specified location."

        # Format data for display table - note the reordered columns
        visible_data = []
        for run in runs:
            visible_data.append([
                run["success_rate"],
                run["model_name"],
                run["formatted_date"]
            ])

        return visible_data, runs, None
    except Exception as e:
        logger.error(f"Error refreshing runs: {str(e)}")
        return [], [], f"Error refreshing runs: {str(e)}"


def on_run_selected(evt: gr.SelectData, runs_data: List[Dict]) -> Tuple:
    """
    Handle run selection event

    Args:
        evt: Selection event data
        runs_data: Full run data list

    Returns:
        Tuple containing updated UI component values
    """
    try:
        if not runs_data or evt.index[0] >= len(runs_data):
            return [], "", None, None, None, [], []

        # Get the run path from the selected data
        run_path = runs_data[evt.index[0]]["path"]
        logger.info(f"Selected run: {run_path}")

        # Load run data
        run_data = load_run_data(run_path)

        if "error" in run_data:
            return [], f"Error: {run_data['error']}", None, None, None, [], []

        tests = run_data.get("tests", [])

        # Format tests for display table - note the reordered columns
        visible_tests_data = []
        for test in tests:
            visible_tests_data.append([
                test["status"],
                test["valid_total"],
                test["name"]
            ])

        # Create charts
        test_result_chart = create_test_result_chart(tests)
        response_time_chart = create_response_time_chart(tests)

        # Format run summary
        run_summary = generate_run_summary(run_data.get("stats", {}))

        # Clear iterations table
        visible_iterations_data = []

        return visible_tests_data, run_summary, test_result_chart, response_time_chart, None, visible_iterations_data, tests
    except Exception as e:
        logger.error(f"Error handling run selection: {str(e)}")
        return [], f"Error handling run selection: {str(e)}", None, None, None, [], []


def on_test_selected(evt: gr.SelectData, tests_data: List[Dict]) -> Tuple:
    """
    Handle test selection event

    Args:
        evt: Selection event data
        tests_data: Full test data list

    Returns:
        Tuple containing updated UI component values
    """
    try:
        if not tests_data or evt.index[0] >= len(tests_data):
            return [], "", None, []

        # Get the test path from the full data
        test_path = tests_data[evt.index[0]]["path"]
        logger.info(f"Selected test: {test_path}")

        # Load test iterations
        iterations = load_test_iterations(test_path)

        # Format iterations for display table
        visible_iterations_data = []
        for iteration in iterations:
            visible_iterations_data.append([
                iteration["status"],
                iteration["response_time"]
            ])

        # Create test summary
        test_data = tests_data[evt.index[0]]
        test_summary = generate_test_summary(test_data)

        # Removed active_tab as we're handling tab selection separately

        return visible_iterations_data, test_summary, None, iterations
    except Exception as e:
        logger.error(f"Error handling test selection: {str(e)}")
        return [], f"Error handling test selection: {str(e)}", None, []


def on_iteration_selected(evt: gr.SelectData, iterations_data: List[Dict]) -> Tuple:
    """
    Handle iteration selection event

    Args:
        evt: Selection event data
        iterations_data: Full iteration data list

    Returns:
        Tuple containing updated UI component values
    """
    try:
        if not iterations_data or evt.index[0] >= len(iterations_data):
            return "", "", "", None, None

        # Get the iteration path from the full data
        iteration_path = iterations_data[evt.index[0]]["path"]
        logger.info(f"Selected iteration: {iteration_path}")

        # Load iteration details
        details = load_iteration_details(iteration_path)

        if "error" in details:
            return "", "", "", None, f"Error: {details['error']}"

        # Create execution log chart if available
        execution_log_chart = None
        if details.get("execution_result"):
            execution_log_chart = create_execution_log_chart(details["execution_result"])

        # Format execution result for display
        exec_result_str = ""
        if details.get("execution_result"):
            try:
                import json
                exec_result_str = json.dumps(details["execution_result"], indent=2)
            except:
                exec_result_str = str(details["execution_result"])

        # Removed active_tab as we're handling tab selection separately

        return (
            details.get("request", ""),
            details.get("response", ""),
            exec_result_str,
            execution_log_chart,
            None
        )
    except Exception as e:
        logger.error(f"Error handling iteration selection: {str(e)}")
        return "", "", "", None, f"Error handling iteration selection: {str(e)}"


# ========== UI Creation ==========

def create_ui():
    """Create the Gradio UI"""

    with gr.Blocks(title="KGML Test Results Viewer", theme=gr.themes.Soft(), css="""
        .tab-nav button.selected {
            font-weight: bold;
            border-bottom-width: 3px;
        }
        .status-passed {
            color: green;
            font-weight: bold;
        }
        .status-failed {
            color: red;
            font-weight: bold;
        }
        .status-unknown {
            color: orange;
            font-weight: bold;
        }
        .summary-header {
            margin-top: 0;
            padding-top: 0;
        }
        .selected-row {
            background-color: rgba(63, 81, 181, 0.2) !important;
        }
        .dataframe-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        /* Style for consistent row heights */
        .dataframe tbody tr {
            height: 48px !important;
            line-height: 1.2;
            max-height: 48px;
            overflow: hidden;
        }
        /* Make sure the content doesn't overflow */
        .dataframe td, .dataframe th {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Improve readability of Markdown content */
        .prose {
            max-width: 100%;
            line-height: 1.5;
        }
    """) as app:
        gr.Markdown("# 📊 KGML Test Results Viewer")
        gr.Markdown("Interactive viewer for KGML test data and execution results")

        # Create hidden state variables to store full data with paths
        runs_data_state = gr.State([])
        tests_data_state = gr.State([])
        iterations_data_state = gr.State([])

        # Error output for the entire UI
        error_output = gr.Markdown(visible=True)

        # Top section: Tables for runs, tests, and iterations side by side
        with gr.Row():
            # Runs table (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("### Available Test Runs")
                runs_table = gr.Dataframe(
                    headers=["Success Rate", "Model", "Date"],
                    row_count=8,
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    column_widths=["120px", "120px", "160px"]
                )
                refresh_button = gr.Button("🔄 Refresh Test Runs", variant="primary", size="sm")

            # Tests table (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("### Tests in Run")
                tests_table = gr.Dataframe(
                    headers=["Status", "Valid/Total", "Test Name"],
                    row_count=8,
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    column_widths=["100px", "100px", "200px"]
                )

            # Iterations table (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("### Test Iterations")
                iterations_table = gr.Dataframe(
                    headers=["Status", "Response Time"],
                    row_count=8,
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    column_widths=["210px", "120px"]
                )

        # Bottom section: Details tabs
        with gr.Tabs() as tabs:
            # Tab for run analysis
            with gr.Tab("📈 Run Analysis", id="run_analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        run_summary = gr.Markdown(label="Run Summary")

                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.Tab("Test Results"):
                                test_result_chart = gr.Plot(label="Test Results Distribution")

                            with gr.Tab("Response Times"):
                                response_time_chart = gr.Plot(label="Response Times")

            # Tab for test details
            with gr.Tab("🧪 Test Details", id="test_details"):
                test_summary = gr.Markdown(label="Test Summary")

            # Tab for iteration details
            with gr.Tab("🔍 Iteration Details", id="iteration_details"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Request KGML")
                        request_text = gr.Code(
                            label="Request sent to the model",
                            language="sql",  # Use SQL for KGML as it has similar structure
                            interactive=False,
                            lines=15
                        )

                    with gr.Column():
                        gr.Markdown("### Response KGML")
                        response_text = gr.Code(
                            label="Response received from the model",
                            language="sql",  # Use SQL for KGML as it has similar structure
                            interactive=False,
                            lines=15
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Execution Result")
                        execution_result_text = gr.Code(
                            label="Result of executing the KGML",
                            language="json",
                            interactive=False,
                            lines=15
                        )

                    with gr.Column():
                        gr.Markdown("### Execution Log")
                        execution_log_chart = gr.Plot(label="Visualization of the execution steps")

        # Set up event handlers
        refresh_button.click(
            fn=refresh_runs,
            outputs=[runs_table, runs_data_state, error_output]
        )

        # Use SelectData events for table selections with state
        runs_table.select(
            fn=on_run_selected,
            inputs=[runs_data_state],
            outputs=[
                tests_table,
                run_summary,
                test_result_chart,
                response_time_chart,
                error_output,
                iterations_table,
                # Remove tabs from outputs
                tests_data_state
            ]
        ).then(
            # Use a separate function to select the tab
            lambda: gr.Tabs(selected="run_analysis"),
            inputs=None,
            outputs=tabs
        )

        tests_table.select(
            fn=on_test_selected,
            inputs=[tests_data_state],
            outputs=[
                iterations_table,
                test_summary,
                error_output,
                # Remove tabs from outputs
                iterations_data_state
            ]
        ).then(
            # Use a separate function to select the tab
            lambda: gr.Tabs(selected="test_details"),
            inputs=None,
            outputs=tabs
        )

        iterations_table.select(
            fn=on_iteration_selected,
            inputs=[iterations_data_state],
            outputs=[
                request_text,
                response_text,
                execution_result_text,
                execution_log_chart,
                error_output,
                # Remove tabs from outputs
            ]
        ).then(
            # Use a separate function to select the tab
            lambda: gr.Tabs(selected="iteration_details"),
            inputs=None,
            outputs=tabs
        )

        # Load initial data when the UI starts
        app.load(
            fn=refresh_runs,
            outputs=[runs_table, runs_data_state, error_output]
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(debug=True)
