#!/usr/bin/env python3
"""
KGML Test Results Viewer Controller

This module serves as the controller component in the MVC pattern for the KGML Test Results Viewer.
It handles the business logic and mediates between the model (kgml_test_result_viewer_util.py)
and the view (kgml_test_result_viewer.py).
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

import plotly.graph_objects as go

# Import utility functions from the model component
from knowledge.reasoning.tests.util.kgml_test_result_viewer_util import (
    find_test_runs,
    load_run_data,
    load_test_iterations,
    load_iteration_details,
    create_test_result_chart,
    create_response_time_chart,
    create_processing_log_chart,
    generate_run_summary,
    generate_test_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLViewerController")


class KGMLTestResultsController:
    """
    Controller class for KGML Test Results Viewer.
    Handles all business logic and interactions between the model and view.
    """

    def __init__(self, base_dir: str = "kgml_test_logs"):
        """
        Initialize the controller with a base directory for test logs.

        Args:
            base_dir: Base directory for KGML test logs
        """
        self.base_dir = base_dir
        self.current_run = None
        self.current_test = None
        self.current_iteration = None

    def get_test_runs(self) -> Tuple[List[List], List[Dict], Optional[str]]:
        """
        Retrieve all test runs and format them for display.

        Returns:
            Tuple containing:
            - List of formatted data for display
            - List of original run data
            - Error message (if any)
        """
        try:
            runs = find_test_runs(self.base_dir)
            logger.info(f"Found {len(runs)} test runs in {self.base_dir}")

            if not runs:
                return [], [], "No test runs found in directory. Check that data exists in the specified location."

            # Format data for display table
            visible_data = []
            for run in runs:
                visible_data.append([
                    run["success_rate"],
                    run["model_name"],
                    run["formatted_date"]
                ])

            return visible_data, runs, None
        except Exception as e:
            logger.error(f"Error getting test runs: {str(e)}")
            return [], [], f"Error getting test runs: {str(e)}"

    def get_run_details(self, run_index: int, runs_data: List[Dict]) -> Tuple[List[List], str, go.Figure, go.Figure, str, List[List], List[Dict]]:
        """
        Get details for a specific test run.

        Args:
            run_index: Index of the selected run
            runs_data: List of run data dictionaries

        Returns:
            Tuple containing:
            - List of formatted test data for display
            - Run summary markdown
            - Test result chart
            - Response time chart
            - Error message (if any)
            - Empty iterations table
            - List of original test data
        """
        try:
            if not runs_data or run_index >= len(runs_data):
                return [], "", None, None, "Invalid run selection", [], []

            # Get the run path from the selected data
            run_path = runs_data[run_index]["path"]
            logger.info(f"Getting details for run: {run_path}")
            self.current_run = run_path

            # Load run data
            run_data = load_run_data(run_path)

            if "error" in run_data:
                return [], "", None, None, f"Error: {run_data['error']}", [], []

            tests = run_data.get("tests", [])

            # Format tests for display table
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

            return visible_tests_data, run_summary, test_result_chart, response_time_chart, "", visible_iterations_data, tests
        except Exception as e:
            logger.error(f"Error getting run details: {str(e)}")
            return [], "", None, None, f"Error getting run details: {str(e)}", [], []

    def get_test_details(self, test_index: int, tests_data: List[Dict]) -> Tuple[List[List], str, str, List[Dict]]:
        """
        Get details for a specific test.

        Args:
            test_index: Index of the selected test
            tests_data: List of test data dictionaries

        Returns:
            Tuple containing:
            - List of formatted iteration data for display
            - Test summary markdown
            - Error message (if any)
            - List of original iteration data
        """
        try:
            if not tests_data or test_index >= len(tests_data):
                return [], "", "Invalid test selection", []

            # Get the test path from the full data
            test_path = tests_data[test_index]["path"]
            logger.info(f"Getting details for test: {test_path}")
            self.current_test = test_path

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
            test_data = tests_data[test_index]
            test_summary = generate_test_summary(test_data)

            return visible_iterations_data, test_summary, "", iterations
        except Exception as e:
            logger.error(f"Error getting test details: {str(e)}")
            return [], "", f"Error getting test details: {str(e)}", []

    def get_iteration_details(self, iteration_index: int, iterations_data: List[Dict]) -> Tuple[str, str, str, go.Figure, str]:
        """
        Get details for a specific test iteration.

        Args:
            iteration_index: Index of the selected iteration
            iterations_data: List of iteration data dictionaries

        Returns:
            Tuple containing:
            - Request KGML text
            - Response KGML text
            - Processing result text
            - Processing log chart
            - Error message (if any)
        """
        try:
            if not iterations_data or iteration_index >= len(iterations_data):
                return "", "", "", None, "Invalid iteration selection"

            # Get the iteration path from the full data
            iteration_path = iterations_data[iteration_index]["path"]
            logger.info(f"Getting details for iteration: {iteration_path}")
            self.current_iteration = iteration_path

            # Load iteration details
            details = load_iteration_details(iteration_path)

            if "error" in details:
                return "", "", "", None, f"Error: {details['error']}"

            # Create processing log chart if available
            processing_log_chart = None
            if details.get("processing_result"):
                processing_log_chart = create_processing_log_chart(details["processing_result"])

            # Format processing result for display
            exec_result_str = ""
            if details.get("processing_result"):
                try:
                    import json
                    exec_result_str = json.dumps(details["processing_result"], indent=2)
                except:
                    exec_result_str = str(details["processing_result"])

            return (
                details.get("request", ""),
                details.get("response", ""),
                exec_result_str,
                processing_log_chart,
                ""
            )
        except Exception as e:
            logger.error(f"Error getting iteration details: {str(e)}")
            return "", "", "", None, f"Error getting iteration details: {str(e)}"

    def export_run_data(self, run_path: str, output_format: str = "json") -> Tuple[bool, str, Optional[Any]]:
        """
        Export run data to a specified format.

        Args:
            run_path: Path to the run directory
            output_format: Format to export data to (json, csv, etc.)

        Returns:
            Tuple containing:
            - Success flag
            - Message (error or success)
            - Exported data (if successful)
        """
        try:
            run_data = load_run_data(run_path)

            if "error" in run_data:
                return False, f"Error loading run data: {run_data['error']}", None

            if output_format == "json":
                import json
                return True, "Run data exported successfully", json.dumps(run_data, indent=2)

            elif output_format == "csv":
                import pandas as pd
                import io

                # Convert tests to a DataFrame
                tests_df = pd.DataFrame([{
                    'Name': t.get('name', 'Unknown'),
                    'Status': t.get('status', 'Unknown'),
                    'Valid': t.get('valid_total', '0/0'),
                    'Iterations': t.get('iterations', 0),
                    'Avg Time': t.get('avg_time', 0)
                } for t in run_data.get('tests', [])])

                buffer = io.StringIO()
                tests_df.to_csv(buffer, index=False)
                return True, "Run data exported as CSV", buffer.getvalue()

            else:
                return False, f"Unsupported export format: {output_format}", None

        except Exception as e:
            logger.error(f"Error exporting run data: {str(e)}")
            return False, f"Error exporting run data: {str(e)}", None
