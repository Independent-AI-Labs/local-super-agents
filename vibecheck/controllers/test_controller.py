"""
Test management controller for VibeKiller.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models.tests import TestResult, TestSuiteResult, TestData
from .. import config


class TestController:
    """
    Controller for test discovery, execution, and results visualization.
    """

    @staticmethod
    def discover_tests(project_path: str) -> List[str]:
        """
        Discover test files in the project.

        Args:
            project_path: Path to the project

        Returns:
            List of discovered test file paths (relative to project_path)
        """
        test_files: List[str] = []
        
        # Common test directories and patterns
        test_dirs = ["tests", "test", "*/tests", "*/test"]
        test_patterns = ["test_*.py", "*_test.py", "test*.py"]
        
        for test_dir in test_dirs:
            dir_path = os.path.join(project_path, test_dir)
            
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                continue
            
            for pattern in test_patterns:
                # Use glob-like pattern matching
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        # Simple pattern matching
                        if TestController._matches_pattern(file, pattern):
                            rel_path = os.path.relpath(os.path.join(root, file), project_path)
                            test_files.append(rel_path)
        
        return test_files

    @staticmethod
    def run_tests(project_path: str, test_paths: List[str]) -> TestSuiteResult:
        """
        Run tests in the project.

        Args:
            project_path: Path to the project
            test_paths: List of test file paths to run (relative to project_path)

        Returns:
            TestSuiteResult with the test results
        """
        if not test_paths:
            return TestSuiteResult(
                name="empty_suite",
                tests=[],
                passed_count=0,
                failed_count=0,
                execution_date=datetime.now()
            )
        
        # Generate a name for the test suite based on the current time
        suite_name = f"test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run the tests
        test_results: List[TestResult] = []
        passed_count = 0
        failed_count = 0
        
        for test_path in test_paths:
            abs_test_path = os.path.join(project_path, test_path)
            if not os.path.exists(abs_test_path):
                continue
            
            # Get test name from the file path
            test_name = os.path.basename(test_path)
            
            # Run the test
            result, output, duration = TestController._execute_test(project_path, test_path)
            
            # Create test result
            test_result = TestResult(
                name=test_name,
                status=result,
                duration=duration,
                output=output
            )
            
            test_results.append(test_result)
            
            if result == "passed":
                passed_count += 1
            elif result in ["failed", "error"]:
                failed_count += 1
        
        # Create test suite result
        suite_result = TestSuiteResult(
            name=suite_name,
            tests=test_results,
            passed_count=passed_count,
            failed_count=failed_count,
            execution_date=datetime.now()
        )
        
        # Save the test suite result
        TestController._save_test_suite_result(project_path, suite_result)
        
        return suite_result

    @staticmethod
    def visualize_test_results(test_suite: TestSuiteResult) -> Dict:
        """
        Generate visualization data for test results.

        Args:
            test_suite: TestSuiteResult to visualize

        Returns:
            Dictionary with visualization data
        """
        # Prepare data for visualization
        visualization_data = {
            "name": test_suite.name,
            "execution_date": test_suite.execution_date.isoformat(),
            "total_tests": len(test_suite.tests),
            "passed_count": test_suite.passed_count,
            "failed_count": test_suite.failed_count,
            "skipped_count": sum(1 for test in test_suite.tests if test.status == "skipped"),
            "pass_percentage": test_suite.passed_count / len(test_suite.tests) * 100 if test_suite.tests else 0,
            "test_results": [
                {
                    "name": test.name,
                    "status": test.status,
                    "duration": test.duration
                }
                for test in test_suite.tests
            ],
            "status_counts": {
                "passed": test_suite.passed_count,
                "failed": test_suite.failed_count,
                "skipped": sum(1 for test in test_suite.tests if test.status == "skipped"),
                "error": sum(1 for test in test_suite.tests if test.status == "error")
            }
        }
        
        return visualization_data

    @staticmethod
    def load_test_results(project_path: str) -> TestData:
        """
        Load test results from the project.

        Args:
            project_path: Path to the project

        Returns:
            TestData with all test results
        """
        results_path = os.path.join(project_path, config.TEST_RESULTS_FILE)
        
        if not os.path.exists(results_path):
            return TestData(test_suites={})
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                return TestData.parse_obj(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading test results: {e}")
            return TestData(test_suites={})

    @staticmethod
    def get_test_suite(project_path: str, suite_name: str) -> Optional[TestSuiteResult]:
        """
        Get a specific test suite result.

        Args:
            project_path: Path to the project
            suite_name: Name of the test suite to get

        Returns:
            TestSuiteResult or None if not found
        """
        test_data = TestController.load_test_results(project_path)
        return test_data.test_suites.get(suite_name)

    @staticmethod
    def _matches_pattern(filename: str, pattern: str) -> bool:
        """
        Check if a filename matches a simple glob-like pattern.

        Args:
            filename: Filename to check
            pattern: Pattern to match (supports * wildcard)

        Returns:
            True if the filename matches the pattern, False otherwise
        """
        if pattern == "*":
            return True
        
        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in filename
        
        if pattern.startswith("*"):
            return filename.endswith(pattern[1:])
        
        if pattern.endswith("*"):
            return filename.startswith(pattern[:-1])
        
        return filename == pattern

    @staticmethod
    def _execute_test(project_path: str, test_path: str) -> Tuple[str, str, float]:
        """
        Execute a test file using pytest.

        Args:
            project_path: Path to the project
            test_path: Path to the test file (relative to project_path)

        Returns:
            Tuple of (result, output, duration)
        """
        abs_test_path = os.path.join(project_path, test_path)
        
        try:
            start_time = time.time()
            
            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", abs_test_path, "-v"],
                cwd=project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            
            # Determine test result
            if result.returncode == 0:
                return "passed", output, duration
            else:
                return "failed", output, duration
            
        except Exception as e:
            return "error", str(e), 0.0

    @staticmethod
    def _save_test_suite_result(project_path: str, suite_result: TestSuiteResult) -> None:
        """
        Save a test suite result to the project.

        Args:
            project_path: Path to the project
            suite_result: TestSuiteResult to save
        """
        # Load existing test data
        test_data = TestController.load_test_results(project_path)
        
        # Add the new test suite result
        test_data.test_suites[suite_result.name] = suite_result
        
        # Ensure the directory exists
        tests_dir = os.path.join(project_path, config.TESTS_DIR)
        os.makedirs(tests_dir, exist_ok=True)
        
        # Save the updated test data
        results_path = os.path.join(project_path, config.TEST_RESULTS_FILE)
        with open(results_path, 'w') as f:
            f.write(test_data.json(indent=2))
