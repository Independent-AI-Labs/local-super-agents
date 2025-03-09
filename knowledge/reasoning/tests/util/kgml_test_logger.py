import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLTestLogger")


class KGMLTestLogger:
    """
    Handles logging of KGML test execution, request-response pairs, and detailed statistics.
    Organizes logs in a directory structure by test name and maintains comprehensive stats.
    """

    def __init__(self, base_dir: str = "kgml_test_logs", model_name: str = "unknown"):
        """
        Initialize the test logger with a base directory.

        Args:
            base_dir: Base directory for all test logs
            model_name: Name of the model being tested
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        # Create a unique run identifier based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

        # Initialize stats file
        self.stats_file = self.run_dir / "stats.json"
        self.stats = {
            "run_id": self.run_id,
            "model_name": model_name,  # Add model name to stats
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_tests": 0,
            "total_prompts": 0,
            "total_responses": 0,
            "valid_responses": 0,
            "invalid_responses": 0,
            "syntax_errors": 0,
            "execution_errors": 0,
            "response_times": [],
            "avg_response_time": None,
            "max_response_time": None,
            "min_response_time": None,
            "tests": {}
        }
        self._save_stats()

        # Create a run log file
        self.log_file = self.run_dir / "run.log"
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(self.file_handler)

        logger.info(f"Test run {self.run_id} initialized in {self.run_dir} for model {model_name}")

    def start_test(self, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new test and create its directory structure.

        Args:
            test_name: Name of the test
            metadata: Additional test metadata

        Returns:
            Path to the test directory
        """
        # Create a sanitized test directory name
        safe_name = "".join(c if c.isalnum() else "_" for c in test_name)
        test_dir = self.run_dir / safe_name
        test_dir.mkdir(exist_ok=True)

        # Initialize test stats with safe default values
        test_stats = {
            "name": test_name,
            "metadata": metadata or {},
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "iterations": 0,
            "prompts": 0,
            "responses": 0,
            "valid_responses": 0,
            "invalid_responses": 0,
            "syntax_errors": 0,
            "execution_errors": 0,
            "response_times": [],
            "avg_response_time": None,
            "goal_reached": None,
            "iterations_to_goal": None,
        }

        self.stats["tests"][test_name] = test_stats
        self.stats["total_tests"] += 1
        self._save_stats()

        logger.info(f"Started test: {test_name}")
        return str(test_dir)

    def log_request_response(self,
                             test_name: str,
                             iteration: int,
                             request: str,
                             response: str,
                             response_time: float,
                             is_valid: bool,
                             has_syntax_errors: bool,
                             execution_result: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a request-response pair for a test iteration.

        Args:
            test_name: Name of the test
            iteration: Iteration number (1-based)
            request: The request prompt sent to the model
            response: The response received from the model
            response_time: Time in seconds for the model to respond
            is_valid: Whether the response is valid KGML
            has_syntax_errors: Whether the response has syntax errors
            execution_result: Result of executing the KGML (if available)
        """
        if test_name not in self.stats["tests"]:
            logger.warning(f"Test {test_name} not found in stats. Starting it now.")
            self.start_test(test_name)

        # Get test directory
        safe_name = "".join(c if c.isalnum() else "_" for c in test_name)
        test_dir = self.run_dir / safe_name

        # Create iteration directory
        iter_dir = test_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)

        # Save request and response to separate files
        if request:  # Only write if not empty
            request_file = iter_dir / "request.kgml"
            with open(request_file, "w", encoding="utf-8") as f:
                f.write(request)

        if response:  # Only write if not empty
            response_file = iter_dir / "response.kgml"
            with open(response_file, "w", encoding="utf-8") as f:
                f.write(response)

        # Save execution result if available
        if execution_result is not None:
            result_file = iter_dir / "execution_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(execution_result, f, indent=2)

        # Save iteration metadata
        metadata = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "response_time_seconds": response_time,
            "is_valid": is_valid,
            "has_syntax_errors": has_syntax_errors,
            "execution_success": execution_result.get("success", False) if execution_result else None,
        }
        meta_file = iter_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update test stats
        test_stats = self.stats["tests"][test_name]

        # Only increment counters if we're actually logging a new response
        # (not just updating an existing one)
        if request and response:
            test_stats["iterations"] = max(test_stats["iterations"], iteration)
            test_stats["prompts"] += 1
            test_stats["responses"] += 1

            if response_time > 0:  # Only track valid response times
                test_stats.setdefault("response_times", []).append(response_time)
                response_times = test_stats.get("response_times", [])
                if response_times:
                    test_stats["avg_response_time"] = sum(response_times) / len(response_times)

            if is_valid:
                test_stats["valid_responses"] = test_stats.get("valid_responses", 0) + 1
            else:
                test_stats["invalid_responses"] = test_stats.get("invalid_responses", 0) + 1

            if has_syntax_errors:
                test_stats["syntax_errors"] = test_stats.get("syntax_errors", 0) + 1

            if execution_result and not execution_result.get("success", False):
                test_stats["execution_errors"] = test_stats.get("execution_errors", 0) + 1

            # Update global stats
            self.stats["total_prompts"] += 1
            self.stats["total_responses"] += 1

            if response_time > 0:  # Only track valid response times
                self.stats.setdefault("response_times", []).append(response_time)
                all_response_times = self.stats.get("response_times", [])
                if all_response_times:
                    self.stats["avg_response_time"] = sum(all_response_times) / len(all_response_times)
                    self.stats["max_response_time"] = max(all_response_times)
                    self.stats["min_response_time"] = min(all_response_times)

            if is_valid:
                self.stats["valid_responses"] = self.stats.get("valid_responses", 0) + 1
            else:
                self.stats["invalid_responses"] = self.stats.get("invalid_responses", 0) + 1

            if has_syntax_errors:
                self.stats["syntax_errors"] = self.stats.get("syntax_errors", 0) + 1

            if execution_result and not execution_result.get("success", False):
                self.stats["execution_errors"] = self.stats.get("execution_errors", 0) + 1

        self._save_stats()
        logger.info(f"Logged iteration {iteration} for test {test_name}")

    def end_test(self, test_name: str, goal_reached: Optional[bool] = None, iterations_to_goal: Optional[int] = None) -> None:
        """
        Mark a test as complete and update its final statistics.

        Args:
            test_name: Name of the test
            goal_reached: Whether the test reached its goal
            iterations_to_goal: Number of iterations it took to reach the goal
        """
        if test_name not in self.stats["tests"]:
            logger.warning(f"Cannot end test {test_name}: not found in stats")
            return

        test_stats = self.stats["tests"][test_name]
        test_stats["end_time"] = datetime.now().isoformat()

        # Only update these fields if they're provided
        if goal_reached is not None:
            test_stats["goal_reached"] = goal_reached

        if iterations_to_goal is not None:
            test_stats["iterations_to_goal"] = iterations_to_goal

        # Generate test summary
        safe_name = "".join(c if c.isalnum() else "_" for c in test_name)
        test_dir = self.run_dir / safe_name

        # Make sure the directory exists
        if not test_dir.exists():
            test_dir.mkdir(exist_ok=True, parents=True)

        summary_file = test_dir / "summary.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(test_stats, f, indent=2)

        self._save_stats()
        goal_status = "reached" if goal_reached else "failed" if goal_reached is not None else "unknown"
        logger.info(f"Ended test: {test_name}, goal {goal_status}")

    def end_run(self) -> Dict[str, Any]:
        """
        Mark the test run as complete and finalize statistics.

        Returns:
            The final run statistics
        """
        self.stats["end_time"] = datetime.now().isoformat()
        self._save_stats()

        # Generate an overall summary report
        summary_report = self._generate_summary_report()
        summary_file = self.run_dir / "summary_report.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_report)

        logger.info(f"Test run {self.run_id} completed")
        return self.stats

    def _save_stats(self) -> None:
        """Save the current stats to the stats file."""
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def _generate_summary_report(self) -> str:
        """Generate a human-readable summary report of the test run."""
        now = datetime.now()
        started = datetime.fromisoformat(self.stats["start_time"])
        duration = now - started

        lines = [
            f"KGML Test Run Summary Report",
            f"==========================",
            f"",
            f"Run ID: {self.run_id}",
            f"Model: {self.stats.get('model_name', 'unknown')}",  # Add model name to the report
            f"Started: {self.stats['start_time']}",
            f"Ended: {self.stats.get('end_time') or now.isoformat()}",
            f"Duration: {duration}",
            f"",
            f"Overall Statistics",
            f"-----------------",
            f"Total Tests: {self.stats['total_tests']}",
            f"Total Prompts: {self.stats['total_prompts']}",
            f"Total Responses: {self.stats['total_responses']}",
        ]

        # Safely format percentages with null checks
        total_responses = max(1, self.stats.get('total_responses', 1))
        valid_responses = self.stats.get('valid_responses', 0)
        invalid_responses = self.stats.get('invalid_responses', 0)
        syntax_errors = self.stats.get('syntax_errors', 0)
        execution_errors = self.stats.get('execution_errors', 0)

        lines.extend([
            f"Valid Responses: {valid_responses} ({valid_responses / total_responses * 100:.1f}%)",
            f"Invalid Responses: {invalid_responses} ({invalid_responses / total_responses * 100:.1f}%)",
            f"Syntax Errors: {syntax_errors} ({syntax_errors / total_responses * 100:.1f}%)",
            f"Execution Errors: {execution_errors} ({execution_errors / total_responses * 100:.1f}%)",
            f"",
            f"Response Time Statistics",
            f"-----------------------",
        ])

        # Safely format response time statistics with null checks
        avg_response_time = self.stats.get('avg_response_time')
        min_response_time = self.stats.get('min_response_time')
        max_response_time = self.stats.get('max_response_time')

        if avg_response_time is not None:
            lines.append(f"Average Response Time: {avg_response_time:.2f}s")
        else:
            lines.append(f"Average Response Time: N/A")

        if min_response_time is not None:
            lines.append(f"Minimum Response Time: {min_response_time:.2f}s")
        else:
            lines.append(f"Minimum Response Time: N/A")

        if max_response_time is not None:
            lines.append(f"Maximum Response Time: {max_response_time:.2f}s")
        else:
            lines.append(f"Maximum Response Time: N/A")

        lines.extend([
            f"",
            f"Test Results",
            f"------------",
        ])

        for test_name, test_stats in self.stats["tests"].items():
            goal_status = "✅ REACHED" if test_stats.get("goal_reached") else "❌ FAILED" if test_stats.get("goal_reached") is not None else "⚠️ UNKNOWN"
            iterations_to_goal = test_stats.get('iterations_to_goal')
            iterations = f"in {iterations_to_goal} iterations" if iterations_to_goal is not None else ""

            lines.append(f"")
            lines.append(f"Test: {test_name}")
            lines.append(f"  Goal: {goal_status} {iterations}")
            lines.append(f"  Iterations: {test_stats.get('iterations', 0)}")

            # Safely format test percentages with null checks
            test_responses = max(1, test_stats.get('responses', 1))
            test_valid = test_stats.get('valid_responses', 0)
            test_syntax_errors = test_stats.get('syntax_errors', 0)
            test_execution_errors = test_stats.get('execution_errors', 0)

            lines.append(f"  Valid Responses: {test_valid}/{test_stats.get('responses', 0)} ({test_valid / test_responses * 100:.1f}%)")
            lines.append(f"  Syntax Errors: {test_syntax_errors}/{test_stats.get('responses', 0)} ({test_syntax_errors / test_responses * 100:.1f}%)")
            lines.append(f"  Execution Errors: {test_execution_errors}/{test_stats.get('responses', 0)} ({test_execution_errors / test_responses * 100:.1f}%)")

            # Safely format test response time with null check
            test_avg_response_time = test_stats.get('avg_response_time')
            if test_avg_response_time is not None:
                lines.append(f"  Avg. Response Time: {test_avg_response_time:.2f}s")
            else:
                lines.append(f"  Avg. Response Time: N/A")

        return "\n".join(lines)
