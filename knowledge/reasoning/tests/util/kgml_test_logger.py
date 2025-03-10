import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLTestLogger")


class KGMLTestLogger:
    """
    Logger for KGML tests that records interactions and results.
    """

    def __init__(self, base_dir: str, model_name: str):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.run_id_time = time.time()  # Store actual timestamp for ISO formatting
        self.run_dir = self.base_dir / f"run_{self.run_id}_{model_name}".replace(":", "_")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.current_test = None
        self.test_data = {}

        # Create a run info file with ISO-formatted timestamps
        with open(self.run_dir / "run_info.txt", "w") as f:
            start_time_iso = datetime.datetime.fromtimestamp(self.run_id_time).isoformat()
            f.write(f"Test Run: {self.run_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Started: {start_time_iso}\n")

    def start_test(self, test_name: str, metadata: Dict[str, Any]):
        """Start a new test with the given name and metadata."""
        self.current_test = test_name
        self.test_data[test_name] = {
            "metadata": metadata,
            "iterations": [],
            "start_time": time.time(),
            "completed": False
        }

        # Create a test directory
        test_dir = self.run_dir / test_name
        test_dir.mkdir(exist_ok=True)

        # Write metadata
        with open(test_dir / "metadata.txt", "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

    def log_request_response(self, test_name: str, iteration: int, request: str, response: str,
                             response_time: float, is_valid: bool, has_syntax_errors: bool,
                             execution_result: Optional[Dict[str, Any]] = None):
        """Log a request-response pair for a test iteration."""
        if test_name not in self.test_data:
            self.start_test(test_name, {"description": "Auto-created test"})

        # Ensure the test directory exists
        test_dir = self.run_dir / test_name
        test_dir.mkdir(exist_ok=True)

        # Create iteration directory if it doesn't exist
        iter_dir = test_dir / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        # Write request
        if request:
            with open(iter_dir / "request.kgml", "w", encoding="utf-8") as f:
                f.write(request)

        # Write response
        if response:
            with open(iter_dir / "response.kgml", "w", encoding="utf-8") as f:
                f.write(response)

        # Write execution result if provided
        if execution_result:
            with open(iter_dir / "execution_result.txt", "w", encoding="utf-8") as f:
                f.write(f"Success: {execution_result.get('success', False)}\n")
                if 'error' in execution_result:
                    f.write(f"Error: {execution_result['error']}\n")
                if 'execution_time' in execution_result:
                    f.write(f"Execution Time: {execution_result['execution_time']:.4f}s\n")
                if 'execution_log' in execution_result:
                    f.write("\nExecution Log:\n")
                    for log_entry in execution_result['execution_log']:
                        f.write(f"- {log_entry}\n")

        # Save metadata to JSON file
        metadata = {
            "is_valid": is_valid,
            "has_syntax_errors": has_syntax_errors,
            "response_time": response_time,
            "execution_success": execution_result.get("success", None) if execution_result else None,
            "timestamp": time.time()
        }

        with open(iter_dir / "metadata.json", "w", encoding="utf-8") as f:
            import json
            json.dump(metadata, f, indent=2)

        # Update test data
        iteration_data = {
            "response_time": response_time,
            "is_valid": is_valid,
            "has_syntax_errors": has_syntax_errors,
            "execution_success": execution_result.get("success", False) if execution_result else None
        }

        # Add to test data
        while len(self.test_data[test_name]["iterations"]) < iteration:
            self.test_data[test_name]["iterations"].append(None)

        if len(self.test_data[test_name]["iterations"]) == iteration - 1:
            self.test_data[test_name]["iterations"].append(iteration_data)
        else:
            self.test_data[test_name]["iterations"][iteration - 1] = iteration_data

    def end_test(self, test_name: str, goal_reached: bool, iterations_to_goal: Optional[int] = None):
        """Mark a test as completed with results."""
        if test_name not in self.test_data:
            logger.warning(f"Trying to end test {test_name} which was not started")
            return

        self.test_data[test_name]["completed"] = True
        self.test_data[test_name]["end_time"] = time.time()
        self.test_data[test_name]["goal_reached"] = goal_reached
        self.test_data[test_name]["iterations_to_goal"] = iterations_to_goal

        # Write summary
        test_dir = self.run_dir / test_name
        with open(test_dir / "summary.txt", "w", encoding="utf-8") as f:
            duration = self.test_data[test_name]["end_time"] - self.test_data[test_name]["start_time"]
            f.write(f"Test: {test_name}\n")
            f.write(f"Goal Reached: {goal_reached}\n")
            if iterations_to_goal is not None:
                f.write(f"Iterations to Goal: {iterations_to_goal}\n")
            f.write(f"Duration: {duration:.2f}s\n")

            # Add iteration summaries
            f.write("\nIterations:\n")
            for i, iter_data in enumerate(self.test_data[test_name]["iterations"], 1):
                if iter_data:
                    status = "✓" if iter_data.get("is_valid", False) else "✗"
                    exec_status = ""
                    if iter_data.get("execution_success") is not None:
                        exec_status = "Execution: " + ("✓" if iter_data["execution_success"] else "✗")
                    f.write(f"  {i}: Response Valid: {status} {exec_status} ({iter_data['response_time']:.2f}s)\n")

    def end_run(self):
        """Finalize the test run."""
        # Complete any incomplete tests
        for test_name, test_data in self.test_data.items():
            if not test_data.get("completed", False):
                self.end_test(test_name, False)

        # Record end time in ISO format for consistency
        end_time = time.time()
        end_time_iso = datetime.datetime.fromtimestamp(end_time).isoformat()

        # Write overall summary
        with open(self.run_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Test Run: {self.run_id}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Started: {datetime.datetime.fromtimestamp(self.run_id_time).isoformat()}\n")
            f.write(f"Completed: {end_time_iso}\n\n")

            total_tests = len(self.test_data)
            successful_tests = sum(1 for t in self.test_data.values() if t.get("goal_reached", False))
            f.write(f"Tests: {total_tests}\n")
            f.write(f"Successful: {successful_tests} ({successful_tests / total_tests * 100:.1f}%)\n\n")

            f.write("Test Results:\n")
            for test_name, test_data in self.test_data.items():
                status = "✅" if test_data.get("goal_reached", False) else "❌"
                iterations = test_data.get("iterations_to_goal", "N/A")
                f.write(f"  {test_name}: {status} (Iterations: {iterations})\n")
