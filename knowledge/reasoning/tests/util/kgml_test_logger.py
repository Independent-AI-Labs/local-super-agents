import datetime
import json
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

        # Load existing metadata if it exists
        metadata_file = iter_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is corrupt, start with new metadata
                metadata = {}
        else:
            metadata = {}

        # Write request if provided
        if request:
            with open(iter_dir / "request.kgml", "w", encoding="utf-8") as f:
                f.write(request)
            # Only update response_time when we have a real request (not an update)
            metadata["response_time"] = response_time

        # Write response if provided
        if response:
            with open(iter_dir / "response.kgml", "w", encoding="utf-8") as f:
                f.write(response)

        # Update metadata from parameters - don't overwrite response_time if it's just an update
        metadata.update({
            "is_valid": is_valid,
            "has_syntax_errors": has_syntax_errors,
            "timestamp": time.time()
        })

        # Add execution information if provided
        if execution_result:
            # Create or update execution_result.txt
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

            # Add execution information to metadata
            metadata["execution_success"] = execution_result.get("success", False)
            metadata["execution_time"] = execution_result.get("execution_time", 0.0)

        # Write updated metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update test data structure
        # Ensure we have enough entries in the iterations list
        while len(self.test_data[test_name]["iterations"]) < iteration:
            self.test_data[test_name]["iterations"].append(None)

        # Preserve existing data we might have stored
        existing_data = {}
        if len(self.test_data[test_name]["iterations"]) >= iteration:
            if self.test_data[test_name]["iterations"][iteration - 1] is not None:
                existing_data = self.test_data[test_name]["iterations"][iteration - 1]

        # Update with new data, preserving existing values if not provided in this update
        iteration_data = existing_data.copy()

        # Don't overwrite response_time with empty updates
        if request:
            iteration_data["response_time"] = response_time

        # Update other fields
        iteration_data.update({
            "is_valid": is_valid,
            "has_syntax_errors": has_syntax_errors,
        })

        # Add execution data if provided
        if execution_result:
            iteration_data["execution_success"] = execution_result.get("success", False)
            iteration_data["execution_time"] = execution_result.get("execution_time", 0.0)

        # Store the updated data
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
                    resp_time = f"{iter_data.get('response_time', 0.0):.2f}s"

                    exec_status = ""
                    exec_time = ""
                    if iter_data.get("execution_success") is not None:
                        exec_status = "Execution: " + ("✓" if iter_data["execution_success"] else "✗")
                        if "execution_time" in iter_data:
                            exec_time = f" ({iter_data['execution_time']:.2f}s)"

                    f.write(f"  {i}: Response Valid: {status} ({resp_time}) {exec_status}{exec_time}\n")

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
