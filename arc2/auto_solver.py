"""
Automated solver that processes all tasks in a directory.
Tracks progress and continues from the last unsolved example.
"""

import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add colorful terminal output
from colorama import init, Fore, Style, Back

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_solver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hardcoded directories
ARC_DATA_DIR = Path("data/arc")  # Directory containing .json task files
SOLVER_DIR = Path(".solver")  # Directory to track progress
RESULTS_DIR = Path("results")  # Directory to store results

# Import the solver modules
from config import TRAINING_DATA_DIR, CACHE_DIR
from data_loader import load_task
from preprocess_data import process_task
from solver import solve_task
from evaluate import evaluate_task


class TerminalOutput:
    """Helper class for formatted terminal output."""

    @staticmethod
    def header(message):
        """Display a header message"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
        print(f"{message.center(50)}")
        print(f"{'='*50}{Style.RESET_ALL}")

    @staticmethod
    def success(message):
        """Display a success message"""
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ {message}{Style.RESET_ALL}")

    @staticmethod
    def error(message):
        """Display an error message"""
        print(f"{Fore.RED}{Style.BRIGHT}✗ {message}{Style.RESET_ALL}")

    @staticmethod
    def info(message):
        """Display an info message"""
        print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

    @staticmethod
    def warning(message):
        """Display a warning message"""
        print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")

    @staticmethod
    def progress_bar(current, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            current    - Required  : current iteration (Int)
            total      - Required  : total iterations (Int)
            prefix     - Optional  : prefix string (Str)
            suffix     - Optional  : suffix string (Str)
            decimals   - Optional  : positive number of decimals in percent complete (Int)
            length     - Optional  : character length of bar (Int)
            fill       - Optional  : bar fill character (Str)
            print_end  - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{Fore.GREEN}{prefix} |{bar}| {percent}% {suffix}', end=print_end, flush=True)
        if current == total:
            print()


class AutoSolver:
    """
    Automated solver that processes all ARC tasks in a directory.
    Tracks progress and continues from the last unsolved example.
    """

    def __init__(self, data_dir: Path = ARC_DATA_DIR, solver_dir: Path = SOLVER_DIR):
        """
        Initialize the AutoSolver.

        Args:
            data_dir: Directory containing ARC task files
            solver_dir: Directory to track progress
        """
        # Display header
        TerminalOutput.header("ARC TASK AUTO SOLVER")

        self.data_dir = data_dir
        self.solver_dir = solver_dir
        self.progress_file = solver_dir / "progress.json"
        self.processed_dir = solver_dir / "processed"
        self.solved_dir = solver_dir / "solved"
        self.failed_dir = solver_dir / "failed"

        # Ensure directories exist
        for dir_path in [self.solver_dir, self.processed_dir,
                         self.solved_dir, self.failed_dir,
                         TRAINING_DATA_DIR, CACHE_DIR, RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            TerminalOutput.info(f"Created directory: {dir_path}")

        # Initialize progress tracking
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """
        Load progress tracking information.

        Returns:
            Dictionary with progress information
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    TerminalOutput.success(f"Loaded previous progress from {self.progress_file}")
                    return progress
            except Exception as e:
                TerminalOutput.error(f"Error loading progress file: {e}")

        # Initialize with default values if file doesn't exist or has errors
        default_progress = {
            "processed_tasks": [],
            "solved_tasks": [],
            "failed_tasks": [],
            "last_task": None,
            "total_tasks": 0,
            "total_processed": 0,
            "total_solved": 0,
            "success_rate": 0.0,
            "start_time": time.time(),
            "last_update_time": time.time()
        }
        TerminalOutput.warning("Creating new progress tracking")
        return default_progress

    def _save_progress(self):
        """Save progress tracking information."""
        self.progress["last_update_time"] = time.time()

        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_task_files(self) -> List[Path]:
        """
        Get all task files from the data directory.

        Returns:
            List of Path objects for task files
        """
        try:
            task_files = list(self.data_dir.glob("*.json"))
            TerminalOutput.info(f"Found {len(task_files)} task files in {self.data_dir}")
            return task_files
        except Exception as e:
            TerminalOutput.error(f"Error finding task files: {e}")
            return []

    def process_task(self, task_file: Path) -> bool:
        """
        Process a single task.

        Args:
            task_file: Path to the task file

        Returns:
            True if task was processed successfully, False otherwise
        """
        task_id = task_file.stem
        TerminalOutput.info(f"Processing task {Fore.YELLOW}{task_id}{Style.RESET_ALL}")

        try:
            # Detailed processing steps with colorful output
            TerminalOutput.info(f"1. Copying task to training directory")
            dest_file = Path(TRAINING_DATA_DIR) / task_file.name
            if not dest_file.exists():
                shutil.copy2(task_file, dest_file)

            TerminalOutput.info(f"2. Loading and preprocessing task")
            task_data = load_task(dest_file)
            if not task_data:
                TerminalOutput.error(f"Failed to load task {task_id}")
                return False

            processed_task = process_task(task_id, task_data)

            # Save to cache
            TerminalOutput.info(f"3. Saving processed task to cache")
            cache_file = Path(CACHE_DIR) / f"{task_id}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_task, f)

            # Solve the task
            TerminalOutput.info(f"4. Solving task")
            test_idx = 0  # Use first test pair
            solve_task(task_id, test_idx)

            # Evaluate the solution
            TerminalOutput.info(f"5. Evaluating solution")
            evaluation = evaluate_task(task_id, test_idx)

            # Update progress
            self.progress["processed_tasks"].append(task_id)
            self.progress["last_task"] = task_id
            self.progress["total_processed"] += 1

            # Colorful result reporting
            if evaluation.get('success', False):
                self.progress["solved_tasks"].append(task_id)
                self.progress["total_solved"] += 1
                shutil.copy2(task_file, self.solved_dir / task_file.name)
                TerminalOutput.success(f"Task {task_id} solved successfully!")
            else:
                self.progress["failed_tasks"].append(task_id)
                shutil.copy2(task_file, self.failed_dir / task_file.name)
                TerminalOutput.error(f"Task {task_id} solution failed")

            # Update overall success rate
            total_processed = self.progress["total_processed"]
            if total_processed > 0:
                self.progress["success_rate"] = (self.progress["total_solved"] / total_processed) * 100

            # Save progress
            self._save_progress()

            # Copy to processed directory to mark as done
            shutil.copy2(task_file, self.processed_dir / task_file.name)

            return True

        except Exception as e:
            TerminalOutput.error(f"Error processing task {task_id}: {e}")
            return False

    def run(self, max_tasks: int = None, retry_failed: bool = False):
        """
        Run the automated solver.

        Args:
            max_tasks: Maximum number of tasks to process (None for all)
            retry_failed: If True, retry previously failed tasks
        """
        # Fancy start banner
        TerminalOutput.header("🤖 STARTING AUTO SOLVER 🤖")

        # Get all task files
        all_task_files = self.get_task_files()

        if not all_task_files:
            TerminalOutput.error("No task files found")
            return

        # Update total tasks
        self.progress["total_tasks"] = len(all_task_files)
        self._save_progress()

        # Process tasks
        tasks_processed = 0

        # Create progress bar
        TerminalOutput.info(f"Total tasks to process: {len(all_task_files)}")

        while True:
            # Check if we've reached the maximum number of tasks
            if max_tasks is not None and tasks_processed >= max_tasks:
                TerminalOutput.warning(f"Reached maximum number of tasks to process ({max_tasks})")
                break

            # Compute available tasks
            unprocessed_tasks = [
                task for task in all_task_files
                if task.stem not in self.progress["processed_tasks"]
            ]

            if not unprocessed_tasks:
                if retry_failed and self.progress["failed_tasks"]:
                    TerminalOutput.warning("Attempting to retry failed tasks")
                    # Retry failed tasks
                    for task_id in self.progress["failed_tasks"]:
                        task_file = self.data_dir / f"{task_id}.json"
                        if task_file.exists():
                            TerminalOutput.info(f"Retrying failed task {task_id}")
                            # Remove from processed and failed lists to retry
                            if task_id in self.progress["processed_tasks"]:
                                self.progress["processed_tasks"].remove(task_id)
                            if task_id in self.progress["failed_tasks"]:
                                self.progress["failed_tasks"].remove(task_id)
                            self._save_progress()

                            if self.process_task(task_file):
                                tasks_processed += 1

                            if max_tasks is not None and tasks_processed >= max_tasks:
                                break
                    continue
                else:
                    TerminalOutput.success("All tasks processed successfully!")
                    break

            # Select next task
            next_task = unprocessed_tasks[0]

            # Display progress
            TerminalOutput.progress_bar(
                tasks_processed,
                len(all_task_files),
                prefix=f'{Fore.GREEN}Processing Tasks{Style.RESET_ALL}',
                suffix=f'{Fore.CYAN}Complete{Style.RESET_ALL}'
            )

            # Process the task
            if self.process_task(next_task):
                tasks_processed += 1

            # Brief pause to avoid overloading
            time.sleep(1)

        # Final summary with colorful statistics
        TerminalOutput.header("SOLVER SUMMARY")
        print(f"{Fore.CYAN}Total Tasks:{Style.RESET_ALL} {self.progress['total_tasks']}")
        print(f"{Fore.GREEN}Processed Tasks:{Style.RESET_ALL} {self.progress['total_processed']}")
        print(f"{Fore.BLUE}Solved Tasks:{Style.RESET_ALL} {self.progress['total_solved']}")
        print(f"{Fore.YELLOW}Success Rate:{Style.RESET_ALL} {self.progress['success_rate']:.2f}%")

        # Calculate and log elapsed time with fancy formatting
        elapsed_time = time.time() - self.progress["start_time"]
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{Fore.MAGENTA}Total Time:{Style.RESET_ALL} {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated ARC Solver")
    parser.add_argument("--dir", type=str, default=str(ARC_DATA_DIR), help="Directory containing ARC task files")
    parser.add_argument("--max", type=int, help="Maximum number of tasks to process")
    parser.add_argument("--retry", action="store_true", help="Retry failed tasks")
    args = parser.parse_args()

    solver = AutoSolver(data_dir=Path(args.dir))
    solver.run(max_tasks=args.max, retry_failed=args.retry)