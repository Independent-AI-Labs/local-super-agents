"""Implementation of the ARC task solver using Gemini."""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Add colorful terminal output
from colorama import init, Fore, Style
from google import genai

init(autoreset=True)

from config import CACHE_DIR, RESULTS_DIR, GEMINI_API_KEY, GEMINI_MODEL
from data_structures import ProcessedTask, PredictedShape
from gemini_interaction import (
    generate_solving_prompt,
    parse_solver_response,
    print_task_details
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def load_processed_task(task_id: str) -> Optional[ProcessedTask]:
    """
    Load a processed task from disk.

    Args:
        task_id: Task identifier

    Returns:
        ProcessedTask object or None if not found
    """
    task_cache_file = Path(CACHE_DIR) / f"{task_id}.pkl"

    if not task_cache_file.exists():
        print(f"{Fore.RED}✗ No processed data found for task {task_id}{Style.RESET_ALL}")
        return None

    try:
        with open(task_cache_file, 'rb') as f:
            processed_task = pickle.load(f)
        return processed_task
    except Exception as e:
        print(f"{Fore.RED}✗ Error loading processed task {task_id}: {e}{Style.RESET_ALL}")
        return None


def solve_task_abstracted(prompt: str) -> str:
    """
    Send a prompt to Gemini for solving an ARC task using the abstracted approach.

    Args:
        prompt: The formatted prompt text

    Returns:
        Raw text response from Gemini
    """
    tries = 0
    max_tries = 3

    while tries < max_tries:
        try:
            # Call Gemini API using the updated client-based approach
            model = client.models.get(model=GEMINI_MODEL)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

            return response.text

        except Exception as e:
            tries += 1
            print(f"{Fore.YELLOW}⚠ Error calling Gemini API (try {tries}/{max_tries}): {e}{Style.RESET_ALL}")
            time.sleep(1)  # Brief pause before retrying

    # If all retries failed
    return "Error: Unable to get a response from Gemini API."


def solve_task(task_id: str, test_pair_index: int = 0) -> Tuple[List[PredictedShape], Optional[Tuple[int, int]]]:
    """
        Solve a specific ARC task using the Gemini API.

        Args:
            task_id: Task identifier
            test_pair_index: Index of the test pair to solve (default: 0)

        Returns:
            Tuple of (list of predicted shapes, optional grid dimensions)
        """
    # Load the processed task
    processed_task = load_processed_task(task_id)

    if not processed_task:
        return [], None

    # Get the test pair and its grid dimensions
    test_pair = processed_task.test_pairs[test_pair_index]

    # Determine input grid dimensions (assuming first input shape defines the grid)
    if test_pair.input_shapes:
        first_input_shape = test_pair.input_shapes[0]
        input_grid_width = first_input_shape.subgrid.shape[1]
        input_grid_height = first_input_shape.subgrid.shape[0]
    else:
        print(f"{Fore.RED}✗ No input shapes found for task {task_id}{Style.RESET_ALL}")
        return [], None

    # Generate solving prompt
    prompt = generate_solving_prompt(processed_task, test_pair_index)

    # Send to Gemini
    response = solve_task_abstracted(prompt)

    # Print task details for debugging and transparency
    print_task_details(task_id, prompt, response)

    # Parse the response
    predicted_shapes, grid_dimensions = parse_solver_response(
        response,
        input_grid_width,
        input_grid_height
    )

    # Save the results
    results_dir = Path(RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    result_file = results_dir / f"{task_id}_test{test_pair_index}_result.pkl"

    with open(result_file, 'wb') as f:
        pickle.dump({
            'task_id': task_id,
            'test_pair_index': test_pair_index,
            'prompt': prompt,
            'response': response,
            'predicted_shapes': predicted_shapes,
            'grid_dimensions': grid_dimensions,
            'timestamp': time.time()
        }, f)

    print(f"{Fore.GREEN}✓ Saved results for task {task_id}, test pair {test_pair_index}{Style.RESET_ALL}")

    return predicted_shapes, grid_dimensions


def solve_all_tasks():
    """
    Solve all preprocessed ARC tasks.
    """
    # Get all processed tasks
    task_files = list(Path(CACHE_DIR).glob("*.pkl"))
    print(f"{Fore.BLUE}ℹ Found {len(task_files)} processed tasks{Style.RESET_ALL}")

    results = {}

    # Calculate total progress
    for i, task_file in enumerate(task_files, 1):
        task_id = task_file.stem

        try:
            # Load the processed task
            with open(task_file, 'rb') as f:
                processed_task = pickle.load(f)

            # Progress tracking
            print(f"\n{Fore.CYAN}===== Processing Task {i}/{len(task_files)} ====={Style.RESET_ALL}")
            print(f"{Fore.GREEN}Task ID:{Style.RESET_ALL} {task_id}")

            # For each test pair
            task_results = []
            for test_idx in range(len(processed_task.test_pairs)):
                print(f"\n{Fore.YELLOW}Solving test pair {test_idx}{Style.RESET_ALL}")

                # Solve the task
                predicted_shapes, grid_dimensions = solve_task(task_id, test_idx)

                # Store the result
                task_results.append({
                    'test_idx': test_idx,
                    'predicted_shapes': predicted_shapes,
                    'grid_dimensions': grid_dimensions
                })

                # Brief pause to avoid hitting rate limits
                time.sleep(1)

            # Store results for this task
            results[task_id] = task_results

        except Exception as e:
            print(f"{Fore.RED}✗ Error solving task {task_id}: {e}{Style.RESET_ALL}")

    # Save overall results
    results_file = Path(RESULTS_DIR) / "all_results.pkl"

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{Fore.GREEN}✓ Completed solving {len(results)} tasks{Style.RESET_ALL}")

    return results


if __name__ == "__main__":
    import argparse

    # Setup argument parsing with colored help
    parser = argparse.ArgumentParser(
        description=f"{Fore.CYAN}ARC Task Solver{Style.RESET_ALL}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        help=f"{Fore.YELLOW}Solve a specific task ID{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--test-idx",
        type=int,
        default=0,
        help=f"{Fore.YELLOW}Test pair index to solve (default: 0){Style.RESET_ALL}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"{Fore.GREEN}Solve all tasks{Style.RESET_ALL}"
    )
    args = parser.parse_args()

    # Run based on arguments
    if args.task:
        solve_task(args.task, args.test_idx)
    elif args.all:
        solve_all_tasks()
    else:
        parser.print_help()
