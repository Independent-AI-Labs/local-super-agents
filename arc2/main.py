"""Main entry point for the ARC solver application."""

import os
import logging
import argparse
from pathlib import Path

from config import TRAINING_DATA_DIR, RESULTS_DIR, CACHE_DIR
from preprocess_data import preprocess_all_tasks
from solver import solve_task, solve_all_tasks
from evaluate import evaluate_task, evaluate_all_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("arc_solver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Ensure all required directories exist."""
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def run_pipeline(task_id=None, test_idx=0, skip_preprocess=False, skip_solve=False, skip_evaluate=False):
    """
    Run the complete ARC solver pipeline.
    
    Args:
        task_id: Optional task ID to process a single task
        test_idx: Test pair index to solve (default: 0)
        skip_preprocess: Skip preprocessing step
        skip_solve: Skip solving step
        skip_evaluate: Skip evaluation step
    """
    setup_directories()
    
    # Step 1: Preprocess
    if not skip_preprocess:
        logger.info("Starting preprocessing step")
        if task_id:
            logger.info(f"Preprocessing single task: {task_id}")
            # For single task, we can directly load and process
            from data_loader import load_task
            from preprocess_data import process_task
            
            task_path = Path(TRAINING_DATA_DIR) / f"{task_id}.json"
            if task_path.exists():
                task_data = load_task(task_path)
                if task_data:
                    processed_task = process_task(task_id, task_data)
                    
                    # Save to disk
                    import pickle
                    task_cache_file = Path(CACHE_DIR) / f"{task_id}.pkl"
                    with open(task_cache_file, 'wb') as f:
                        pickle.dump(processed_task, f)
                    
                    logger.info(f"Processed task {task_id}")
                else:
                    logger.error(f"Failed to load task {task_id}")
            else:
                logger.error(f"Task file not found: {task_path}")
        else:
            logger.info("Preprocessing all tasks")
            preprocess_all_tasks()
    
    # Step 2: Solve
    if not skip_solve:
        logger.info("Starting solving step")
        if task_id:
            logger.info(f"Solving single task: {task_id}")
            solve_task(task_id, test_idx)
        else:
            logger.info("Solving all tasks")
            solve_all_tasks()
    
    # Step 3: Evaluate
    if not skip_evaluate:
        logger.info("Starting evaluation step")
        if task_id:
            logger.info(f"Evaluating single task: {task_id}")
            result = evaluate_task(task_id, test_idx)
            success = result.get('success', False)
            logger.info(f"Task {task_id} evaluation result: {'SUCCESS' if success else 'FAILURE'}")
        else:
            logger.info("Evaluating all tasks")
            evaluations = evaluate_all_tasks()
            
            # Print summary
            total_tasks = 0
            successful_tasks = 0
            
            for task_id, task_evals in evaluations.items():
                for test_idx, eval_data in task_evals.items():
                    total_tasks += 1
                    if eval_data.get('success', False):
                        successful_tasks += 1
            
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
            logger.info(f"Overall success rate: {success_rate:.2%} ({successful_tasks}/{total_tasks})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC Solver Pipeline")
    
    parser.add_argument("--task", type=str, help="Process a specific task ID")
    parser.add_argument("--test-idx", type=int, default=0, help="Test pair index to solve")
    
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip-solve", action="store_true", help="Skip solving step")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation step")
    
    parser.add_argument("--preprocess-only", action="store_true", help="Only run preprocessing")
    parser.add_argument("--solve-only", action="store_true", help="Only run solving")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Handle "only" flags
    if args.preprocess_only:
        args.skip_solve = True
        args.skip_evaluate = True
    
    if args.solve_only:
        args.skip_preprocess = True
        args.skip_evaluate = True
    
    if args.evaluate_only:
        args.skip_preprocess = True
        args.skip_solve = True
    
    # Run the pipeline
    run_pipeline(
        task_id=args.task,
        test_idx=args.test_idx,
        skip_preprocess=args.skip_preprocess,
        skip_solve=args.skip_solve,
        skip_evaluate=args.skip_evaluate
    )
