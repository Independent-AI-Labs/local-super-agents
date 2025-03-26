"""Functions for evaluating ARC task predictions with strict matching criteria."""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
import numpy as np
from difflib import SequenceMatcher

from data_structures import ProcessedTask, Shape, PredictedShape
from config import CACHE_DIR, RESULTS_DIR
from solver import load_processed_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def string_similarity(a: str, b: str) -> float:
    """
    Calculate the similarity between two strings using SequenceMatcher.
    Used only for statistical analysis, not for match determination.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, a, b).ratio()


def evaluate_prediction(predicted_shapes: List[PredictedShape], 
                        actual_shapes: List[Shape]) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate a prediction against the actual shapes with strict matching criteria.
    Only exact position matches and perfect shape matches count as successful.
    
    Args:
        predicted_shapes: List of predicted shapes
        actual_shapes: List of actual shapes from the ground truth
        
    Returns:
        Tuple of (success boolean, evaluation details)
    """
    # Check if shape counts match
    if len(predicted_shapes) != len(actual_shapes):
        return False, {
            'reason': f"Shape count mismatch: predicted {len(predicted_shapes)}, actual {len(actual_shapes)}",
            'match_rate': 0.0,
            'similarity_stats': {'avg': 0.0, 'min': 0.0, 'max': 0.0}
        }
    
    # Create sets of positions for quick lookup
    actual_positions = {shape.bottom_left_position for shape in actual_shapes}
    pred_positions = {shape.position for shape in predicted_shapes}
    
    # Check if positions match exactly
    positions_match = actual_positions == pred_positions
    
    if not positions_match:
        return False, {
            'reason': "Position mismatch",
            'match_rate': 0.0,
            'similarity_stats': {'avg': 0.0, 'min': 0.0, 'max': 0.0}
        }
    
    # Create a mapping from position to shapes
    actual_by_pos = {shape.bottom_left_position: shape for shape in actual_shapes}
    pred_by_pos = {shape.position: shape for shape in predicted_shapes}
    
    # Track similarity statistics (for analysis only)
    similarities = []
    position_matches = 0
    
    # Check each position
    for pos in actual_positions:
        actual_shape = actual_by_pos[pos]
        pred_shape = pred_by_pos.get(pos)
        
        if not pred_shape:
            continue
        
        position_matches += 1
        
        # Calculate similarity for statistics
        sim = string_similarity(actual_shape.description, pred_shape.description)
        similarities.append(sim)
    
    # Only success if all positions match exactly
    success = position_matches == len(actual_shapes)
    
    # Calculate similarity statistics
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
    else:
        avg_sim = min_sim = max_sim = 0.0
    
    return success, {
        'success': success,
        'match_rate': position_matches / len(actual_shapes) if actual_shapes else 0,
        'similarity_stats': {
            'avg': avg_sim,
            'min': min_sim,
            'max': max_sim
        }
    }


def evaluate_task(task_id: str, test_pair_index: int = 0) -> Dict[str, Any]:
    """
    Evaluate the prediction for a specific task and test pair.
    
    Args:
        task_id: Task identifier
        test_pair_index: Index of the test pair to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    # Load the processed task
    processed_task = load_processed_task(task_id)
    
    if not processed_task:
        return {'error': f"Processed task {task_id} not found"}
    
    # Check if the test pair exists and has output shapes
    if test_pair_index >= len(processed_task.test_pairs):
        return {'error': f"Test pair index {test_pair_index} out of range"}
    
    test_pair = processed_task.test_pairs[test_pair_index]
    
    if not test_pair.output_shapes:
        return {'error': f"No output shapes available for evaluation"}
    
    # Load the prediction results
    result_file = Path(RESULTS_DIR) / f"{task_id}_test{test_pair_index}_result.pkl"
    
    if not result_file.exists():
        return {'error': f"No prediction results found for task {task_id}, test pair {test_pair_index}"}
    
    try:
        with open(result_file, 'rb') as f:
            prediction_data = pickle.load(f)
        
        predicted_shapes = prediction_data['predicted_shapes']
        
        # Evaluate the prediction
        success, eval_details = evaluate_prediction(
            predicted_shapes,
            test_pair.output_shapes
        )
        
        evaluation = {
            'task_id': task_id,
            'test_pair_index': test_pair_index,
            'success': success,
            'details': eval_details,
            'predicted_shapes': predicted_shapes,
            'actual_shapes': test_pair.output_shapes
        }
        
        # Save the evaluation results
        eval_file = Path(RESULTS_DIR) / f"{task_id}_test{test_pair_index}_eval.pkl"
        
        with open(eval_file, 'wb') as f:
            pickle.dump(evaluation, f)
        
        logger.info(f"Evaluated task {task_id}, test pair {test_pair_index}: {'SUCCESS' if success else 'FAILURE'}")
        
        return evaluation
    
    except Exception as e:
        logger.error(f"Error evaluating task {task_id}, test pair {test_pair_index}: {e}")
        return {'error': str(e)}


def evaluate_all_tasks() -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Evaluate all solved tasks.
    
    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    # Find all result files
    result_files = list(Path(RESULTS_DIR).glob("*_result.pkl"))
    logger.info(f"Found {len(result_files)} result files to evaluate")
    
    evaluations = {}
    
    for result_file in result_files:
        # Extract task ID and test pair index from filename
        filename = result_file.stem
        parts = filename.split('_')
        
        if len(parts) < 3 or not parts[1].startswith('test'):
            logger.warning(f"Skipping file with unexpected format: {filename}")
            continue
        
        task_id = parts[0]
        test_pair_index = int(parts[1].replace('test', ''))
        
        # Evaluate the task
        evaluation = evaluate_task(task_id, test_pair_index)
        
        # Store the result
        if task_id not in evaluations:
            evaluations[task_id] = {}
        
        evaluations[task_id][test_pair_index] = evaluation
    
    # Calculate overall success rate
    total_tasks = 0
    successful_tasks = 0
    
    for task_id, task_evals in evaluations.items():
        for test_idx, eval_data in task_evals.items():
            total_tasks += 1
            if eval_data.get('success', False):
                successful_tasks += 1
    
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
    logger.info(f"Overall success rate: {success_rate:.2%} ({successful_tasks}/{total_tasks})")
    
    # Save overall results
    overall_file = Path(RESULTS_DIR) / "evaluation_summary.pkl"
    
    with open(overall_file, 'wb') as f:
        pickle.dump({
            'evaluations': evaluations,
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': success_rate
        }, f)
    
    return evaluations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ARC task predictions")
    parser.add_argument("--task", type=str, help="Evaluate a specific task ID")
    parser.add_argument("--test-idx", type=int, default=0, help="Test pair index to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all tasks")
    args = parser.parse_args()
    
    if args.task:
        evaluate_task(args.task, args.test_idx)
    elif args.all:
        evaluate_all_tasks()
    else:
        parser.print_help()
