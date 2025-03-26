"""Enhanced script for preprocessing ARC tasks."""

import os
import logging
import pickle
from pathlib import Path
import time
from tqdm import tqdm

import numpy as np

from config import CACHE_DIR, TRAINING_DATA_DIR
from data_loader import iter_tasks, convert_to_numpy
from data_structures import ProcessedTask, ProcessedPair, Shape
from object_extractor import extract_shapes, extract_and_print_shapes, print_grid_details
from gemini_interaction import get_shape_description, load_description_cache, save_description_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_task(task_id: str, task_data: dict, verbose: bool = True) -> ProcessedTask:
    """
    Process a single ARC task with enhanced logging and debugging.
    
    Args:
        task_id: Task identifier
        task_data: Raw task data
        verbose: Whether to print detailed grid information
        
    Returns:
        ProcessedTask object
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Processing Task: {task_id}")
        print(f"{'='*50}")
        print(f"Training Pairs: {len(task_data['train'])}")
        print(f"Test Pairs: {len(task_data['test'])}")
        print(f"{'='*50}")

    processed_task = ProcessedTask(task_id=task_id)
    
    # Process training pairs
    for train_idx, train_pair in enumerate(task_data['train']):
        # Convert grids to NumPy arrays, preserving zero values
        input_grid = convert_to_numpy(train_pair['input'])
        output_grid = convert_to_numpy(train_pair['output'])
        
        if verbose:
            print(f"\nTraining Pair {train_idx + 1}:")
            print("Input Grid:")
            print_grid_details(input_grid)
            print("\nOutput Grid:")
            print_grid_details(output_grid)
        
        # Create ProcessedPair
        processed_pair = ProcessedPair(
            input_grid=input_grid,
            output_grid=output_grid
        )
        
        # Extract shapes from input and output grids
        input_shapes = extract_and_print_shapes(input_grid) if verbose else extract_shapes(input_grid)
        output_shapes = extract_and_print_shapes(output_grid) if verbose else extract_shapes(output_grid)
        
        # Add descriptions to shapes
        for shape in input_shapes + output_shapes:
            shape.description = get_shape_description(shape)
        
        processed_pair.input_shapes = input_shapes
        processed_pair.output_shapes = output_shapes
        
        # Add to processed task
        processed_task.train_pairs.append(processed_pair)
    
    # Process test pairs
    for test_idx, test_pair in enumerate(task_data['test']):
        # Convert input grid to NumPy array
        input_grid = convert_to_numpy(test_pair['input'])
        
        if verbose:
            print(f"\nTest Pair {test_idx + 1}:")
            print("Input Grid:")
            print_grid_details(input_grid)
        
        # Create ProcessedPair
        processed_pair = ProcessedPair(
            input_grid=input_grid,
            output_grid=np.zeros_like(input_grid)  # Placeholder
        )
        
        # Extract and potentially print input shapes
        input_shapes = extract_and_print_shapes(input_grid) if verbose else extract_shapes(input_grid)
        
        # Add descriptions to input shapes
        for shape in input_shapes:
            shape.description = get_shape_description(shape)
        
        processed_pair.input_shapes = input_shapes
        
        # If output is available in the test pair (for known tasks), process it
        if 'output' in test_pair:
            output_grid = convert_to_numpy(test_pair['output'])
            
            if verbose:
                print("\nTest Output Grid:")
                print_grid_details(output_grid)
            
            processed_pair.output_grid = output_grid
            output_shapes = extract_and_print_shapes(output_grid) if verbose else extract_shapes(output_grid)
            
            # Add descriptions to output shapes
            for shape in output_shapes:
                shape.description = get_shape_description(shape)
            
            processed_pair.output_shapes = output_shapes
        
        # Add to processed task
        processed_task.test_pairs.append(processed_pair)
    
    return processed_task


def preprocess_all_tasks(force_reprocess: bool = False, verbose: bool = False):
    """
    Process all ARC tasks and save them to disk with detailed logging.
    
    Args:
        force_reprocess: If True, reprocess all tasks even if they exist in cache
        verbose: If True, print detailed information about each task
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load description cache
    load_description_cache()
    
    # Get list of all tasks
    task_count = 0
    for task_id, task_data in iter_tasks():
        task_count += 1
    
    logger.info(f"Found {task_count} tasks to process")
    
    # Process each task
    processed_count = 0
    
    for task_id, task_data in tqdm(iter_tasks(), total=task_count, desc="Processing tasks"):
        # Check if task is already processed
        task_cache_file = Path(CACHE_DIR) / f"{task_id}.pkl"
        
        if not force_reprocess and task_cache_file.exists():
            logger.debug(f"Task {task_id} already processed, skipping")
            processed_count += 1
            continue
        
        try:
            # Process the task with optional verbose logging
            processed_task = process_task(task_id, task_data, verbose=verbose)
            
            # Save to disk
            with open(task_cache_file, 'wb') as f:
                pickle.dump(processed_task, f)
            
            processed_count += 1
            
            # Log progress
            if processed_count % 10 == 0:
                logger.info(f"Processed {processed_count}/{task_count} tasks")
                
            # Save description cache periodically
            if processed_count % 5 == 0:
                save_description_cache()
                
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
    
    # Save description cache one final time
    save_description_cache()
    
    logger.info(f"Processed {processed_count}/{task_count} tasks successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess ARC tasks")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all tasks")
    parser.add_argument("--verbose", action="store_true", help="Print detailed task information")
    args = parser.parse_args()
    
    preprocess_all_tasks(force_reprocess=args.force, verbose=args.verbose)
