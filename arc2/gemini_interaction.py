"""Functions for interacting with the Google Gemini API."""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from google import genai
import numpy as np

from config import GEMINI_API_KEY, GEMINI_MODEL, CACHE_DIR, USE_DESCRIPTION_CACHE
from grid_utils import grid_to_string, format_grid_ascii, crop_grid_to_shape
from data_structures import Shape, ProcessedTask, ProcessedPair, PredictedShape

# Add colorful terminal output
from colorama import init, Fore, Style, Back
init(autoreset=True)

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini API using the updated client-based approach
client = genai.Client(api_key=GEMINI_API_KEY)

# Cache for shape descriptions
description_cache = {}


def load_description_cache():
    """Load the shape description cache from disk if it exists."""
    global description_cache
    
    cache_file = Path(CACHE_DIR) / "description_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                description_cache = json.load(f)
            print(f"{Fore.GREEN}✓ Loaded {len(description_cache)} cached descriptions{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Error loading description cache: {e}{Style.RESET_ALL}")
            description_cache = {}


def save_description_cache():
    """Save the shape description cache to disk."""
    cache_file = Path(CACHE_DIR) / "description_cache.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(description_cache, f)
        print(f"{Fore.GREEN}✓ Saved {len(description_cache)} descriptions to cache{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}✗ Error saving description cache: {e}{Style.RESET_ALL}")


def get_shape_description(shape: Shape) -> str:
    """
    Get a detailed description of a shape using Gemini API.
    
    Args:
        shape: Shape object to describe
        
    Returns:
        String description of the shape
    """
    # Crop the grid to the actual shape
    cropped_grid = crop_grid_to_shape(shape.subgrid)
    
    # Convert shape subgrid to string for use as cache key
    grid_str = str(cropped_grid.tolist())
    
    # Check cache if enabled
    if USE_DESCRIPTION_CACHE and grid_str in description_cache:
        return description_cache[grid_str]
    
    # Convert grid to a readable ASCII representation
    grid_text = format_grid_ascii(cropped_grid)
    
    # Construct the prompt
    prompt = f"""
Provide a detailed description of the following shape:

{grid_text}

Description should include:
1. Geometric properties (shape, orientation, symmetry)
2. Grid details (non-zero cell locations, patterns)
3. Unique characteristics
4. Size and dimensions

Format your response as:
- Concise but comprehensive description
- Avoiding overly technical language
- Focusing on visual and structural properties
    """
    
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
        try:
            # Call Gemini API using the updated client-based approach
            model = client.models.get(model=GEMINI_MODEL)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            
            # Extract and clean description
            description = response.text.strip()
            
            # Cache the result if enabled
            if USE_DESCRIPTION_CACHE:
                description_cache[grid_str] = description
                # Periodically save the cache
                if len(description_cache) % 10 == 0:
                    save_description_cache()
            
            return description
        
        except Exception as e:
            tries += 1
            print(f"{Fore.YELLOW}⚠ Error calling Gemini API (try {tries}/{max_tries}): {e}{Style.RESET_ALL}")
            time.sleep(1)  # Brief pause before retrying
    
    # If all retries failed
    return f"A shape with unique grid pattern: {grid_text}"


def generate_solving_prompt(processed_task: ProcessedTask, test_pair_index: int) -> str:
    """
    Generate a comprehensive prompt for solving an ARC task.
    
    Args:
        processed_task: The processed task
        test_pair_index: Index of the test pair to solve
        
    Returns:
        Formatted prompt string
    """
    # Get the test pair
    test_pair = processed_task.test_pairs[test_pair_index]
    
    # Start building the prompt
    prompt_parts = [f"ARC TASK: {processed_task.task_id}\n"]
    
    # Add training examples
    prompt_parts.append("TRAINING EXAMPLES:\n")
    for i, train_pair in enumerate(processed_task.train_pairs):
        prompt_parts.append(f"Example {i+1}:\n")
        
        # Input shapes
        prompt_parts.append("INPUT SHAPES:\n")
        for j, shape in enumerate(train_pair.input_shapes):
            shape_grid = format_grid_ascii(crop_grid_to_shape(shape.subgrid))
            prompt_parts.append(f"SHAPE {j+1}\n")
            prompt_parts.append(f"POS: {shape.bottom_left_position[0]},{shape.bottom_left_position[1]}\n")
            prompt_parts.append(f"{shape_grid}\n")
            prompt_parts.append(f"DESCRIPTION: {shape.description}\n\n")
        
        # Output shapes
        prompt_parts.append("OUTPUT SHAPES:\n")
        for j, shape in enumerate(train_pair.output_shapes):
            shape_grid = format_grid_ascii(crop_grid_to_shape(shape.subgrid))
            prompt_parts.append(f"SHAPE {j+1}\n")
            prompt_parts.append(f"POS: {shape.bottom_left_position[0]},{shape.bottom_left_position[1]}\n")
            prompt_parts.append(f"{shape_grid}\n")
            prompt_parts.append(f"DESCRIPTION: {shape.description}\n\n")
        
        prompt_parts.append("\n")
    
    # Add test input
    prompt_parts.append("TEST INPUT:\n")
    for j, shape in enumerate(test_pair.input_shapes):
        shape_grid = format_grid_ascii(crop_grid_to_shape(shape.subgrid))
        prompt_parts.append(f"SHAPE {j+1}\n")
        prompt_parts.append(f"POS: {shape.bottom_left_position[0]},{shape.bottom_left_position[1]}\n")
        prompt_parts.append(f"{shape_grid}\n")
        prompt_parts.append(f"DESCRIPTION: {shape.description}\n\n")
    
    # Task solving instructions
    prompt_parts.append("""
YOUR TASK:
Predict the output shapes for the test input based on the transformations in the training examples.

OUTPUT FORMAT:
For each output shape, provide:
1. SHAPE number
2. Position (x,y coordinates)
3. Detailed description
4. Grid representation

Example output:
SHAPE 1
POS: 0,0
Description of the shape's characteristics and transformation.
0000
0110
0000

Total grid width and height must match the input grid dimensions.
""")
    
    return "\n".join(prompt_parts)


def parse_solver_response(response_text: str, input_grid_width: int, input_grid_height: int) -> Tuple[List[PredictedShape], Optional[Tuple[int, int]]]:
    """
    Parse the raw text response from Gemini into a structured format.
    
    Args:
        response_text: Raw text response from Gemini
        input_grid_width: Width of the input grid
        input_grid_height: Height of the input grid
        
    Returns:
        Tuple of (list of predicted shapes, grid dimensions)
    """
    # Print the full response for debugging
    print(f"\n{Fore.CYAN}==== GEMINI RESPONSE ===={Style.RESET_ALL}")
    print(response_text)
    print(f"{Fore.CYAN}==== END OF RESPONSE ===={Style.RESET_ALL}\n")

    predicted_shapes = []
    
    # Split the response into sections
    sections = response_text.split("\n\n")
    
    # Look for shape sections
    current_shape = None
    shape_details = {}
    
    for section in sections:
        lines = section.strip().split("\n")
        
        # Check for shape start
        if lines[0].startswith("SHAPE "):
            # If we were processing a previous shape, add it
            if current_shape is not None and 'grid' in shape_details:
                predicted_shapes.append(
                    PredictedShape(
                        description=shape_details.get('description', 'No description'),
                        position=shape_details.get('pos', (0, 0))
                    )
                )
            
            # Reset for new shape
            current_shape = lines[0].strip()
            shape_details = {'number': current_shape}
        
        # Parse position
        elif any(line.startswith("POS:") for line in lines):
            pos_line = next(line for line in lines if line.startswith("POS:"))
            try:
                x, y = map(int, pos_line.replace("POS:", "").strip().split(","))
                shape_details['pos'] = (x, y)
            except Exception:
                print(f"{Fore.YELLOW}⚠ Could not parse position: {pos_line}{Style.RESET_ALL}")
        
        # Collect description
        elif any(line.startswith("Description") for line in lines):
            shape_details['description'] = "\n".join(lines)
        
        # Collect grid
        elif all(set(line).issubset({'0', '1', ' '}) for line in lines if line.strip()):
            # Convert grid to numpy array
            try:
                grid_lines = [list(map(int, list(line.replace(' ', '0')))) for line in lines if line.strip()]
                shape_details['grid'] = np.array(grid_lines)
            except Exception:
                print(f"{Fore.YELLOW}⚠ Could not parse grid{Style.RESET_ALL}")
    
    # Add the last shape
    if current_shape is not None and 'grid' in shape_details:
        predicted_shapes.append(
            PredictedShape(
                description=shape_details.get('description', 'No description'),
                position=shape_details.get('pos', (0, 0))
            )
        )
    
    # Determine grid dimensions
    grid_dimensions = (input_grid_width, input_grid_height)
    
    return predicted_shapes, grid_dimensions


def print_task_details(task_id: str, prompt: str, response: str):
    """
    Print detailed information about the task solving process.
    
    Args:
        task_id: Identifier for the task
        prompt: Prompt sent to the model
        response: Response from the model
    """
    print("\n" + "="*50)
    print(f"TASK SOLVING DETAILS FOR: {task_id}")
    print("="*50)

    print("\n--- FULL DETAILED PROMPT ---")
    print(prompt)

    print("\n--- FULL DETAILED RESPONSE ---")
    print(response)

    print("\n" + "="*50)
    print("END OF TASK DETAILS")
    print("="*50 + "\n")