import os
import traceback

import gradio as gr

from integration.util.misc_util import select_and_list_directory_contents, list_directory_contents
from quantbench.errors.quantization_error import QuantizationError


def validate_inputs(input_dir_val, output_dir_val, quant_types_val):
    if not input_dir_val or not output_dir_val or not quant_types_val:
        raise ValueError("Please provide input directory, output directory, and select at least one quantization type.")
    if not os.path.isdir(input_dir_val):
        raise ValueError(f"Input directory '{input_dir_val}' does not exist or is not a directory.")


def ensure_output_directory_exists(output_dir_val):
    if not os.path.isdir(output_dir_val):
        try:
            os.makedirs(output_dir_val, exist_ok=True)
            return f"Created output directory: {output_dir_val}\n"
        except OSError as e:
            raise OSError(f"Could not create output directory '{output_dir_val}': {e}")
    return ""


def remove_duplicate_results(all_results_data):
    seen_paths = set()
    unique_results_data = []
    for row in reversed(all_results_data):
        quant_path = row[0]  # Assume quant path is always the first element
        if quant_path not in seen_paths:
            unique_results_data.insert(0, row)
            seen_paths.add(quant_path)
    return unique_results_data


def update_button_states(interactive_value):
    return gr.update(interactive=interactive_value)


def update_input_directory(input_dir_val: str):
    selected_dir, dir_content = select_and_list_directory_contents(input_dir_val)
    dir_name = os.path.split(selected_dir)[-1]
    return selected_dir, gr.update(value=dir_content), gr.update(value=f"Quantize {dir_name}")


def update_output_directory(output_dir_val: str):
    selected_dir, dir_content = select_and_list_directory_contents(output_dir_val)
    # Also update the shared variable so that the background thread’s data
    # and the UI remain in sync (if you choose to poll it elsewhere)
    return selected_dir, gr.update(value=dir_content)


def periodic_update_output_dir_content(output_dir_val: str):  # Receive output_dir_component as argument
    if output_dir_val:
        try:
            contents = list_directory_contents(output_dir_val)  # Use output_dir_val
            return gr.update(value=contents)
        except Exception as e:
            error_msg = f"Error updating output directory content: {e}"
            print(error_msg)
            gr.Warning(error_msg)
            return gr.update()  # Return empty update to prevent errors in Gradio
    else:
        return gr.update()  # Return empty update if output_dir not initialized yet



def update_output_dir_label(format_val: str):
    return gr.update(label=f"Model Output Directory ({format_val})", )


def calculate_file_size_and_percentage(original_size_bytes, quantized_size_bytes):
    """Calculates file size in GB and percentage of original size."""
    if original_size_bytes == 0:
        return "0.00 GB", "0.00%"  # Avoid division by zero
    quantized_size_gb = quantized_size_bytes / (1024 ** 3)
    percentage_of_original = (quantized_size_bytes / original_size_bytes) * 100
    return f"{quantized_size_gb:.2f} GB", f"{percentage_of_original:.2f}%"


def handle_error(output_console_text, error_message):
    """
    Helper function to handle errors by yielding the console text and raising a QuantizationError.
    """
    output_console_text += error_message
    yield output_console_text, []
    raise QuantizationError(error_message)

