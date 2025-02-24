import os
import subprocess
import time
from datetime import timedelta

import gradio as gr  # Assuming gradio is used for UI, based on `gr.Progress`

# Import paths from integration.data.config
from integration.data.config import LLAMACPP_WORKING_DIR, LLAMACPP_QUANTIZE_PATH
from quantbench.quant_util import calculate_file_size_and_percentage, get_available_quant_types

# Constants for file paths and script names
CONVERT_HF_TO_GGUF_SCRIPT = "convert_hf_to_gguf.py"
MODEL_FILE_EXTENSION = ".safetensors"
GGUF_FILE_EXTENSION = ".gguf"
F16_QUANT_TYPE = "F16"


def _run_subprocess_command(command, output_console):
    """
    Runs a subprocess command and streams its output to the output console.

    Args:
        command (list): The command to run as a list of strings.
        output_console (gr.Textbox): Gradio Textbox to display the output.

    Returns:
        tuple: (return_code, output_console_text) - Return code of the process and accumulated console output.
    """
    output_console_text = ""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()

        if stdout_line:
            output_console_text += stdout_line
            output_console.value = output_console_text  # Live update
        if stderr_line:
            output_console_text += stderr_line
            output_console.value = output_console_text  # Live update

        if not stdout_line and not stderr_line and process.poll() is not None:
            break

    return process.returncode, output_console_text


def _get_gguf_file_path(output_dir, model_filename_base, quant_type):
    """
    Constructs the GGUF file path based on output directory, model name, and quantization type.

    Args:
        output_dir (str): Output directory path.
        model_filename_base (str): Base name of the model file.
        quant_type (str): Quantization type.

    Returns:
        str: Full path to the GGUF file.
    """
    return os.path.join(output_dir, f"{model_filename_base}-{quant_type}{GGUF_FILE_EXTENSION}")


def _get_time_to_quantize(start_time, end_time):
    """
    Calculates and formats the Time To Quantize (TTQ).

    Args:
        start_time (float): Start time of the process.
        end_time (float): End time of the process.

    Returns:
        str: Formatted TTQ string.
    """
    return str(timedelta(seconds=int(end_time - start_time)))


def _quantize_model(f16_gguf_path, output_file_path, quant_type, output_console):
    """
    Quantizes a model using llama.cpp's quantize tool.

    Args:
        f16_gguf_path (str): Path to the F16 GGUF file.
        output_file_path (str): Path to save the quantized GGUF file.
        quant_type (str): Quantization type.
        output_console (gr.Textbox): Gradio Textbox for output.

    Returns:
        tuple: (return_code, output_console_text) - Return code and console output.
    """
    command = [LLAMACPP_QUANTIZE_PATH, f16_gguf_path, output_file_path, quant_type]
    output_console_text = f"Running command: {' '.join(command)}\n"
    output_console.value = output_console_text  # Live update
    return _run_subprocess_command(command, output_console)


def _convert_to_f16_gguf(input_dir_val, output_dir_val, output_console):
    """
    Converts a model in safetensors format to F16 GGUF format using llama.cpp's conversion script.

    Args:
        input_dir_val (str): Input directory containing the safetensors file.
        output_dir_val (str): Output directory to save the F16 GGUF file.
        output_console (gr.Textbox): Gradio Textbox for output.

    Returns:
        tuple: (return_code, output_console_text) - Return code and console output.
    """
    convert_hf_to_gguf_path = os.path.join(LLAMACPP_WORKING_DIR, CONVERT_HF_TO_GGUF_SCRIPT)
    command_f16 = ["python", convert_hf_to_gguf_path, input_dir_val, "--outfile", output_dir_val]
    output_console_text = f"Running command: {' '.join(command_f16)}\n"
    output_console.value = output_console_text  # Live update
    return _run_subprocess_command(command_f16, output_console)


def quantize_and_benchmark_process(input_dir_val, output_dir_val, quant_types_val, output_console):
    """
    Quantizes and benchmarks a model for different quantization types.

    Args:
        input_dir_val (str): Input directory containing the .safetensors model.
        output_dir_val (str): Output directory to save quantized models.
        quant_types_val (list): List of quantization types to perform.
        output_console (gr.Textbox): Gradio Textbox for output.

    Returns:
        tuple: (output_console_text, new_results_data) - Console output and benchmark results data.
    """
    output_console_text = ""
    new_results_data = []
    available_quants = get_available_quant_types()
    valid_quant_types_val = [q for q in quant_types_val if q in available_quants]

    # Input validation
    if not input_dir_val or not output_dir_val or not valid_quant_types_val:
        error_message = "Error: Please provide input directory, output directory, and select at least one quantization type.\n"
        output_console_text += error_message
        return output_console_text, []

    if not os.path.isdir(input_dir_val):
        error_message = f"Error: Input directory '{input_dir_val}' does not exist or is not a directory.\n"
        output_console_text += error_message
        return output_console_text, []

    if not os.path.isdir(output_dir_val):
        try:
            os.makedirs(output_dir_val, exist_ok=True)
            output_console_text += f"Created output directory: {output_dir_val}\n"
        except OSError as e:
            error_message = f"Error: Could not create output directory '{output_dir_val}': {e}\n"
            output_console_text += error_message
            return output_console_text, []

    safetensors_files = [f for f in os.listdir(input_dir_val) if f.endswith(MODEL_FILE_EXTENSION)]
    if not safetensors_files:
        error_message = "Error: No .safetensors files found in the input directory.\n"
        output_console_text += error_message
        return output_console_text, []

    model_filename_base = safetensors_files[0].replace(MODEL_FILE_EXTENSION, "")
    input_file_path = os.path.join(input_dir_val, safetensors_files[0])
    output_console_text += f"Found safetensors file: {safetensors_files[0]}\n"
    original_size_bytes = os.path.getsize(input_file_path)

    # --- F16 Conversion and Benchmarking ---
    f16_gguf_path = _get_gguf_file_path(output_dir_val, model_filename_base, F16_QUANT_TYPE)

    if F16_QUANT_TYPE in valid_quant_types_val:
        f16_start_time = time.time()
        if not os.path.exists(f16_gguf_path):
            return_code_f16, f16_conversion_output = _convert_to_f16_gguf(input_dir_val, output_dir_val, output_console)
            output_console_text += f16_conversion_output
            if return_code_f16 != 0:
                error_message = f"Error during F16 conversion. See console output for details.\n"
                output_console_text += error_message
                return output_console_text, []
        else:
            output_console_text += f"F16 GGUF already exists: {f16_gguf_path}\n"
            output_console.value = output_console_text  # Live update

        f16_end_time = time.time()
        ttq_f16 = _get_time_to_quantize(f16_start_time, f16_end_time)
        f16_size_bytes = os.path.getsize(f16_gguf_path)
        f16_size_gb, f16_size_percent = calculate_file_size_and_percentage(original_size_bytes, f16_size_bytes)

        benchmark_result_f16 = {
            "TTQ": ttq_f16,
            "Size (GB)": f16_size_gb,
            "Size (%)": f16_size_percent,
            "Test Runs": "N/A",
            "Avg Model Load Time": "N/A",
            "Avg Image Encoding Time / MPixels": "N/A",
            "Avg Prompt Comprehension (t/s)": "100",  # Placeholder
            "Avg Eval / Generate (t/s)": "100",  # Placeholder
            "Quality (% of Ground Truth)": "100.00%",
        }
        new_results_data.append([f16_gguf_path, F16_QUANT_TYPE] + list(benchmark_result_f16.values()))

    # --- Other Quantization Types ---
    with gr.Progress(track_tqdm=True) as progress:
        for quant_type in progress.tqdm(valid_quant_types_val, desc="Quantizing models"):
            if quant_type == F16_QUANT_TYPE:
                continue

            output_file_path = _get_gguf_file_path(output_dir_val, model_filename_base, quant_type)

            if os.path.exists(output_file_path):
                output_console_text += f"Skipping {quant_type} (already exists): {output_file_path}\n"
                output_console.value = output_console_text  # Live update
                new_results_data.append([output_file_path, quant_type] + ["N/A"] * 9)  # Placeholders for benchmark data
                continue

            start_time = time.time()
            return_code, quantization_output = _quantize_model(f16_gguf_path, output_file_path, quant_type, output_console)
            output_console_text += quantization_output

            if return_code == 0:
                end_time = time.time()
                ttq = _get_time_to_quantize(start_time, end_time)
                quantized_size_bytes = os.path.getsize(output_file_path)
                quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(original_size_bytes, quantized_size_bytes)

                benchmark_result = {
                    "TTQ": ttq,
                    "Size (GB)": quantized_size_gb,
                    "Size (%)": quantized_size_percent,
                    "Test Runs": "N/A",
                    "Avg Model Load Time": "N/A",
                    "Avg Image Encoding Time / MPixels": "N/A",
                    "Avg Prompt Comprehension (t/s)": "50",  # Placeholder
                    "Avg Eval / Generate (t/s)": "50",  # Placeholder
                    "Quality (% of Ground Truth)": "95.50%",  # Placeholder
                }
                new_results_data.append([output_file_path, quant_type] + list(benchmark_result.values()))
            else:
                error_message = f"Error during quantization ({quant_type}). See console output for details.\n"
                output_console_text += error_message
                output_console.value = output_console_text  # Live update
                new_results_data.append([output_file_path, quant_type] + ["ERROR"] * 9)  # Placeholders for benchmark data

    return output_console_text, new_results_data
