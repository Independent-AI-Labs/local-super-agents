import os
import subprocess
import threading
from datetime import timedelta

from integration.data.config import LLAMACPP_WORKING_DIR, LLAMACPP_QUANTIZE_PATH, LLAMACPP_IMATRIX_PATH
from quantbench.errors.quantization_error import QuantizationError

# Constants
CONVERT_HF_TO_GGUF_SCRIPT = "convert_hf_to_gguf.py"
MODEL_FILE_EXTENSION = ".safetensors"
GGUF_FILE_EXTENSION = ".gguf"
F16_QUANT_TYPE = "F16"
IMATRIX_FILE_NAME = "imatrix.dat"
IMATRIX_CUSTOM_FILE_NAME = "imatrix-custom.dat"
IMATRIX_DATASET_FILE = "train-data.txt"  # Hardcoded dataset file for now


class SupportedFeatures:
    AVAILABLE_QUANT_TYPES = [
        "F16", "Q8_0", "Q4_0", "IQ4_XS", "Q4_K_M", "Q4_K_S", "Q6_K", "Q5_K_M",
        "Q5_K_S", "IQ4_NL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"
    ]
    IMATRIX_OPTIONS = ["None (Reduced Quality)", "Use Included", "Generate Custom Matrix"]
    OUTPUT_FORMATS = ["GGUF"]
    RESULTS_TABLE_HEADERS = [
        "Model File Path",
        "Type",
        "TTQ",
        "Size",
        "% of F16",
        "Test Runs",
        "Model Load (s)",
        "Encode (s/mpx)",
        "Prompt (t/s)",
        "Resp. (t/s)",
        "Quality (%)",
    ]
    RESULTS_TABLE_COLUMNS_WIDTH = ["16%", "8%", "8%", "8%", "8%", "7%", "9%", "9%", "9%", "9%", "9%"]


def _run_subprocess_command(command):
    """
    Runs a subprocess command and returns its exit code and accumulated output.
    """
    output_lines = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    def read_stream(stream):
        for line in iter(stream.readline, ''):
            output_lines.append(line)
        stream.close()

    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout,))
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr,))
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    process.wait()

    return process.returncode, ''.join(output_lines)

def _get_gguf_file_path(output_dir, model_filename_base, quant_type):
    return os.path.join(output_dir, f"{model_filename_base}-{quant_type}{GGUF_FILE_EXTENSION}")


def _get_time_to_quantize(start_time, end_time):
    return str(timedelta(seconds=int(end_time - start_time)))


def _create_benchmark_result(ttq, size_gb, size_percent, error=False):
    default_value = "ERROR" if error else "N/A"
    return {
        "TTQ": ttq if not error else default_value,
        "Size (GB)": size_gb if not error else default_value,
        "Size (%)": size_percent if not error else default_value,
        "Test Runs": default_value,
        "Load (s)": default_value,
        "Encode (s/mpx)": default_value,
        "Prompt (t/s)": default_value,
        "Resp. (t/s)": default_value,
        "Quality (%)": default_value,
    }


def _quantize_model(f16_gguf_path, output_file_path, quant_type, imatrix_path=None):
    command = [LLAMACPP_QUANTIZE_PATH]
    if imatrix_path:
        command.extend(["--imatrix", imatrix_path])
    command.extend([f16_gguf_path, output_file_path, quant_type])
    output_console_text = f"Running command: {' '.join(command)}\n"
    ret_code, cmd_output = _run_subprocess_command(command)
    output_console_text += cmd_output
    return ret_code, output_console_text


def _convert_to_f16_gguf(input_dir_val, output_dir_val):
    convert_hf_to_gguf_path = os.path.join(LLAMACPP_WORKING_DIR, CONVERT_HF_TO_GGUF_SCRIPT)
    command_f16 = ["python", convert_hf_to_gguf_path, input_dir_val, "--outfile", output_dir_val]
    output_console_text = f"Running command: {' '.join(command_f16)}\n"
    ret_code, cmd_output = _run_subprocess_command(command_f16)
    output_console_text += cmd_output
    return ret_code, output_console_text


def _generate_imatrix(model_file, train_data_file, output_file):
    command = [
        LLAMACPP_IMATRIX_PATH,
        "-m", model_file,
        "-f", train_data_file,
        "-o", output_file,
        "--n-gpu-layers", "9999",  # Based on hardware configuration.
    ]
    output_console_text = f"Generating imatrix with command: {' '.join(command)}\n"
    ret_code, cmd_output = _run_subprocess_command(command)
    output_console_text += cmd_output
    return ret_code, output_console_text


def get_available_quant_types():
    """
    Returns the list of available quantization types.

    Returns:
        list: List of quantization types.
    """
    return SupportedFeatures.AVAILABLE_QUANT_TYPES


def calculate_file_size_and_percentage(original_size_bytes, quantized_size_bytes):
    """Calculates file size in GB and percentage of original size."""
    if original_size_bytes == 0:
        return "0.00 GB", "0.00%"  # Avoid division by zero
    quantized_size_gb = quantized_size_bytes / (1024 ** 3)
    percentage_of_original = (quantized_size_bytes / original_size_bytes) * 100
    return f"{quantized_size_gb:.2f} GB", f"{percentage_of_original:.2f}%"

def _handle_error(output_console_text, error_message):
    """
    Helper function to handle errors by yielding the console text and raising a QuantizationError.
    """
    output_console_text += error_message
    yield output_console_text, []
    raise QuantizationError(error_message)
