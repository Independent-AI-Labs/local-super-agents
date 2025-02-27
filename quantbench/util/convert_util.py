import glob
import json
import os
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

import onnx
import torch
from onnx2pytorch import ConvertModel

from integration.data.config import LLAMACPP_WORKING_DIR
from integration.util.misc_util import run_subprocess_command
from quantbench.qb_config import CONVERT_HF_TO_GGUF_SCRIPT, SAFETENSORS, GGUF, PYTORCH, ONNX, GPTQ, UNSUPPORTED, SupportedFeatures


def detect_model_format(input_dir: str) -> SupportedFeatures.MODEL_FORMAT_TYPES:
    """
    Automatically detect the format of the model in the input directory.

    Args:
        input_dir (str): Path to the directory containing the model files.

    Returns:
        ModelFormat: Detected model format.
    """
    input_dir_path = Path(input_dir)

    # Check for safetensors (includes handling for sharded models)
    safetensors_files = list(input_dir_path.glob("*.safetensors*"))
    if safetensors_files:
        return SAFETENSORS

    # Check for PyTorch models
    pytorch_files = list(input_dir_path.glob("*.bin")) + list(input_dir_path.glob("*.pt")) + list(input_dir_path.glob("*.pth"))
    if pytorch_files:
        return PYTORCH

    # Check for ONNX models (including those with external data)
    onnx_files = list(input_dir_path.glob("*.onnx"))
    onnx_folder = input_dir_path / "onnx"
    if onnx_files or (onnx_folder.exists() and onnx_folder.is_dir()):
        return ONNX

    # Check for GGUF models
    gguf_files = list(input_dir_path.glob("*.gguf"))
    if gguf_files:
        return GGUF

    # Check for GPTQ models - typically these have a quantize_config.json
    quantize_config = input_dir_path / "quantize_config.json"
    if quantize_config.exists():
        return GPTQ

    # If no known format is detected
    return UNSUPPORTED


def convert_safetensors_to_gguf(input_dir: str, output_path: str) -> Tuple[int, str]:
    """
    Convert a model in safetensors format to GGUF format.

    Args:
        input_dir (str): Path to the directory containing the safetensors model.
        output_path (str): Path to save the GGUF model.

    Returns:
        Tuple[int, str]: Exit code and output text.
    """
    return convert_to_full_precision_gguf(input_dir, output_path)


def convert_to_full_precision_gguf(input_dir_val, output_dir_val):
    """
    Convert a Hugging Face model to GGUF format.

    Args:
        input_dir_val (str): Path to the input model directory.
        output_dir_val (str): Path to save the GGUF model.

    Returns:
        Tuple[int, str]: Exit code and output text.
    """
    convert_hf_to_gguf_path = os.path.join(LLAMACPP_WORKING_DIR, CONVERT_HF_TO_GGUF_SCRIPT)
    command_fp = ["python", convert_hf_to_gguf_path, input_dir_val, "--outfile", output_dir_val]
    output_console_text = f"Running command: {' '.join(command_fp)}\n"
    ret_code, cmd_output = run_subprocess_command(command_fp)
    output_console_text += cmd_output
    return ret_code, output_console_text


def convert_gguf_to_onnx(input_path: str, output_dir: str, precision: str = "fp32",
                         engine: str = "cpu", cache_dir: Optional[str] = None) -> Tuple[int, str]:
    """
    Convert a GGUF model to ONNX format using onnxruntime_genai.

    Args:
        input_path (str): Path to the GGUF model.
        output_dir (str): Directory to save the ONNX model.
        precision (str): Precision to use for conversion (default: "fp32").
        engine (str): Engine to use for conversion (default: "cpu").
        cache_dir (Optional[str]): Cache directory to use (default: None).

    Returns:
        Tuple[int, str]: Exit code and output text.
    """
    # Extract model name from the input path
    model_name = os.path.basename(input_path).split('.')[0]

    # Prepare the command
    command = [
        "python3", "-m", "onnxruntime_genai.models.builder",
        "-m", model_name,
        "-i", input_path,
        "-o", output_dir,
        "-p", precision,
        "-e", engine
    ]

    # Add cache directory if provided
    if cache_dir:
        command.extend(["-c", cache_dir])

    # Run the command
    output_console_text = f"Running command: {' '.join(command)}\n"
    ret_code, cmd_output = run_subprocess_command(command)
    output_console_text += cmd_output

    return ret_code, output_console_text


def convert_onnx_to_pytorch(onnx_path: str, pytorch_path: str) -> Tuple[int, str]:
    """
    Converts an ONNX model to a PyTorch model.

    Args:
        onnx_path (str): Path to the ONNX model.
        pytorch_path (str): Path to save the PyTorch model.

    Returns:
        Tuple[int, str]: Exit code and output text.
    """
    try:
        # Check if onnx_path is a directory with external data
        onnx_path_obj = Path(onnx_path)
        if onnx_path_obj.is_dir():
            # Find the main .onnx file in the directory
            onnx_files = list(onnx_path_obj.glob("*.onnx"))
            if not onnx_files:
                return 1, f"No .onnx file found in directory {onnx_path}"
            onnx_path = str(onnx_files[0])

        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)

        # Convert to PyTorch
        pytorch_model = ConvertModel(onnx_model)

        # Save the PyTorch model
        torch.save(pytorch_model.state_dict(), pytorch_path)

        output_console_text = f"Successfully converted ONNX model to PyTorch and saved to {pytorch_path}\n"
        return 0, output_console_text
    except Exception as e:
        output_console_text = f"Error during ONNX to PyTorch conversion: {e}\n"
        return 1, output_console_text


def convert_pytorch_to_gptq(pytorch_path: str, gptq_path: str, bits: int = 4,
                            group_size: int = 128) -> Tuple[int, str]:
    """
    Convert a PyTorch model to GPTQ format.

    Args:
        pytorch_path (str): Path to the PyTorch model.
        gptq_path (str): Path to save the GPTQ model.
        bits (int): Quantization bits (default: 4).
        group_size (int): Group size for quantization (default: 128).

    Returns:
        Tuple[int, str]: Exit code and output text.
    """
    try:
        # This is a placeholder for the actual GPTQ conversion logic
        # In a real implementation, you would use a library like auto-gptq

        # Example command:
        # command = [
        #     "python", "-m", "auto_gptq.quantize",
        #     "--model_path", pytorch_path,
        #     "--output_dir", gptq_path,
        #     "--bits", str(bits),
        #     "--group_size", str(group_size)
        # ]
        # ret_code, cmd_output = run_subprocess_command(command)

        # For now, we'll just create a dummy quantize_config.json
        os.makedirs(gptq_path, exist_ok=True)
        quantize_config = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": False
        }
        with open(os.path.join(gptq_path, "quantize_config.json"), "w") as f:
            json.dump(quantize_config, f, indent=2)

        output_console_text = f"PyTorch to GPTQ conversion not fully implemented. Created placeholder config in {gptq_path}\n"
        return 0, output_console_text
    except Exception as e:
        output_console_text = f"Error during PyTorch to GPTQ conversion: {e}\n"
        return 1, output_console_text


def convert_gguf_to_pytorch(input_path: str, output_path: str, intermediate_dir: str) -> Tuple[int, str]:
    """Convert a GGUF model to PyTorch format using ONNX as intermediate."""
    # Create ONNX directory
    onnx_dir = os.path.join(intermediate_dir, "onnx_model")
    os.makedirs(onnx_dir, exist_ok=True)

    # Step 1: Convert GGUF to ONNX
    ret_code, conversion_output = convert_gguf_to_onnx(input_path, onnx_dir)
    if ret_code != 0:
        return ret_code, conversion_output

    # Find the generated ONNX file
    onnx_files = glob.glob(os.path.join(onnx_dir, "**", "*.onnx"), recursive=True)
    if not onnx_files:
        return 1, conversion_output + "No ONNX file found after conversion.\n"

    # Step 2: Convert ONNX to PyTorch
    onnx_ret_code, onnx_conversion_output = convert_onnx_to_pytorch(onnx_files[0], output_path)
    return onnx_ret_code, conversion_output + onnx_conversion_output


def convert_gguf_to_gptq(input_path: str, output_path: str, intermediate_dir: str) -> Tuple[int, str]:
    """Convert a GGUF model to GPTQ format using PyTorch as intermediate."""
    # Create PyTorch directory
    pytorch_dir = os.path.join(intermediate_dir, "pytorch_model")
    os.makedirs(pytorch_dir, exist_ok=True)

    # Step 1: Convert GGUF to PyTorch
    ret_code, conversion_output = convert_gguf_to_pytorch(input_path, pytorch_dir, intermediate_dir)
    if ret_code != 0:
        return ret_code, conversion_output

    # Find the generated PyTorch file(s)
    pytorch_files = glob.glob(os.path.join(pytorch_dir, "**", "*.pt"), recursive=True)
    if not pytorch_files:
        return 1, conversion_output + "No PyTorch file found after conversion.\n"

    # Step 2: Convert PyTorch to GPTQ
    pytorch_ret_code, pytorch_conversion_output = convert_pytorch_to_gptq(pytorch_files[0], output_path)
    return pytorch_ret_code, conversion_output + pytorch_conversion_output


def convert_safetensors_to_onnx(input_path: str, output_path: str, intermediate_dir: str) -> Tuple[int, str]:
    """Convert a SafeTensors model to ONNX format using GGUF as intermediate."""
    # Step 1: Convert to GGUF first
    fp_model_name = f"{os.path.basename(input_path.rstrip(os.sep))}-F16"
    gguf_path = os.path.join(intermediate_dir, f"{fp_model_name}.gguf")
    ret_code, conversion_output = convert_safetensors_to_gguf(input_path, gguf_path)

    if ret_code != 0:
        return ret_code, conversion_output

    # Step 2: Convert GGUF to ONNX
    gguf_ret_code, gguf_conversion_output = convert_gguf_to_onnx(gguf_path, output_path)
    return gguf_ret_code, conversion_output + gguf_conversion_output


def convert_safetensors_to_pytorch(input_path: str, output_path: str, intermediate_dir: str) -> Tuple[int, str]:
    """Convert a SafeTensors model to PyTorch format using GGUF and ONNX as intermediates."""
    # Step 1: Convert to GGUF first
    fp_model_name = f"{os.path.basename(input_path.rstrip(os.sep))}-F16"
    gguf_path = os.path.join(intermediate_dir, f"{fp_model_name}.gguf")
    ret_code, conversion_output = convert_safetensors_to_gguf(input_path, gguf_path)

    if ret_code != 0:
        return ret_code, conversion_output

    # Step 2: Convert GGUF to PyTorch
    gguf_ret_code, gguf_conversion_output = convert_gguf_to_pytorch(gguf_path, output_path, intermediate_dir)
    return gguf_ret_code, conversion_output + gguf_conversion_output


def convert_safetensors_to_gptq(input_path: str, output_path: str, intermediate_dir: str) -> Tuple[int, str]:
    """Convert a SafeTensors model to GPTQ format using GGUF and PyTorch as intermediates."""
    # Step 1: Convert to PyTorch via GGUF
    pytorch_dir = os.path.join(intermediate_dir, "pytorch_model")
    os.makedirs(pytorch_dir, exist_ok=True)
    ret_code, conversion_output = convert_safetensors_to_pytorch(input_path, pytorch_dir, intermediate_dir)

    if ret_code != 0:
        return ret_code, conversion_output

    # Find the generated PyTorch file(s)
    pytorch_files = glob.glob(os.path.join(pytorch_dir, "**", "*.pt"), recursive=True)
    if not pytorch_files:
        return 1, conversion_output + "No PyTorch file found after conversion.\n"

    # Step 2: Convert PyTorch to GPTQ
    pytorch_ret_code, pytorch_conversion_output = convert_pytorch_to_gptq(pytorch_files[0], output_path)
    return pytorch_ret_code, conversion_output + pytorch_conversion_output


def convert_model(input_path: str, output_path: str, target_format: str,
                  intermediate_dir: Optional[str] = None) -> Tuple[int, str]:
    """
    Convert a model from its source format to the target format.

    Args:
        input_path (str): Path to the input model.
        output_path (str): Path to save the output model.
        target_format (str): Target format for conversion.
        intermediate_dir (Optional[str]): Directory for intermediate files (default: None).

    Returns:
        Tuple[int, str]: Exit code and output text.
    """
    # Detect input format
    source_format = detect_model_format(input_path)

    # Use a temporary directory for intermediate conversions if not provided
    if intermediate_dir is None:
        intermediate_dir = os.path.join(os.path.dirname(output_path), "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)

    output_console_text = f"Converting from {source_format} to {target_format}\n"

    # If source and target are the same, just copy the files
    if source_format == target_format:
        output_console_text += f"Source and target formats are the same ({source_format}). No conversion needed.\n"
        # Implement file copying logic here if needed
        if os.path.isfile(input_path):
            shutil.copy(input_path, output_path)
            output_console_text += f"Copied file from {input_path} to {output_path}\n"
        elif os.path.isdir(input_path):
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            shutil.copytree(input_path, output_path)
            output_console_text += f"Copied directory from {input_path} to {output_path}\n"
        return 0, output_console_text

    # Check if the conversion is supported
    if source_format not in SupportedFeatures.CONVERSION_PATHS or target_format not in SupportedFeatures.CONVERSION_PATHS[source_format]:
        output_console_text += f"Direct conversion from {source_format} to {target_format} is not supported.\n"

        # Check if we can find a path through intermediate formats
        conversion_found = False
        for intermediate_format in SupportedFeatures.CONVERSION_PATHS.keys():
            if (source_format in SupportedFeatures.CONVERSION_PATHS and
                    intermediate_format in SupportedFeatures.CONVERSION_PATHS[source_format] and
                    target_format in SupportedFeatures.CONVERSION_PATHS[intermediate_format]):
                conversion_found = True
                output_console_text += f"Using {intermediate_format} as intermediate format.\n"
                break

        if not conversion_found:
            return 1, output_console_text + "No conversion path available.\n"

    # Direct conversion mappings
    conversion_map = {
        # Direct conversions
        (SAFETENSORS, GGUF): convert_safetensors_to_gguf,
        (GGUF, ONNX): convert_gguf_to_onnx,
        (ONNX, PYTORCH): convert_onnx_to_pytorch,
        (PYTORCH, GPTQ): convert_pytorch_to_gptq,

        # Multi-step conversions that have dedicated functions
        (SAFETENSORS, ONNX): lambda inp, out: convert_safetensors_to_onnx(inp, out, intermediate_dir),
        (SAFETENSORS, PYTORCH): lambda inp, out: convert_safetensors_to_pytorch(inp, out, intermediate_dir),
        (SAFETENSORS, GPTQ): lambda inp, out: convert_safetensors_to_gptq(inp, out, intermediate_dir),
        (GGUF, PYTORCH): lambda inp, out: convert_gguf_to_pytorch(inp, out, intermediate_dir),
        (GGUF, GPTQ): lambda inp, out: convert_gguf_to_gptq(inp, out, intermediate_dir),
    }

    conversion_key = (source_format, target_format)

    # If we have a direct conversion function, use it
    if conversion_key in conversion_map:
        ret_code, conversion_output = conversion_map[conversion_key](input_path, output_path)
        output_console_text += conversion_output
        return ret_code, output_console_text

    # If no direct path, find the shortest valid path
    # This is a simple implementation - in practice, you might want to use a graph
    # algorithm to find the optimal path based on conversion costs or other metrics
    valid_paths = []

    # Function to find all valid paths recursively
    def find_paths(current_format, path, visited):
        if current_format == target_format:
            valid_paths.append(path[:])
            return

        if current_format in visited:
            return

        visited.add(current_format)

        for next_format in SupportedFeatures.CONVERSION_PATHS.get(current_format, []):
            if next_format not in visited:
                path.append(next_format)
                find_paths(next_format, path, visited.copy())
                path.pop()

    # Start path finding from source format
    find_paths(source_format, [source_format], set())

    if not valid_paths:
        return 1, output_console_text + "No valid conversion path found.\n"

    # Sort paths by length to find the shortest one
    valid_paths.sort(key=len)
    shortest_path = valid_paths[0]

    output_console_text += f"Using conversion path: {' -> '.join(shortest_path)}\n"

    # Execute the conversion step by step
    current_input = input_path
    current_format = source_format

    for i in range(1, len(shortest_path)):
        next_format = shortest_path[i]

        # For the last step, use the final output path
        if i == len(shortest_path) - 1:
            next_output = output_path
        else:
            # Create an intermediate file
            intermediate_filename = f"intermediate_{i}_{next_format}"
            next_output = os.path.join(intermediate_dir, intermediate_filename)

        conversion_key = (current_format, next_format)

        if conversion_key in conversion_map:
            ret_code, conversion_output = conversion_map[conversion_key](current_input, next_output)
            output_console_text += conversion_output

            if ret_code != 0:
                return ret_code, output_console_text

            current_input = next_output
            current_format = next_format
        else:
            return 1, output_console_text + f"No conversion function found for {conversion_key}\n"

    return 0, output_console_text


def is_sharded_model(input_dir: str) -> bool:
    """
    Check if a model is sharded by looking for model shard patterns.

    Args:
        input_dir (str): Path to the model directory.

    Returns:
        bool: True if the model is sharded, False otherwise.
    """
    input_dir_path = Path(input_dir)

    # Check for safetensors shards (like model.safetensors.index.json or model-00001-of-00003.safetensors)
    safetensors_index = list(input_dir_path.glob("*.safetensors.index.json"))
    if safetensors_index:
        return True

    numbered_shards = list(input_dir_path.glob("*-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].*"))
    if numbered_shards:
        return True

    # Check for pytorch shards
    pytorch_index = list(input_dir_path.glob("*.bin.index.json"))
    if pytorch_index:
        return True

    return False


def get_model_shards(input_dir: str) -> List[str]:
    """
    Get the list of shard files for a sharded model.

    Args:
        input_dir (str): Path to the model directory.

    Returns:
        List[str]: List of paths to shard files.
    """
    input_dir_path = Path(input_dir)
    shards = []

    # Try to find an index file first
    index_files = list(input_dir_path.glob("*.index.json"))
    if index_files:
        # Load the index file to get shard information
        with open(index_files[0], 'r') as f:
            index_data = json.load(f)

        # Extract shard paths from the index file
        # Note: The exact structure depends on the model format
        if 'weight_map' in index_data:
            # Common format for HF models
            unique_shards = set(index_data['weight_map'].values())
            shards = [str(input_dir_path / shard) for shard in unique_shards]
        else:
            # Fallback to scanning for numbered shards
            shards = [str(path) for path in input_dir_path.glob("*-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].*")]
    else:
        # If no index file, look for numbered shards directly
        shards = [str(path) for path in input_dir_path.glob("*-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].*")]

    return sorted(shards)
