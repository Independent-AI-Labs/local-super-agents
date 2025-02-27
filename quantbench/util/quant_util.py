import os
from datetime import timedelta

import torch
from onnxruntime.quantization import quantize_dynamic, QuantType  # Import for ONNX quantization
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader, TensorDataset

from integration.data.config import LLAMACPP_QUANTIZE_PATH, LLAMACPP_IMATRIX_PATH
from integration.util.misc_util import run_subprocess_command
from quantbench.qb_config import GGUF_FILE_EXTENSION, PYTORCH_FILE_EXTENSION


def get_gguf_file_path(output_dir, model_filename_base, quant_type):
    return os.path.join(output_dir, f"{model_filename_base}-{quant_type}{GGUF_FILE_EXTENSION}")


def get_time_to_quantize(start_time, end_time):
    return str(timedelta(seconds=int(end_time - start_time)))


def generate_imatrix(model_file, train_data_file, output_file):
    command = [
        LLAMACPP_IMATRIX_PATH,
        "-m", model_file,
        "-f", train_data_file,
        "-o", output_file,
        "--n-gpu-layers", "9999",  # Based on hardware configuration.
    ]
    output_console_text = f"Generating imatrix with command: {' '.join(command)}\n"
    ret_code, cmd_output = run_subprocess_command(command)
    output_console_text += cmd_output
    return ret_code, output_console_text


def create_benchmark_result(ttq, size_gb, size_percent, error=False):
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


def quantize_model(f16_gguf_path, output_file_path, quant_type, imatrix_path=None):
    command = [LLAMACPP_QUANTIZE_PATH]
    if imatrix_path:
        command.extend(["--imatrix", imatrix_path])
    command.extend([f16_gguf_path, output_file_path, quant_type])
    output_console_text = f"Running command: {' '.join(command)}\n"
    ret_code, cmd_output = run_subprocess_command(command)
    output_console_text += cmd_output
    return ret_code, output_console_text


def quantize_onnx_dynamic(onnx_model_path, quantized_model_path):
    """
    Applies dynamic quantization to an ONNX model.

    Args:
        onnx_model_path (str): Path to the original ONNX model.
        quantized_model_path (str): Path to save the quantized ONNX model.
    Returns:
        tuple: Exit code and output text.
    """
    try:
        quantized_model = quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8,
            per_channel=True
        )
        output_console_text = f"ONNX model quantized to: {quantized_model_path}\n"
        return 0, output_console_text
    except Exception as e:
        output_console_text = f"Error during ONNX dynamic quantization: {e}\n"
        return 1, output_console_text


def quantize_pytorch_gptq(pytorch_model_path, dataset_representative_data):
    """
    Applies GPTQ quantization to a PyTorch model.

    Args:
        pytorch_model_path (str): Path to the original PyTorch model.
        dataset_representative_data (list): Representative data for quantization.
    Returns:
        tuple: Exit code and output text.
    """
    try:
        # Load the PyTorch model
        model = torch.load(pytorch_model_path)

        # Prepare the model for quantization
        model.eval()
        qconfig = get_default_qconfig("x86")
        model.qconfig = qconfig
        model_prepared = prepare(model)

        # Create a DataLoader for the representative data
        dataset = TensorDataset(torch.tensor(dataset_representative_data))
        data_loader = DataLoader(dataset, batch_size=1)

        # Calibrate the model
        with torch.no_grad():
            for data in data_loader:
                model_prepared(*data)

        # Convert the model to a quantized model
        model_quantized = convert(model_prepared)

        # Save the quantized model
        quantized_model_path = pytorch_model_path.replace(PYTORCH_FILE_EXTENSION, "_quantized" + PYTORCH_FILE_EXTENSION)
        torch.save(model_quantized.state_dict(), quantized_model_path)

        output_console_text = f"PyTorch model quantized and saved to: {quantized_model_path}\n"
        return 0, output_console_text
    except Exception as e:
        output_console_text = f"Error during PyTorch GPTQ quantization: {e}\n"
        return 1, output_console_text
