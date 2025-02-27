import os
from typing import Optional

import torch
from safetensors import safe_open

from quantbench.errors.quantization_error import QuantizationError
from quantbench.processors.quantization_processor import QuantizationProcessor
from quantbench.qb_config import SupportedFeatures, INPUT_MODEL_FILE_EXTENSION, GGUF, ONNX, PYTORCH, GPTQ
from quantbench.util.convert_util import convert_model
from quantbench.util.quant_util import get_gguf_file_path, create_benchmark_result, quantize_pytorch_gptq
from quantbench.util.ui_util import calculate_file_size_and_percentage


def get_fp_model(output_dir_val: str, model_filename_base: str) -> Optional[str]:
    """
    Retrieves the file path of a full-precision (FP) model.

    This function searches for a full-precision model file in the specified output directory.
    It iterates through the supported full-precision quantization types and checks if a corresponding
    GGUF file exists. If found, it returns the file path.

    Args:
        output_dir_val (str): The directory where the model files are located.
        model_filename_base (str): The base name of the model file.

    Returns:
        Optional[str]: The file path of the full-precision model if found, otherwise None.
    """
    output_dir_val = os.path.join(output_dir_val, "intermediate")
    for fp_quant_type in SupportedFeatures.FP_QUANT_TYPES:
        fp_model_path = get_gguf_file_path(output_dir_val, model_filename_base, fp_quant_type)
        if os.path.exists(fp_model_path):
            return fp_model_path
    return None


class GGUFQuantizationProcessor(QuantizationProcessor):
    """
    Quantization processor for GGUF format.
    """

    def __init__(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
        super().__init__(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)

    def process(self):
        yield from self._validate_inputs()

        # Check if a full precision GGUF file already exists
        full_precision_gguf_path = get_fp_model(self.output_dir_val, self.model_filename_base)

        if full_precision_gguf_path is None:
            self.output_console_text += f"Warning: No full precision GGUF file found. Creating a new one.\n"
            yield self.output_console_text, []

            # Create FP GGUF output path. Hardcoded for now.
            gguf_output_path = os.path.join(self.output_dir_val, "intermediate", f"{self.model_filename_base}-F16.gguf")

            # Convert to full precision GGUF using the new universal conversion logic
            ret_code_fp, fp_conversion_output = convert_model(
                self.input_dir_val,
                gguf_output_path,
                target_format=GGUF
            )

            self.output_console_text += fp_conversion_output
            yield self.output_console_text, []

            if ret_code_fp != 0:
                raise QuantizationError("Error during full-precision conversion. See console output for details.\n")

        # Get the path to the full precision model (will be available now)
        full_precision_gguf_path = get_fp_model(self.output_dir_val, self.model_filename_base)

        # Handle imatrix
        imatrix_path = yield from self._handle_imatrix(full_precision_gguf_path)

        # Process all quantization types
        quant_types_to_process = self.valid_quant_types_val[:]
        for quant_type in self.progress.tqdm(quant_types_to_process, desc="Quantizing models..."):
            yield from self._process_quant_type(quant_type, full_precision_gguf_path, imatrix_path)

        yield self.output_console_text, self.new_results_data


class ONNXQuantizationProcessor(QuantizationProcessor):
    def __init__(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
        super().__init__(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)

    def process(self):
        yield from self._validate_inputs()

        # Define intermediate and output paths
        intermediate_dir = os.path.join(self.output_dir_val, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)

        onnx_path = os.path.join(self.output_dir_val, f"{self.model_filename_base}.onnx")

        # Convert input model to ONNX using universal conversion
        ret_code_onnx, onnx_conversion_output = convert_model(
            self.input_dir_val,
            onnx_path,
            target_format=ONNX,
            intermediate_dir=intermediate_dir
        )

        self.output_console_text += onnx_conversion_output
        yield self.output_console_text, []

        if ret_code_onnx != 0:
            raise QuantizationError("Error during ONNX conversion. See console output for details.\n")

        self.output_console_text += f"Model converted to ONNX: {onnx_path}\n"
        yield self.output_console_text, []

        # Convert ONNX to PyTorch
        pytorch_path = os.path.join(self.output_dir_val, f"{self.model_filename_base}.pt")

        ret_code_pytorch, pytorch_conversion_output = convert_model(
            onnx_path,
            pytorch_path,
            target_format=PYTORCH
        )

        self.output_console_text += pytorch_conversion_output
        yield self.output_console_text, []

        if ret_code_pytorch != 0:
            raise QuantizationError("Error during ONNX to PyTorch conversion. See console output for details.\n")

        self.output_console_text += f"Model converted to PyTorch: {pytorch_path}\n"
        yield self.output_console_text, []

        # Get file size for benchmarking
        if os.path.exists(pytorch_path):
            size_bytes = os.path.getsize(pytorch_path)
            size_gb, size_percent = calculate_file_size_and_percentage(self.fp_size_bytes, size_bytes)
            benchmark_result = create_benchmark_result("N/A", size_gb, size_percent)
            self.new_results_data.append([pytorch_path, "PyTorch"] + list(benchmark_result.values()))

        yield self.output_console_text, self.new_results_data


class GPTQQuantizationProcessor(QuantizationProcessor):
    """
    Quantization processor for GPTQ format.
    Using direct conversion from input to GPTQ instead of going through ONNX.
    """

    def __init__(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
        super().__init__(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
        self.dataset_representative_data = []

    def process(self):
        yield from self._validate_inputs()

        # Define intermediate and output paths
        intermediate_dir = os.path.join(self.output_dir_val, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)

        # First convert to PyTorch directly, without going through ONNX
        pytorch_path = os.path.join(intermediate_dir, f"{self.model_filename_base}.pt")

        # Using universal conversion to get PyTorch model first
        ret_code_pytorch, pytorch_conversion_output = convert_model(
            self.input_dir_val,
            pytorch_path,
            target_format=PYTORCH,
            intermediate_dir=intermediate_dir
        )

        self.output_console_text += pytorch_conversion_output
        yield self.output_console_text, []

        if ret_code_pytorch != 0:
            raise QuantizationError("Error during conversion to PyTorch. See console output for details.\n")

        self.output_console_text += f"Model converted to PyTorch: {pytorch_path}\n"
        yield self.output_console_text, []

        # Create dataset for quantization
        yield from self._create_dataset_representative_data()

        # Quantize using GPTQ
        try:
            gptq_output_dir = os.path.join(self.output_dir_val, f"{self.model_filename_base}_gptq")
            os.makedirs(gptq_output_dir, exist_ok=True)

            # Either use direct PyTorch to GPTQ conversion
            ret_code_gptq, gptq_conversion_output = convert_model(
                pytorch_path,
                gptq_output_dir,
                target_format=GPTQ
            )

            self.output_console_text += gptq_conversion_output
            yield self.output_console_text, []

            if ret_code_gptq != 0:
                # Fallback to manual quantization if the conversion fails
                self.output_console_text += "Falling back to manual GPTQ quantization...\n"
                yield self.output_console_text, []

                quantized_model = quantize_pytorch_gptq(pytorch_path, self.dataset_representative_data)

                # Save the quantized model
                quantized_model_path = os.path.join(gptq_output_dir, f"{self.model_filename_base}-gptq.pt")
                torch.save(quantized_model, quantized_model_path)

                # Create quantize_config.json
                with open(os.path.join(gptq_output_dir, "quantize_config.json"), "w") as f:
                    import json
                    json.dump({
                        "bits": 4,
                        "group_size": 128,
                        "desc_act": False
                    }, f, indent=2)

                self.output_console_text += f"Quantized model saved to: {quantized_model_path}\n"
            else:
                # Find the quantized model path
                import glob
                quantized_model_files = glob.glob(os.path.join(gptq_output_dir, "*.pt"))
                if quantized_model_files:
                    quantized_model_path = quantized_model_files[0]
                    self.output_console_text += f"Quantized model saved to: {quantized_model_path}\n"
                else:
                    quantized_model_path = gptq_output_dir
                    self.output_console_text += f"Quantized model directory: {quantized_model_path}\n"

        except Exception as e:
            raise QuantizationError(f"Error during GPTQ quantization: {str(e)}") from e

        yield self.output_console_text, []

        # Calculate size for benchmarking
        if os.path.isdir(quantized_model_path):
            # If it's a directory, sum the sizes of all files
            total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, _, filenames in os.walk(quantized_model_path)
                             for filename in filenames)
            quantized_size_bytes = total_size
        else:
            # If it's a file
            quantized_size_bytes = os.path.getsize(quantized_model_path)

        quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(
            self.fp_size_bytes, quantized_size_bytes
        )

        benchmark_result = create_benchmark_result("N/A", quantized_size_gb, quantized_size_percent)

        self.new_results_data.append([quantized_model_path, "GPTQ"] + list(benchmark_result.values()))
        yield self.output_console_text, self.new_results_data

    def _create_dataset_representative_data(self):
        """
        Create dataset representative data using the first tensor of the safetensors
        """
        model_path = None
        # Find a safetensors file in the input directory
        for filename in os.listdir(self.input_dir_val):
            if filename.endswith(INPUT_MODEL_FILE_EXTENSION):
                model_path = os.path.join(self.input_dir_val, filename)
                break

        if model_path is None:
            # If no safetensors file found, try to detect format and create dummy data
            self.output_console_text += "No .safetensors files found. Creating generic representative data.\n"
            yield self.output_console_text, []

            # Default input shape for representative data
            input_shape = (1, 768)  # Common embedding dimension, adjust if needed
            self.dataset_representative_data = [torch.randn(input_shape) for _ in range(10)]
            return

        try:
            # Try to read the safetensors file to get tensor shape
            with safe_open(model_path, framework="pt", device="cpu") as f:
                first_tensor_name = next(iter(f.keys()))
                tensor_info = f.get_tensor(first_tensor_name)
                # Use a shape that makes sense for quantization
                if len(tensor_info.shape) > 1:
                    input_shape = tensor_info.shape[1:]
                else:
                    input_shape = tensor_info.shape
        except Exception as e:
            self.output_console_text += f"Could not read tensor shape from safetensors: {str(e)}\n"
            self.output_console_text += "Using default shape instead.\n"
            yield self.output_console_text, []
            input_shape = (1, 768)  # Default shape if we can't determine from file

        self.output_console_text += f"Using tensor shape for representative data: {input_shape}\n"
        yield self.output_console_text, []

        # Create 10 random tensors with the appropriate shape
        self.dataset_representative_data = [torch.randn(input_shape) for _ in range(10)]


def create_quantization_processor(output_format_val, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
    """
    Factory function to create a quantization processor based on the output format.
    """
    if output_format_val == GGUF:  # GGUF
        return GGUFQuantizationProcessor(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    elif output_format_val == PYTORCH:  # PYTORCH
        # TODO
        pass
    elif output_format_val == ONNX:  # ONNX
        return ONNXQuantizationProcessor(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    elif output_format_val == GPTQ:  # GPTQ
        return GPTQQuantizationProcessor(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    else:
        raise ValueError(f"Unsupported output format: {output_format_val}")


def quantize_and_benchmark_process(input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val, progress, status_update_callback):
    """
    Router function that selects and runs a quantization processor based on the output format.
    """
    processor = create_quantization_processor(output_format_val, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    yield from processor.process()
