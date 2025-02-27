import os
from typing import Optional

import torch
from safetensors import safe_open

from quantbench.errors.quantization_error import QuantizationError
from quantbench.processors.quantization_processor import QuantizationProcessor
from quantbench.qb_config import SupportedFeatures, INPUT_MODEL_FILE_EXTENSION
from quantbench.util.convert_util import convert_onnx_to_pytorch, convert_tf_to_onnx, convert_to_full_precision_gguf
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

        full_precision_gguf_path = get_fp_model(self.output_dir_val, self.model_filename_base)

        if full_precision_gguf_path is None:
            self.output_console_text += f"Warning: No full precision GGUF file found. Creating a new one.\n"
            yield self.output_console_text, []

            ret_code_fp, fp_conversion_output = convert_to_full_precision_gguf(self.input_dir_val, self.output_dir_val)
            self.output_console_text += fp_conversion_output
            yield self.output_console_text, []

            if ret_code_fp != 0:
                raise QuantizationError("Error during full-precision conversion. See console output for details.\n")

        full_precision_gguf_path = get_fp_model(self.output_dir_val, self.model_filename_base)

        # imatrix logic
        imatrix_path = yield from self._handle_imatrix(full_precision_gguf_path)

        quant_types_to_process = self.valid_quant_types_val[:]

        for quant_type in self.progress.tqdm(quant_types_to_process, desc="Quantizing models..."):
            yield from self._process_quant_type(quant_type, full_precision_gguf_path, imatrix_path)

        yield self.output_console_text, self.new_results_data


class ONNXQuantizationProcessor(QuantizationProcessor):
    def __init__(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
        super().__init__(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)

    def process(self):
        yield from self._validate_inputs()
        onnx_path = os.path.join(self.output_dir_val, f"{self.model_filename_base}.onnx")

        # Convert to ONNX
        ret_code_onnx, onnx_conversion_output = convert_tf_to_onnx(self.input_dir_val, onnx_path)
        self.output_console_text += onnx_conversion_output
        yield self.output_console_text, []
        if ret_code_onnx != 0:
            raise QuantizationError("Error during ONNX conversion. See console output for details.\n")

        self.output_console_text += f"Model converted to ONNX: {onnx_path}\n"
        yield self.output_console_text, []

        # Convert ONNX to Pytorch
        pytorch_path = os.path.join(self.output_dir_val, f"{self.model_filename_base}.pt")
        ret_code_pytorch, pytorch_conversion_output = convert_onnx_to_pytorch(onnx_path, pytorch_path)
        self.output_console_text += pytorch_conversion_output
        yield self.output_console_text, []
        if ret_code_pytorch != 0:
            raise QuantizationError("Error during ONNX to Pytorch conversion. See console output for details.\n")
        self.output_console_text += f"Model converted to Pytorch: {pytorch_path}\n"
        yield self.output_console_text, []
        yield self.output_console_text, self.new_results_data


class GPTQQuantizationProcessor(ONNXQuantizationProcessor):
    def __init__(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
        super().__init__(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
        self.dataset_representative_data = []

    def process(self):
        yield from super().process()
        pytorch_path = os.path.join(self.output_dir_val, f"{self.model_filename_base}.pt")
        # Generate dummy data, replace with real logic
        # dataset_representative_data = [torch.randn(5) for _ in range(10)]  # Change shape and data type as needed
        yield from self._create_dataset_representative_data()
        try:
            quantized_model = quantize_pytorch_gptq(pytorch_path, self.dataset_representative_data)
        except ValueError as e:
            raise QuantizationError(str(e)) from e

        self.output_console_text += "Model quantized using GPTQ\n"
        yield self.output_console_text, []

        # save the model
        quantized_model_path = os.path.join(self.output_dir_val, f"{self.model_filename_base}-gptq.pt")
        torch.save(quantized_model, quantized_model_path)

        self.output_console_text += f"Quantized model saved to: {quantized_model_path}\n"
        yield self.output_console_text, []

        # Placeholder for benchmarking
        # TODO: Add benchmarking logic here
        quantized_size_bytes = os.path.getsize(quantized_model_path)

        quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(self.fp_size_bytes, quantized_size_bytes)  # need the full-precision size

        benchmark_result = create_benchmark_result("N/A", quantized_size_gb, quantized_size_percent)

        self.new_results_data.append([quantized_model_path, "GPTQ"] + list(benchmark_result.values()))
        yield self.output_console_text, self.new_results_data

    def _create_dataset_representative_data(self):
        """
        Create a dataset representative data using the first tensor of the safetensors
        """
        model_path = None
        # Load the safetensors model to get the input shape
        for filename in os.listdir(self.input_dir_val):
            if filename.endswith(INPUT_MODEL_FILE_EXTENSION):
                model_path = os.path.join(self.input_dir_val, filename)
                break
        if model_path is None:
            raise ValueError("No .safetensors files found in the input directory")

        with safe_open(model_path, framework="pt", device="cpu") as f:
            first_tensor_name = next(iter(f.keys()))
            tensor_info = f.get_tensor(first_tensor_name)
            input_shape = tensor_info.shape[1:]
        self.output_console_text += f"Tensor shape found: {input_shape}\n"
        yield self.output_console_text, []
        self.dataset_representative_data = [torch.randn(input_shape) for _ in range(10)]


def create_quantization_processor(output_format_val, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
    """
    Factory function to create a quantization processor based on the output format.
    """
    if output_format_val == SupportedFeatures.OUTPUT_FORMATS[0]:  # GGUF
        return GGUFQuantizationProcessor(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    elif output_format_val == SupportedFeatures.OUTPUT_FORMATS[1]:  # ONNX
        return ONNXQuantizationProcessor(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    elif output_format_val == SupportedFeatures.OUTPUT_FORMATS[2]:  # GPTQ
        return GPTQQuantizationProcessor(input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    else:
        raise ValueError(f"Unsupported output format: {output_format_val}")


def quantize_and_benchmark_process(input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val, progress, status_update_callback):
    """
    Router function that selects and runs a quantization processor based on the output format.
    """
    processor = create_quantization_processor(output_format_val, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback)
    yield from processor.process()
