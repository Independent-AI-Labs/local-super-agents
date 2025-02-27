import os
import time
from abc import ABC, abstractmethod

from quantbench.errors.quantization_error import QuantizationError
from quantbench.qb_config import get_available_quant_types, INPUT_MODEL_FILE_EXTENSION, IMATRIX_DATASET_FILE, SupportedFeatures, IMATRIX_CUSTOM_FILE_NAME, \
    FP16_QUANT_TYPE, IMATRIX_FILE_NAME
from quantbench.util.convert_util import convert_to_f16_gguf
from quantbench.util.quant_util import generate_imatrix, get_time_to_quantize, create_benchmark_result, get_gguf_file_path, quantize_model
from quantbench.util.ui_util import handle_error, calculate_file_size_and_percentage


class QuantizationProcessor(ABC):
    """
    Abstract base class for quantization processors.
    """

    def __init__(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, progress, status_update_callback):
        self.input_dir_val = input_dir_val
        self.output_dir_val = output_dir_val
        self.quant_types_val = quant_types_val
        self.imatrix_val = imatrix_val
        self.progress = progress
        self.status_update_callback = status_update_callback
        self.output_console_text = ""
        self.new_results_data = []
        self.available_quants = get_available_quant_types()
        self.valid_quant_types_val = [q for q in self.quant_types_val if q in self.available_quants]
        self.model_filename_base = os.path.split(self.input_dir_val)[-1]
        self.original_size_bytes = 0
        self.f16_size_bytes = 0

    @abstractmethod
    def process(self):
        """
        Abstract method to process quantization.
        """
        pass

    def _validate_inputs(self):
        if not self.input_dir_val:
            yield from handle_error(self.output_console_text, "Error: Please provide input directory.\n")
        if not self.output_dir_val:
            yield from handle_error(self.output_console_text, "Error: Please provide output directory.\n")
        if not self.valid_quant_types_val:
            yield from handle_error(self.output_console_text, "Error: Please select at least one quantization type.\n")
        if not os.path.isdir(self.input_dir_val):
            yield from handle_error(self.output_console_text, f"Error: Input directory '{self.input_dir_val}' does not exist or is not a directory.\n")
        if not os.path.isdir(self.output_dir_val):
            try:
                os.makedirs(self.output_dir_val, exist_ok=True)
                self.output_console_text += f"Created output directory: {self.output_dir_val}\n"
                yield self.output_console_text, []
            except OSError as e:
                yield from handle_error(self.output_console_text, f"Error: Could not create output directory '{self.output_dir_val}': {e}\n")

        safetensors_files = [f for f in os.listdir(self.input_dir_val) if f.endswith(INPUT_MODEL_FILE_EXTENSION)]
        if not safetensors_files:
            self.output_console_text += "Error: No .safetensors files found in the input directory.\n"
            yield self.output_console_text, []
            raise QuantizationError("Error: No .safetensors files found in the input directory.\n")

        self.input_file_path = os.path.join(self.input_dir_val, safetensors_files[0])
        self.output_console_text += f"Found safetensors file: {safetensors_files[0]}\n"
        yield self.output_console_text, []
        self.original_size_bytes = os.path.getsize(self.input_file_path)

    def _handle_imatrix(self, f16_gguf_path):
        imatrix_path = None
        if self.imatrix_val != SupportedFeatures.IMATRIX_OPTIONS[0]:
            train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMATRIX_DATASET_FILE)
            if self.imatrix_val == SupportedFeatures.IMATRIX_OPTIONS[2]:
                imatrix_path = os.path.join(self.output_dir_val, IMATRIX_CUSTOM_FILE_NAME)
                if not os.path.exists(imatrix_path):
                    if not os.path.exists(f16_gguf_path):
                        ret_code_f16, f16_conversion_output = convert_to_f16_gguf(self.input_dir_val, self.output_dir_val)
                        self.output_console_text += f16_conversion_output
                        yield self.output_console_text, []
                        if ret_code_f16 != 0:
                            raise QuantizationError("Error during F16 conversion. See console output for details.\n")
                    ret_code_imatrix, imatrix_output = generate_imatrix(f16_gguf_path, train_data_path, imatrix_path)
                    self.output_console_text += imatrix_output
                    yield self.output_console_text, []
                    if ret_code_imatrix != 0:
                        raise QuantizationError("Error during imatrix generation. See console output for details.\n")
                    else:
                        self.output_console_text += f"Imatrix generated: {imatrix_path}\n"
                        yield self.output_console_text, []
            elif self.imatrix_val == SupportedFeatures.IMATRIX_OPTIONS[1]:
                imatrix_path = os.path.join(self.input_dir_val, IMATRIX_FILE_NAME)
                if not os.path.exists(imatrix_path):
                    raise QuantizationError("Error: Included imatrix.dat not found in input directory.\n")
                self.output_console_text += f"Using included imatrix: {imatrix_path}\n"
                yield self.output_console_text, []
        return imatrix_path

    def _process_quant_type(self, quant_type, f16_gguf_path, imatrix_path):
        if quant_type == FP16_QUANT_TYPE:
            f16_start_time = time.time()
            if not os.path.exists(f16_gguf_path):
                ret_code_f16, f16_conversion_output = convert_to_f16_gguf(self.input_dir_val, self.output_dir_val)
                self.output_console_text += f16_conversion_output
                yield self.output_console_text, []
                if ret_code_f16 != 0:
                    raise QuantizationError("Error during F16 conversion. See console output for details.\n")
            else:
                self.output_console_text += f"F16 GGUF already exists: {f16_gguf_path}\n"
                yield self.output_console_text, []

            f16_end_time = time.time()
            ttq_f16 = get_time_to_quantize(f16_start_time, f16_end_time)
            self.f16_size_bytes = os.path.getsize(f16_gguf_path)
            f16_size_gb, f16_size_percent = calculate_file_size_and_percentage(self.f16_size_bytes, self.f16_size_bytes)
            benchmark_result_f16 = create_benchmark_result(ttq_f16, f16_size_gb, f16_size_percent)
            self.new_results_data.append([f16_gguf_path, FP16_QUANT_TYPE] + list(benchmark_result_f16.values()))
        else:
            output_file_path = get_gguf_file_path(self.output_dir_val, self.model_filename_base, quant_type)
            if os.path.exists(output_file_path):
                quantized_size_bytes = os.path.getsize(output_file_path)
                quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(self.f16_size_bytes, quantized_size_bytes)
                benchmark_result = create_benchmark_result("N/A", quantized_size_gb, quantized_size_percent)
                self.output_console_text += f"Skipping {quant_type} (already exists): {output_file_path}\n"
                yield self.output_console_text, []
                self.new_results_data.append([output_file_path, quant_type] + list(benchmark_result.values()))
                return
            start_time = time.time()
            ret_code, quantization_output = quantize_model(f16_gguf_path, output_file_path, quant_type, imatrix_path)
            self.output_console_text += quantization_output
            yield self.output_console_text, []
            if ret_code == 0:
                end_time = time.time()
                ttq = get_time_to_quantize(start_time, end_time)
                quantized_size_bytes = os.path.getsize(output_file_path)
                quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(self.f16_size_bytes, quantized_size_bytes)
                benchmark_result = create_benchmark_result(ttq, quantized_size_gb, quantized_size_percent)
                self.new_results_data.append([output_file_path, quant_type] + list(benchmark_result.values()))
            else:
                self.new_results_data.append([output_file_path, quant_type] + list(create_benchmark_result("ERROR", "ERROR", "ERROR", error=True).values()))
                raise QuantizationError("Unspecified error during quantization.\n")

