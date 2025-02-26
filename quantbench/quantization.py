import os
import time

from quantbench.errors.quantization_error import QuantizationError
from quantbench.util.quant_util import get_available_quant_types, _get_gguf_file_path, _convert_to_f16_gguf, _generate_imatrix, \
    _get_time_to_quantize, calculate_file_size_and_percentage, _create_benchmark_result, _quantize_model, MODEL_FILE_EXTENSION, F16_QUANT_TYPE, \
    IMATRIX_DATASET_FILE, IMATRIX_FILE_NAME, _handle_error, SupportedFeatures, IMATRIX_CUSTOM_FILE_NAME


def quantize_and_benchmark_process(input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val, progress, status_update_callback):
    """
    Generator that quantizes and benchmarks a model for different quantization types.
    It yields updates (console_text, results_data) so that the UI can be updated live.
    During intermediate updates, results_data is None.
    """
    output_console_text = ""
    new_results_data = []
    available_quants = get_available_quant_types()
    valid_quant_types_val = [q for q in quant_types_val if q in available_quants]

    # Input validation
    if not input_dir_val:
        yield from _handle_error(output_console_text, "Error: Please provide input directory.\n")
    if not output_dir_val:
        yield from _handle_error(output_console_text, "Error: Please provide output directory.\n")
    if not valid_quant_types_val:
        yield from _handle_error(output_console_text, "Error: Please select at least one quantization type.\n")
    if not os.path.isdir(input_dir_val):
        yield from _handle_error(output_console_text, f"Error: Input directory '{input_dir_val}' does not exist or is not a directory.\n")
    if not os.path.isdir(output_dir_val):
        try:
            os.makedirs(output_dir_val, exist_ok=True)
            output_console_text += f"Created output directory: {output_dir_val}\n"
            yield output_console_text, []
        except OSError as e:
            yield from _handle_error(output_console_text, f"Error: Could not create output directory '{output_dir_val}': {e}\n")
    safetensors_files = [f for f in os.listdir(input_dir_val) if f.endswith(MODEL_FILE_EXTENSION)]
    if not safetensors_files:
        output_console_text += "Error: No .safetensors files found in the input directory.\n"
        yield output_console_text, []
        raise QuantizationError("Error: No .safetensors files found in the input directory.\n")

    model_filename_base = os.path.split(input_dir_val)[-1]
    input_file_path = os.path.join(input_dir_val, safetensors_files[0])
    output_console_text += f"Found safetensors file: {safetensors_files[0]}\n"
    yield output_console_text, []
    original_size_bytes = os.path.getsize(input_file_path)

    # --- F16 Conversion ---
    f16_gguf_path = _get_gguf_file_path(output_dir_val, model_filename_base, F16_QUANT_TYPE)
    f16_size_bytes = 0

    quant_types_to_process = valid_quant_types_val[:]
    if F16_QUANT_TYPE in quant_types_to_process:
        quant_types_to_process.remove(F16_QUANT_TYPE)
        quant_types_to_process.insert(0, F16_QUANT_TYPE)

    # imatrix logic
    imatrix_path = None
    if imatrix_val != SupportedFeatures.IMATRIX_OPTIONS[0]:
        train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMATRIX_DATASET_FILE)
        if imatrix_val == SupportedFeatures.IMATRIX_OPTIONS[2]:
            imatrix_path = os.path.join(output_dir_val, IMATRIX_CUSTOM_FILE_NAME)
            if not os.path.exists(imatrix_path):
                if not os.path.exists(f16_gguf_path):
                    ret_code_f16, f16_conversion_output = _convert_to_f16_gguf(input_dir_val, output_dir_val)
                    output_console_text += f16_conversion_output
                    yield output_console_text, []
                    if ret_code_f16 != 0:
                        raise QuantizationError("Error during F16 conversion. See console output for details.\n")
                ret_code_imatrix, imatrix_output = _generate_imatrix(f16_gguf_path, train_data_path, imatrix_path)
                output_console_text += imatrix_output
                yield output_console_text, []
                if ret_code_imatrix != 0:
                    raise QuantizationError("Error during imatrix generation. See console output for details.\n")
                else:
                    output_console_text += f"Imatrix generated: {imatrix_path}\n"
                    yield output_console_text, []
        elif imatrix_val == SupportedFeatures.IMATRIX_OPTIONS[1]:
            imatrix_path = os.path.join(input_dir_val, IMATRIX_FILE_NAME)
            if not os.path.exists(imatrix_path):
                raise QuantizationError("Error: Included imatrix.dat not found in input directory.\n")
            output_console_text += f"Using included imatrix: {imatrix_path}\n"
            yield output_console_text, []

    for quant_type in progress.tqdm(quant_types_to_process, desc="Quantizing models..."):
        if quant_type == F16_QUANT_TYPE:
            f16_start_time = time.time()
            if not os.path.exists(f16_gguf_path):
                ret_code_f16, f16_conversion_output = _convert_to_f16_gguf(input_dir_val, output_dir_val)
                output_console_text += f16_conversion_output
                yield output_console_text, []
                if ret_code_f16 != 0:
                    raise QuantizationError("Error during F16 conversion. See console output for details.\n")
            else:
                output_console_text += f"F16 GGUF already exists: {f16_gguf_path}\n"
                yield output_console_text, []

            f16_end_time = time.time()
            ttq_f16 = _get_time_to_quantize(f16_start_time, f16_end_time)
            f16_size_bytes = os.path.getsize(f16_gguf_path)
            f16_size_gb, f16_size_percent = calculate_file_size_and_percentage(f16_size_bytes, f16_size_bytes)
            benchmark_result_f16 = _create_benchmark_result(ttq_f16, f16_size_gb, f16_size_percent)
            new_results_data.append([f16_gguf_path, F16_QUANT_TYPE] + list(benchmark_result_f16.values()))
        else:
            output_file_path = _get_gguf_file_path(output_dir_val, model_filename_base, quant_type)
            if os.path.exists(output_file_path):
                quantized_size_bytes = os.path.getsize(output_file_path)
                quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(f16_size_bytes, quantized_size_bytes)
                benchmark_result = _create_benchmark_result("N/A", quantized_size_gb, quantized_size_percent)
                output_console_text += f"Skipping {quant_type} (already exists): {output_file_path}\n"
                yield output_console_text, []
                new_results_data.append([output_file_path, quant_type] + list(benchmark_result.values()))
                continue
            start_time = time.time()
            ret_code, quantization_output = _quantize_model(f16_gguf_path, output_file_path, quant_type, imatrix_path)
            output_console_text += quantization_output
            yield output_console_text, []
            if ret_code == 0:
                end_time = time.time()
                ttq = _get_time_to_quantize(start_time, end_time)
                quantized_size_bytes = os.path.getsize(output_file_path)
                quantized_size_gb, quantized_size_percent = calculate_file_size_and_percentage(f16_size_bytes, quantized_size_bytes)
                benchmark_result = _create_benchmark_result(ttq, quantized_size_gb, quantized_size_percent)
                new_results_data.append([output_file_path, quant_type] + list(benchmark_result.values()))
            else:
                new_results_data.append([output_file_path, quant_type] + list(_create_benchmark_result("ERROR", "ERROR", "ERROR", error=True).values()))
                raise QuantizationError("Unspecified error during quantization.\n")

    yield output_console_text, new_results_data
