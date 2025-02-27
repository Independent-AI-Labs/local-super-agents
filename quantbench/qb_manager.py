import traceback

from quantbench.processors.quantization import quantize_and_benchmark_process
from quantbench.util.ui_util import validate_inputs, ensure_output_directory_exists, remove_duplicate_results, update_button_states


def start_process(input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val, progress, status_update_callback, gr):
    # Validate inputs using the utility function
    validate_inputs(input_dir_val, output_dir_val, quant_types_val)

    # Ensure output directory exists; exceptions are handled in run_quantization_benchmark
    dir_creation_message = ensure_output_directory_exists(output_dir_val)
    console_output = dir_creation_message
    new_results_data_chunk = []

    try:
        # quantize_and_benchmark_process is now a generator
        process_generator = quantize_and_benchmark_process(
            input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val, progress, status_update_callback
        )
        for output_chunk in process_generator:
            if isinstance(output_chunk, tuple):
                # Assuming tuple is (console_output_chunk, new_results_data_chunk)
                console_output_chunk, new_results_data_chunk = output_chunk
                console_output += console_output_chunk
            else:
                # Assuming it's just a console output chunk
                console_output += output_chunk
        unique_results_data = remove_duplicate_results(new_results_data_chunk)
    except Exception as e:
        error_str = f"An error occurred during quantization:\n{str(e)}"
        print(traceback.format_exc())  # Still print full traceback to server console for debugging
        console_output += f"{error_str}\n"
        status_output = f"{error_str}"  # User friendly error for status
        gr.Warning(error_str)
        return (
            console_output,
            None,
            update_button_states(True),
            update_button_states(True),
            status_output,
        )  # Re-enable buttons and return error status

    gr.Info("Quantization finished!")
    return (
        console_output,
        gr.update(value=unique_results_data),
        update_button_states(True),
        update_button_states(True),
        "Finished.",
    )  # Re-enable buttons, and finished status
