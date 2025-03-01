import os
import subprocess
import threading
import time
from datetime import timedelta
from typing import List, Tuple, Optional

from integration.data.config import LLAMACPP_WORKING_DIR, LLAMACPP_QUANTIZE_PATH, LLAMACPP_IMATRIX_PATH
from quantbench.errors.quantization_error import QuantizationError
from quantbench.manage.qb_config import GGUF_FILE_EXTENSION, CONVERT_HF_TO_GGUF_SCRIPT, SupportedFeatures, INPUT_MODEL_FILE_EXTENSION, IMATRIX_FILE_NAME, \
    IMATRIX_CUSTOM_FILE_NAME, IMATRIX_DATASET_FILE
from quantbench.manage.state_manager import StateManager, ProcessStatus
from quantbench.services.logging_service import LoggingService


class ProcessService:
    """Service for handling quantization and benchmarking processes."""

    def __init__(self):
        self.state_manager = StateManager.get_instance()
        self.logger = LoggingService.get_instance()

    def get_imatrix_options(self) -> List[str]:
        """Returns the list of available imatrix options."""
        return ["None (Reduced Quality)", "Use Included", "Generate Custom Matrix"]

    def get_output_formats(self) -> List[str]:
        """Returns the list of available output formats."""
        return ["GGUF"]

    def get_results_table_headers(self) -> List[str]:
        """Returns the headers for the results table."""
        return [
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

    def get_results_table_column_widths(self) -> List[str]:
        """Returns the column widths for the results table."""
        return ["16%", "8%", "8%", "8%", "8%", "7%", "9%", "9%", "9%", "9%", "9%"]

    def run_subprocess_command(self, command: List[str]) -> Tuple[int, str]:
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
                self.logger.log_debug(line.strip())
            stream.close()

        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout,))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr,))
        stdout_thread.start()
        stderr_thread.start()
        stdout_thread.join()
        stderr_thread.join()
        process.wait()

        return process.returncode, ''.join(output_lines)

    def get_gguf_file_path(self, output_dir: str, model_filename_base: str, quant_type: str) -> str:
        """Generate path for GGUF file."""
        return os.path.join(output_dir, f"{model_filename_base}-{quant_type}{GGUF_FILE_EXTENSION}")

    def get_time_to_quantize(self, start_time: float, end_time: float) -> str:
        """Calculate time taken for quantization."""
        return str(timedelta(seconds=int(end_time - start_time)))

    def calculate_file_size_and_percentage(self, original_size_bytes: int, quantized_size_bytes: int) -> Tuple[str, str]:
        """Calculates file size in GB and percentage of original size."""
        if original_size_bytes == 0:
            return "0.00 GB", "0.00%"  # Avoid division by zero
        quantized_size_gb = quantized_size_bytes / (1024 ** 3)
        percentage_of_original = (quantized_size_bytes / original_size_bytes) * 100
        return f"{quantized_size_gb:.2f} GB", f"{percentage_of_original:.2f}%"

    def create_benchmark_result(self, ttq: str, size_gb: str, size_percent: str, error: bool = False) -> dict:
        """Create benchmark result dictionary."""
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

    def convert_to_gguf(self, input_dir: str, output_dir: str) -> Tuple[int, str, str]:
        """
        Convert the model to GGUF format.
        Returns: (return_code, output_file_path, console_output)
        """
        model_filename_base = os.path.basename(input_dir)
        output_file_path = self.get_gguf_file_path(output_dir, model_filename_base, SupportedFeatures.HIGHEST_FP_QUANT_TYPE)

        convert_hf_to_gguf_path = os.path.join(LLAMACPP_WORKING_DIR, CONVERT_HF_TO_GGUF_SCRIPT)
        command = ["python", convert_hf_to_gguf_path, input_dir, "--outfile", output_file_path]

        self.logger.log_user(f"Converting model to {SupportedFeatures.HIGHEST_FP_QUANT_TYPE} GGUF format")
        self.logger.log_debug(f"Running command: {' '.join(command)}")

        ret_code, cmd_output = self.run_subprocess_command(command)

        if ret_code == 0:
            self.logger.log_user(f"Successfully converted model to {SupportedFeatures.HIGHEST_FP_QUANT_TYPE} GGUF: {output_file_path}")
        else:
            self.logger.log_error(f"Error converting model to {SupportedFeatures.HIGHEST_FP_QUANT_TYPE} GGUF")

        return ret_code, output_file_path, cmd_output

    def generate_imatrix(self, model_file: str, train_data_file: str, output_file: str) -> Tuple[int, str]:
        """
        Generate importance matrix for the model.
        Returns: (return_code, console_output)
        """
        command = [
            LLAMACPP_IMATRIX_PATH,
            "-m", model_file,
            "-f", train_data_file,
            "-o", output_file,
            "--n-gpu-layers", "9999",  # Based on hardware configuration.
        ]

        self.logger.log_user(f"Generating importance matrix")
        self.logger.log_debug(f"Running command: {' '.join(command)}")

        ret_code, cmd_output = self.run_subprocess_command(command)

        if ret_code == 0:
            self.logger.log_user(f"Successfully generated importance matrix: {output_file}")
        else:
            self.logger.log_error(f"Error generating importance matrix")

        return ret_code, cmd_output

    def quantize_model(self, fp_gguf_path: str, output_file_path: str, quant_type: str, imatrix_path: Optional[str] = None) -> Tuple[int, str]:
        """
        Quantize the model to the specified quantization type.
        Returns: (return_code, console_output)
        """
        command = [LLAMACPP_QUANTIZE_PATH]
        if imatrix_path:
            command.extend(["--imatrix", imatrix_path])
        command.extend([fp_gguf_path, output_file_path, quant_type])

        self.logger.log_user(f"Quantizing model to {quant_type}")
        self.logger.log_debug(f"Running command: {' '.join(command)}")

        ret_code, cmd_output = self.run_subprocess_command(command)

        if ret_code == 0:
            self.logger.log_user(f"Successfully quantized model to {quant_type}: {output_file_path}")
        else:
            self.logger.log_error(f"Error quantizing model to {quant_type}")

        return ret_code, cmd_output

    def validate_inputs(self) -> None:
        """
        Validate inputs before starting the process.
        Raises ValueError if inputs are invalid.
        """
        state = self.state_manager.state

        if not state.input_dir:
            raise ValueError("Please provide input directory")

        if not state.output_dir:
            raise ValueError("Please provide output directory")

        if not state.selected_quant_types:
            # If no quant types selected, we'll just convert to F16/F32
            self.logger.log_warning("No quantization types selected. Will only convert to base format.")

        if not os.path.isdir(state.input_dir):
            raise ValueError(f"Input directory '{state.input_dir}' does not exist or is not a directory")

        # Check for safetensors files
        safetensors_files = [f for f in os.listdir(state.input_dir) if f.endswith(INPUT_MODEL_FILE_EXTENSION)]
        if not safetensors_files:
            raise ValueError("No .safetensors files found in the input directory")

        # Create output directory if it doesn't exist
        if not os.path.isdir(state.output_dir):
            try:
                os.makedirs(state.output_dir, exist_ok=True)
                self.logger.log_user(f"Created output directory: {state.output_dir}")
            except OSError as e:
                raise ValueError(f"Could not create output directory '{state.output_dir}': {e}")

    def prepare_imatrix(self, base_model_path: str) -> Optional[str]:
        """
        Prepare importance matrix based on user selection.
        Returns the path to the imatrix file or None if not applicable.
        """
        state = self.state_manager.state
        imatrix_options = self.get_imatrix_options()

        if state.imatrix_option == imatrix_options[0]:  # None
            return None

        if state.imatrix_option == imatrix_options[1]:  # Use Included
            imatrix_path = os.path.join(state.input_dir, IMATRIX_FILE_NAME)
            if not os.path.exists(imatrix_path):
                raise ValueError("Error: Included imatrix.dat not found in input directory")

            self.logger.log_user(f"Using included importance matrix: {imatrix_path}")
            return imatrix_path

        if state.imatrix_option == imatrix_options[2]:  # Generate Custom Matrix
            imatrix_path = os.path.join(state.output_dir, IMATRIX_CUSTOM_FILE_NAME)

            if not os.path.exists(imatrix_path):
                self.state_manager.update_state(process_status=ProcessStatus.GENERATING_IMATRIX)

                train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMATRIX_DATASET_FILE)
                ret_code, _ = self.generate_imatrix(base_model_path, train_data_path, imatrix_path)

                if ret_code != 0:
                    raise QuantizationError("Error during imatrix generation")

            return imatrix_path

        return None

    def process_quantization_and_benchmark(self, progress=None) -> None:
        """
        Main process function that handles the entire workflow.
        """
        state = self.state_manager.state

        try:
            # Step 1: Validate inputs
            self.validate_inputs()

            # Step 2: Setup variables
            model_filename_base = os.path.basename(state.input_dir)
            safetensors_files = [f for f in os.listdir(state.input_dir) if f.endswith(INPUT_MODEL_FILE_EXTENSION)]
            input_file_path = os.path.join(state.input_dir, safetensors_files[0])

            self.logger.log_user(f"Found safetensors file: {safetensors_files[0]}")

            # Step 3: Convert to base format (F16)
            base_gguf_path = self.get_gguf_file_path(state.output_dir, model_filename_base, SupportedFeatures.HIGHEST_FP_QUANT_TYPE)
            base_size_bytes = 0

            if not os.path.exists(base_gguf_path):
                self.state_manager.update_state(process_status=ProcessStatus.CONVERTING)

                ret_code, base_gguf_path, _ = self.convert_to_gguf(
                    state.input_dir, state.output_dir
                )

                if ret_code != 0:
                    raise QuantizationError("Error during base model conversion")
            else:
                self.logger.log_user(f"Base model already exists: {base_gguf_path}")

            base_size_bytes = os.path.getsize(base_gguf_path)
            self.state_manager.update_state(base_model_path=base_gguf_path)

            # Step 4: Prepare imatrix if needed
            imatrix_path = self.prepare_imatrix(base_gguf_path)
            self.state_manager.update_state(imatrix_path=imatrix_path)

            # Step 5: Process quantization for selected types
            self.state_manager.update_state(process_status=ProcessStatus.QUANTIZING)

            # Handle the case when no quant types are selected - we'll just report the base model
            if not state.selected_quant_types:
                base_size_gb, base_size_percent = self.calculate_file_size_and_percentage(
                    base_size_bytes, base_size_bytes
                )
                benchmark_result = self.create_benchmark_result("N/A", base_size_gb, base_size_percent)

                new_result = [base_gguf_path, SupportedFeatures.HIGHEST_FP_QUANT_TYPE] + list(benchmark_result.values())

                # Update results
                results_data = state.results_data.copy()
                results_data.append(new_result)
                self.state_manager.update_state(results_data=results_data)

                self.logger.log_user("No quantization types selected. Process completed with base model only.")
                self.state_manager.update_state(process_status=ProcessStatus.FINISHED)
                return

            # Add base model to results first
            if SupportedFeatures.HIGHEST_FP_QUANT_TYPE in state.selected_quant_types:
                base_size_gb, base_size_percent = self.calculate_file_size_and_percentage(
                    base_size_bytes, base_size_bytes
                )
                benchmark_result = self.create_benchmark_result("N/A", base_size_gb, base_size_percent)

                new_result = [base_gguf_path, SupportedFeatures.HIGHEST_FP_QUANT_TYPE] + list(benchmark_result.values())

                # Update results
                results_data = state.results_data.copy()
                results_data.append(new_result)
                self.state_manager.update_state(results_data=results_data)

            # Process other quantization types
            quant_types_to_process = [state.selected_quant_types]

            if len(progress.iterables) > 0 and progress:
                quant_types_iter = progress.tqdm(quant_types_to_process, desc="Quantizing models...")
            else:
                quant_types_iter = quant_types_to_process

            for quant_type in quant_types_iter[0]:
                output_file_path = self.get_gguf_file_path(state.output_dir, model_filename_base, quant_type)

                if os.path.exists(output_file_path):
                    self.logger.log_user(f"Skipping {quant_type} (already exists): {output_file_path}")

                    quantized_size_bytes = os.path.getsize(output_file_path)
                    quantized_size_gb, quantized_size_percent = self.calculate_file_size_and_percentage(
                        base_size_bytes, quantized_size_bytes
                    )
                    benchmark_result = self.create_benchmark_result("N/A", quantized_size_gb, quantized_size_percent)

                    new_result = [output_file_path, quant_type] + list(benchmark_result.values())
                else:
                    start_time = time.time()

                    ret_code, _ = self.quantize_model(
                        base_gguf_path, output_file_path, quant_type, imatrix_path
                    )

                    if ret_code == 0:
                        end_time = time.time()
                        ttq = self.get_time_to_quantize(start_time, end_time)

                        quantized_size_bytes = os.path.getsize(output_file_path)
                        quantized_size_gb, quantized_size_percent = self.calculate_file_size_and_percentage(
                            base_size_bytes, quantized_size_bytes
                        )
                        benchmark_result = self.create_benchmark_result(ttq, quantized_size_gb, quantized_size_percent)

                        new_result = [output_file_path, quant_type] + list(benchmark_result.values())
                    else:
                        benchmark_result = self.create_benchmark_result("ERROR", "ERROR", "ERROR", error=True)
                        new_result = [output_file_path, quant_type] + list(benchmark_result.values())
                        raise QuantizationError(f"Error during quantization to {quant_type}")

                # Update results
                results_data = state.results_data.copy()
                results_data.append(new_result)
                self.state_manager.update_state(results_data=results_data)

            # Step 6: Update final state
            self.state_manager.update_state(process_status=ProcessStatus.FINISHED)
            self.logger.log_user("Quantization process completed successfully")

        except Exception as e:
            import traceback
            self.logger.log_error(f"Error in quantization process: {str(e)}", exc_info=True)
            self.state_manager.update_state(
                process_status=ProcessStatus.ERROR,
                error_message=str(e)
            )
            raise
