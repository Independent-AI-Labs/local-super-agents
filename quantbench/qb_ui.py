import os
import traceback

import gradio as gr

from quantbench.qb_config import get_available_quant_types, SupportedFeatures, UNSUPPORTED, SAFETENSORS
from quantbench.qb_manager import start_process
from quantbench.util.convert_util import detect_model_format
from quantbench.util.ui_util import (update_button_states, periodic_update_output_dir_content, update_input_directory, update_output_directory,
                                     update_output_dir_label)


class QuantBenchUI:
    def __init__(self):
        self.benchmark_button = None
        self.imatrix_radio = None
        self.input_dir = None
        self.input_dir_button = None
        self.input_dir_content = None
        self.input_format_display = None
        self.output_console = None
        self.output_dir = None
        self.output_dir_button = None
        self.output_dir_content = None
        self.output_dir_content_data = []
        self.output_format_radio = None
        self.progress = None
        self.quant_types = None
        self.quantize_and_bench_button = None
        self.quantize_button = None
        self.results_table = None
        self.results_table_data = []
        self.status_textbox = None
        self.timer = None  # Initialize timer as None
        self.detected_input_format = None

    def create_ui(self):
        with gr.Blocks(title="LLM/VLM Quantization and Benchmarking Tool", theme=gr.themes.Ocean()) as demo:
            gr.Markdown("# ⚛️ QUANTBENCH: LLM/VLM Quantization and Benchmarking")
            with gr.Tabs():
                with gr.Tab("⚙️ Quant Configuration"):
                    with gr.Row():
                        with gr.Column():
                            self.input_dir_button = gr.Button("⤵️ Select Input Directory")
                            self.input_dir = gr.Textbox(
                                label="Model Input Directory",
                                placeholder="Path to directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.input_format_display = gr.Textbox(
                                label="Detected Input Format",
                                placeholder="Select an input directory to detect format...",
                                interactive=False,
                                value=r"",
                            )
                            self.imatrix_radio = gr.Radio(
                                SupportedFeatures.IMATRIX_OPTIONS,
                                label="Importance Matrix (imatrix.dat)",
                                value=SupportedFeatures.IMATRIX_OPTIONS[0],
                                interactive=True,
                            )
                            self.input_dir_content = gr.DataFrame(
                                headers=["Name", "Size"],
                                datatype=["str", "str"],
                                label=None,
                                max_height=200,
                                interactive=False,
                            )
                        with gr.Column():
                            self.output_dir_button = gr.Button("↗️ Select Output Directory")
                            self.output_dir = gr.Textbox(
                                label=f"Model Output Directory",
                                placeholder="Path to output directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.output_format_radio = gr.Radio(
                                choices=SupportedFeatures.OUTPUT_FORMATS,
                                label="Output Format",
                                interactive=False,  # Disabled until input directory is selected
                            )
                            self.output_dir_content = gr.DataFrame(
                                headers=["Name", "Size"],
                                datatype=["str", "str"],
                                max_height=200,
                                interactive=False,
                            )

                    with gr.Row():
                        self.quant_types = gr.CheckboxGroup(
                            label="Quantization Types", choices=get_available_quant_types(), value=get_available_quant_types()[0],
                        )
                    with gr.Row():
                        with gr.Column():
                            self.quantize_button = gr.Button(" Quantize Model", variant="primary")
                            with gr.Row():
                                self.benchmark_button = gr.Button("💪 Benchmark Quantized", variant="secondary")
                                self.quantize_and_bench_button = gr.Button("⚛️ Quantize and Benchmark", variant="secondary")
                    # Place the status TextBox in its own row below the buttons.
                    with gr.Row():
                        self.progress = gr.Progress(track_tqdm=True)
                        self.status_textbox = gr.Textbox(label="Status", lines=1, interactive=False)
                with gr.Tab("🖥️ Output Console"):
                    self.output_console = gr.Textbox(label="Output Console", lines=37, autoscroll=True)
                with gr.Tab("📊 Benchmark Results"):
                    self.results_table = gr.DataFrame(
                        headers=SupportedFeatures.RESULTS_TABLE_HEADERS,
                        datatype=["str"] * len(SupportedFeatures.RESULTS_TABLE_HEADERS),
                        label="Benchmark Results",
                        interactive=False,
                        wrap=True,
                        column_widths=SupportedFeatures.RESULTS_TABLE_COLUMNS_WIDTH,
                    )

            # --- Event Handlers ---
            self.input_dir_button.click(
                fn=self.update_input_directory_with_format_detection,
                inputs=[self.input_dir],
                outputs=[
                    self.input_dir,
                    self.input_dir_content,
                    self.input_format_display,
                    self.output_format_radio,
                    self.quantize_button
                ],
            )
            self.output_dir_button.click(
                fn=update_output_directory,
                inputs=[self.output_dir],
                outputs=[self.output_dir, self.output_dir_content],
            )

            self.output_format_radio.select(
                fn=update_output_dir_label,
                inputs=[self.output_format_radio],
                outputs=[self.output_dir],
            )

            self.quantize_button.click(
                fn=self.run_quantization,
                inputs=[self.input_dir, self.output_dir, self.quant_types, self.imatrix_radio, self.output_format_radio],
                outputs=[
                    self.quantize_button,
                    self.quantize_and_bench_button,
                    self.output_console,
                    self.results_table,
                    self.status_textbox,
                ],
            )
            self.quantize_and_bench_button.click(
                fn=self.run_quantization,
                inputs=[self.input_dir, self.output_dir, self.quant_types, self.imatrix_radio, self.output_format_radio],
                outputs=[
                    self.quantize_button,
                    self.quantize_and_bench_button,
                    self.output_console,
                    self.results_table,
                    self.status_textbox,
                ],
            )

            self.timer = gr.Timer(value=3, active=True)  # Create Timer instance and store it in self.timer
            self.timer.tick(  # Use .tick event listener on the Timer instance
                fn=periodic_update_output_dir_content,
                inputs=self.output_dir,  # Pass self.output_dir as input
                outputs=self.output_dir_content,
            )

        return demo

    def get_compatible_output_formats(self, input_format):
        """
        Determine compatible output formats based on the detected input format.

        Args:
            input_format (ModelFormat): The detected input format

        Returns:
            list: List of compatible output formats
        """
        if input_format == UNSUPPORTED:
            return []

        # Get compatible formats list
        compatible_formats = SupportedFeatures.CONVERSION_PATHS.get(input_format, [])

        # Add "UNCHANGED" option for all non-SAFETENSORS input formats
        # This represents keeping the format the same, just quantizing
        if input_format != SAFETENSORS:
            compatible_formats = ["UNCHANGED"] + compatible_formats

        return compatible_formats

    def update_input_directory_with_format_detection(self, current_dir_val):
        """
        Update the input directory and detect the model format

        Args:
            current_dir_val (str): Current input directory value

        Returns:
            tuple: Updated values for UI components
        """
        # Use the existing update function to get directory and content
        input_dir, input_dir_content, quantize_button_enabled = update_input_directory(current_dir_val)

        # If a directory was selected, detect the model format
        if input_dir and os.path.isdir(input_dir):
            try:
                self.detected_input_format = detect_model_format(input_dir)
                format_str = self.detected_input_format if self.detected_input_format != "UNSUPPORTED" else "Unsupported"

                # Get compatible output formats based on detected input format
                compatible_output_formats = self.get_compatible_output_formats(self.detected_input_format)

                # Set default output format to first compatible one if any exist
                default_output_format = compatible_output_formats[0] if compatible_output_formats else None

                # Update output format radio (choices and value)
                output_format_update = gr.update(
                    choices=compatible_output_formats,
                    value=default_output_format,
                    interactive=True if compatible_output_formats else False
                )

                return (
                    input_dir,
                    input_dir_content,
                    f"{format_str}",
                    output_format_update,
                    quantize_button_enabled
                )
            except Exception as e:
                print(f"Error detecting model format: {str(e)}")
                return (
                    input_dir,
                    input_dir_content,
                    "Error detecting format",
                    gr.update(choices=[], interactive=False),
                    gr.update(interactive=False)
                )

        # If no directory is selected, reset the format display and disable output format radio
        return (
            input_dir,
            input_dir_content,
            "",
            gr.update(choices=[], interactive=False),
            quantize_button_enabled
        )

    def run_quantization(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val):
        # Immediately disable the buttons and set initial status.
        yield (
            update_button_states(False),  # quantize_button
            update_button_states(False),  # quantize_and_bench_button
            "",
            gr.update(value=self.results_table_data),
            gr.update(
                value=(
                    "Building importance matrix..."
                    if imatrix_val == SupportedFeatures.IMATRIX_OPTIONS[2]
                    else "Processing..."
                )
            ),
        )

        # Handle "UNCHANGED" format option specially
        if output_format_val == "UNCHANGED":
            # Use the detected input format as the output format
            output_format_val = self.detected_input_format

        def status_update_callback(status):
            print(status)

        try:
            console_output = ""
            # Pass the status_update_callback to process_quantization as positional argument
            process_output = start_process(
                input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val, self.progress, status_update_callback, gr
            )

            (
                console_output_process,
                updated_results_table,
                quantize_button_update,
                quantize_and_bench_button_update,
                status_output,
            ) = process_output  # unpack here

            console_output += console_output_process  # append process output to dir creation output

            yield (
                quantize_button_update,
                quantize_and_bench_button_update,
                console_output,
                updated_results_table,
                gr.update(value=status_output),  # Final status from process_quantization (should be "Finished." or error)
            )

        except ValueError as ve:  # Catch validation errors
            error_str_ui = f"Validation Error:\n{str(ve)}"
            yield (
                update_button_states(True),  # Re-enable buttons
                update_button_states(True),
                f"{error_str_ui}\n",  # Output UI exception to console
                gr.update(value=self.results_table_data),
                gr.update(value=f"{error_str_ui}"),  # Update status textbox with UI exception
            )
            gr.Warning(error_str_ui)
        except OSError as ose:  # Catch OS directory creation errors
            error_str_ui = f"OS Error:\n{str(ose)}"
            yield (
                update_button_states(True),  # Re-enable buttons
                update_button_states(True),
                f"{error_str_ui}\n",  # Output UI exception to console
                gr.update(value=self.results_table_data),
                gr.update(value=f"{error_str_ui}"),  # Update status textbox with UI exception
            )
            gr.Warning(error_str_ui)
        except Exception as e_ui:  # Catch any other UI related exceptions
            error_str_ui = f"UI Exception occurred:\n{str(e_ui)}"
            print(traceback.format_exc())  # Print UI traceback to server console
            yield (
                update_button_states(True),  # Re-enable buttons
                update_button_states(True),
                f"{error_str_ui}\n",  # Output UI exception to console
                gr.update(value=self.results_table_data),
                gr.update(value=f"{error_str_ui}"),  # Update status textbox with UI exception
            )
            gr.Warning(error_str_ui)


if __name__ == "__main__":
    ui = QuantBenchUI()
    demo = ui.create_ui()
    demo.queue()
    demo.launch()
