import traceback

import gradio as gr

from quantbench.qb_config import get_available_quant_types, SupportedFeatures
from quantbench.qb_manager import start_process
from quantbench.util.ui_util import (update_button_states, periodic_update_output_dir_content, update_input_directory, update_output_directory,
                                     update_output_dir_label)


# TODO It's OK to use this and not pydantic here, as this is not a data model?
class QuantBenchUI:
    def __init__(self):
        self.benchmark_button = None
        self.imatrix_radio = None
        self.input_dir = None
        self.input_dir_button = None
        self.input_dir_content = None
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
        # (self.output_dir and self.output_dir_content will be created in create_ui)
        self.timer = None  # Initialize timer as None

    def create_ui(self):
        with gr.Blocks(title="LLM/VLM Quantization and Benchmarking Tool", theme=gr.themes.Ocean()) as demo:
            gr.Markdown("# ⚛️ QUANTBENCH: LLM/VLM Quantization and Benchmarking")
            with gr.Tabs():
                with gr.Tab("⚙️ Quant Configuration"):
                    with gr.Row():
                        with gr.Column():
                            self.input_dir_button = gr.Button("⤵️ Select Input Directory")
                            self.input_dir = gr.Textbox(
                                label="Model Input Directory (SAFETENSORS)",
                                placeholder="Path to directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.imatrix_radio = gr.Radio(
                                SupportedFeatures.IMATRIX_OPTIONS,  # Use imatrix options here
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
                                label=f"Model Output Directory ({SupportedFeatures.OUTPUT_FORMATS[0]})",
                                placeholder="Path to output directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.output_format_radio = gr.Radio(
                                SupportedFeatures.OUTPUT_FORMATS,  # Use output format options here
                                label="Output Format",
                                value="GGUF",
                                interactive=True,
                            )
                            self.output_dir_content = gr.DataFrame(
                                headers=["Name", "Size"],
                                datatype=["str", "str"],
                                max_height=200,
                                interactive=False,
                            )

                    with gr.Row():
                        # TODO Replace get_available_quant_types[0] with a full precision value based on the current model.
                        self.quant_types = gr.CheckboxGroup(
                            label="Quantization Types", choices=get_available_quant_types(), value=get_available_quant_types()[0]
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
                        headers=SupportedFeatures.RESULTS_TABLE_HEADERS,  # Use headers constant here
                        datatype=["str"] * len(SupportedFeatures.RESULTS_TABLE_HEADERS),
                        label="Benchmark Results",
                        interactive=False,
                        wrap=True,
                        column_widths=SupportedFeatures.RESULTS_TABLE_COLUMNS_WIDTH,  # Use column widths constant here
                    )

            # --- Event Handlers ---
            self.input_dir_button.click(
                fn=update_input_directory,
                inputs=[self.input_dir],
                outputs=[self.input_dir, self.input_dir_content, self.quantize_button],
            )
            self.output_dir_button.click(
                fn=update_output_directory,
                inputs=[self.output_dir],
                outputs=[self.output_dir, self.output_dir_content],
            )
            # Both quantization buttons call the same function.
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
