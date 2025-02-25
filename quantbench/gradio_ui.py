import os
import traceback

import gradio as gr

from integration.util.misc_util import select_and_list_directory_contents, list_directory_contents
from quantbench.quant_util import get_available_quant_types
from quantbench.quantization import quantize_and_benchmark_process


class QuantBenchUI:
    def __init__(self):
        self.results_table_data = []
        # Shared variable to store the latest output directory contents
        self.output_dir_content_data = []
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
                                label="Model Input Directory (.safetensors)",
                                placeholder="Path to directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.input_dir_content = gr.DataFrame(
                                headers=["Name", "Size"],
                                datatype=["str", "str"],
                                label="Input Directory Content",
                                max_height=200,
                                interactive=False,
                            )
                        with gr.Column():
                            self.output_dir_button = gr.Button("↗️ Select Output Directory")
                            self.output_dir = gr.Textbox(
                                label="Model Output Directory (.gguf)",
                                placeholder="Path to output directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.output_dir_content = gr.DataFrame(
                                headers=["Name", "Size"],
                                datatype=["str", "str"],
                                label="Output Directory Content",
                                max_height=200,
                                interactive=False,
                            )
                    with gr.Row():
                        self.quant_types = gr.CheckboxGroup(
                            label="Quantization Types", choices=get_available_quant_types(), value=["F16"]
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
                    self.output_console = gr.Textbox(label="Output Console", lines=35, autoscroll=True)
                with gr.Tab("📊 Benchmark Results"):
                    self.results_table = gr.DataFrame(
                        headers=[
                            "Model File Path",
                            "Type",
                            "TTQ",
                            "Size",
                            "% of F16",
                            "Test Runs",
                            "Load (s)",
                            "Encode (s/mpx)",
                            "Prompt (t/s)",
                            "Resp. (t/s)",
                            "Quality (%)",
                        ],
                        datatype=["str"] * 11,
                        label="Benchmark Results",
                        interactive=False,
                        wrap=True,
                        column_widths=["16%", "8%", "8%", "8%", "8%", "7%", "9%", "9%", "9%", "9%", "9%"],
                    )

            # --- Event Handlers ---
            self.input_dir_button.click(
                fn=self.update_input_directory,
                inputs=[],
                outputs=[self.input_dir, self.input_dir_content, self.quantize_button],
            )
            self.output_dir_button.click(
                fn=self.update_output_directory,
                inputs=[],
                outputs=[self.output_dir, self.output_dir_content],
            )
            # Both quantization buttons call the same function.
            self.quantize_button.click(
                fn=self.run_quantization,
                inputs=[self.input_dir, self.output_dir, self.quant_types],
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
                inputs=[self.input_dir, self.output_dir, self.quant_types],
                outputs=[
                    self.quantize_button,
                    self.quantize_and_bench_button,
                    self.output_console,
                    self.results_table,
                    self.status_textbox,
                ],
            )

            self.timer = gr.Timer(value=5, active=True)  # Create Timer instance and store it in self.timer
            self.timer.tick(  # Use .tick event listener on the Timer instance
                fn=self.periodic_update_output_dir_content,
                inputs=self.output_dir,  # Pass self.output_dir as input
                outputs=self.output_dir_content,
            )

        return demo

    def update_input_directory(self):
        selected_dir, dir_content = select_and_list_directory_contents(self.input_dir.value)
        dir_name = os.path.split(selected_dir)[-1]
        return selected_dir, gr.update(value=dir_content), gr.update(value=f"Quantize {dir_name}")

    def update_output_directory(self):
        selected_dir, dir_content = select_and_list_directory_contents(self.output_dir.value)
        # Also update the shared variable so that the background thread’s data
        # and the UI remain in sync (if you choose to poll it elsewhere)
        self.output_dir_content_data = dir_content
        return selected_dir, gr.update(value=dir_content)

    def periodic_update_output_dir_content(self, output_dir):  # Receive output_dir_component as argument
        if output_dir:  # Use local variable output_dir_val
            try:
                contents = list_directory_contents(output_dir)  # Use output_dir_val
                self.output_dir_content_data = contents  # Keep this updated if used elsewhere
                return gr.update(value=contents)
            except Exception as e:
                print(f"Error updating output directory content: {e}")
                return gr.update()  # Return empty update to prevent errors in Gradio
        else:
            return gr.update()  # Return empty update if output_dir not initialized yet

    def validate_inputs(self, input_dir_val, output_dir_val, quant_types_val):
        if not input_dir_val or not output_dir_val or not quant_types_val:
            raise ValueError(
                "Please provide input directory, output directory, and select at least one quantization type."
            )
        if not os.path.isdir(input_dir_val):
            raise ValueError(f"Input directory '{input_dir_val}' does not exist or is not a directory.")

    def ensure_output_directory_exists(self, output_dir_val):
        if not os.path.isdir(output_dir_val):
            try:
                os.makedirs(output_dir_val, exist_ok=True)
                return f"Created output directory: {output_dir_val}\n"
            except OSError as e:
                raise OSError(f"Could not create output directory '{output_dir_val}': {e}")
        return ""

    def update_button_states(self, interactive_value):
        return gr.update(interactive=interactive_value)

    def process_quantization(self, input_dir_val, output_dir_val, quant_types_val, status_update_callback):
        # Validate inputs; if there is an error, it will be raised as an exception.
        self.validate_inputs(input_dir_val, output_dir_val, quant_types_val)

        # Ensure output directory exists; exceptions are handled in run_quantization_benchmark
        dir_creation_message = self.ensure_output_directory_exists(output_dir_val)
        console_output = dir_creation_message

        try:
            # Pass status_update_callback to quantize_and_benchmark_process
            console_output_process, new_results_data = quantize_and_benchmark_process(
                input_dir_val, output_dir_val, quant_types_val, self.output_console, self.progress, status_update_callback
            )
            console_output += console_output_process
            self.results_table_data.extend(new_results_data)
            unique_results_data = self.remove_duplicate_results(self.results_table_data)
        except Exception as e:
            error_str = f"An error occurred during quantization:\n{str(e)}"
            print(traceback.format_exc())  # Still print full traceback to server console for debugging
            console_output += f"{error_str}\n"
            status_output = f"{error_str}"  # User friendly error for status
            return (
                console_output,
                gr.update(value=self.results_table_data),
                self.update_button_states(True),
                self.update_button_states(True),
                status_output,
            )  # Re-enable buttons and return error status

        return (
            console_output,
            gr.update(value=unique_results_data),
            self.update_button_states(True),
            self.update_button_states(True),
            "Finished.",
        )  # Re-enable buttons, and finished status

    def remove_duplicate_results(self, all_results_data):
        seen_paths = set()
        unique_results_data = []
        for row in reversed(all_results_data):
            quant_path = row[0]  # Assume quant path is always the first element in the list-row
            if quant_path not in seen_paths:
                unique_results_data.insert(0, row)
                seen_paths.add(quant_path)
        return unique_results_data

    def run_quantization(self, input_dir_val, output_dir_val, quant_types_val):
        # Immediately disable the buttons and set initial status.
        yield (
            self.update_button_states(False),
            self.update_button_states(False),
            "",
            gr.update(value=self.results_table_data),
            gr.update(value="Processing..."),
        )

        # TODO Why is this not called? - It is a callback function, it should be called inside `process_quantization` or `quantize_and_benchmark_process` if passed correctly.
        def status_update_callback(status_message):
            print(status_message)
            self.update_output_directory()  # Although this is called, it is not returning and updating UI in `run_quantization` generator.
            yield gr.update(value=status_message)  # This yield here is in wrong context, it should be called within `run_quantization`

        try:
            console_output = ""

            # Pass the status_update_callback to process_quantization as positional argument
            process_output = self.process_quantization(
                input_dir_val,
                output_dir_val,
                quant_types_val,
                lambda status: status_update_callback(status),  # Pass lambda for callback as positional argument
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
                self.update_button_states(True),  # Re-enable buttons
                self.update_button_states(True),
                f"{error_str_ui}\n",  # Output UI exception to console
                gr.update(value=self.results_table_data),
                gr.update(value=f"{error_str_ui}"),  # Update status textbox with UI exception
            )
        except OSError as ose:  # Catch OS directory creation errors
            error_str_ui = f"OS Error:\n{str(ose)}"
            yield (
                self.update_button_states(True),  # Re-enable buttons
                self.update_button_states(True),
                f"{error_str_ui}\n",  # Output UI exception to console
                gr.update(value=self.results_table_data),
                gr.update(value=f"{error_str_ui}"),  # Update status textbox with UI exception
            )
        except Exception as e_ui:  # Catch any other UI related exceptions
            error_str_ui = f"UI Exception occurred:\n{str(e_ui)}"
            print(traceback.format_exc())  # Print UI traceback to server console
            yield (
                self.update_button_states(True),  # Re-enable buttons
                self.update_button_states(True),
                f"{error_str_ui}\n",  # Output UI exception to console
                gr.update(value=self.results_table_data),
                gr.update(value=f"{error_str_ui}"),  # Update status textbox with UI exception
            )


if __name__ == "__main__":
    ui = QuantBenchUI()
    demo = ui.create_ui()
    demo.queue()
    demo.launch()
