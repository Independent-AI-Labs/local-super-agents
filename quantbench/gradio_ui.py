import os
import traceback

import gradio as gr

from integration.util.misc_util import get_folder_path
from quantbench.quant_util import get_available_quant_types
from quantbench.quantization import quantize_and_benchmark_process


class QuantBenchUI:
    def __init__(self):
        self.results_table_data = []

    def create_ui(self):
        with gr.Blocks(title="LLM/VLM Quantization and Benchmarking Tool", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# ⚛️ QUANTBENCH: LLM/VLM Quantization and Benchmarking")
            with gr.Tabs():
                with gr.Tab("Quant Configuration"):
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
                            self.quantize_button = gr.Button("Quantize", variant="primary")
                            self.quantize_and_bench_button = gr.Button("Quantize and Benchmark", variant="secondary")
                    # Place the progress bar in its own row below the buttons.
                    with gr.Row():
                        # TODO Replace this with another component that can visualize the progress, as progress is a state of components in gradio, not a component itself.
                        self.progress = gr.Progress(track_tqdm=True)
                with gr.Tab("Output Console"):
                    self.output_console = gr.Textbox(label="Output Console", lines=30, autoscroll=True)
                with gr.Tab("Benchmark Results"):
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
                outputs=[self.input_dir, self.input_dir_content],
            )
            self.output_dir_button.click(
                fn=self.update_output_directory,
                inputs=[],
                outputs=[self.output_dir, self.output_dir_content],
            )
            # Both quantization buttons call the same function.
            self.quantize_button.click(
                fn=self.run_quantization_benchmark,
                inputs=[self.input_dir, self.output_dir, self.quant_types],
                outputs=[self.quantize_button, self.quantize_and_bench_button, self.output_console, self.results_table],
            )
            self.quantize_and_bench_button.click(
                fn=self.run_quantization_benchmark,
                inputs=[self.input_dir, self.output_dir, self.quant_types],
                outputs=[self.quantize_button, self.quantize_and_bench_button, self.output_console, self.results_table],
            )
        return demo

    def update_input_directory(self):
        selected_dir, dir_content = get_folder_path(self.input_dir.value)
        return selected_dir, gr.update(value=dir_content)

    def update_output_directory(self):
        selected_dir, dir_content = get_folder_path(self.output_dir.value)
        return selected_dir, gr.update(value=dir_content)

    def validate_inputs(self, input_dir_val, output_dir_val, quant_types_val):
        if not input_dir_val or not output_dir_val or not quant_types_val:
            return "Error: Please provide input directory, output directory, and select at least one quantization type.\n"
        if not os.path.isdir(input_dir_val):
            return f"Error: Input directory '{input_dir_val}' does not exist or is not a directory.\n"
        return None

    def ensure_output_directory_exists(self, output_dir_val):
        if not os.path.isdir(output_dir_val):
            try:
                os.makedirs(output_dir_val, exist_ok=True)
                return f"Created output directory: {output_dir_val}\n"
            except OSError as e:
                return f"Error: Could not create output directory '{output_dir_val}': {e}\n"
        return ""

    def update_button_states(self, interactive_value):
        return gr.update(interactive=interactive_value)

    def process_quantization(self, input_dir_val, output_dir_val, quant_types_val):
        # Validate inputs; if there is an error, raise a gr.Error so it appears as a floating error message.
        error_message = self.validate_inputs(input_dir_val, output_dir_val, quant_types_val)
        if error_message:
            raise gr.Error(error_message)
        dir_creation_message = self.ensure_output_directory_exists(output_dir_val)
        if "Error" in dir_creation_message:
            raise gr.Error(dir_creation_message)
        console_output = dir_creation_message

        try:
            console_output_process, new_results_data = quantize_and_benchmark_process(
                input_dir_val, output_dir_val, quant_types_val, self.output_console, self.progress
            )
            console_output += console_output_process
            self.results_table_data.extend(new_results_data)
            unique_results_data = self.remove_duplicate_results(self.results_table_data)
        except Exception as e:
            print(traceback.format_exc())
            raise gr.Error(f"An error occurred during quantization:\n{str(e)}")

        return console_output, gr.update(value=unique_results_data), self.update_button_states(True), self.update_button_states(True)

    def remove_duplicate_results(self, all_results_data):
        seen_paths = set()
        unique_results_data = []
        for row in reversed(all_results_data):
            quant_path = row[0]  # Assume quant path is always the first element in the list-row
            if quant_path not in seen_paths:
                unique_results_data.insert(0, row)
                seen_paths.add(quant_path)
        return unique_results_data

    def run_quantization_benchmark(self, input_dir_val, output_dir_val, quant_types_val):
        # Immediately disable the buttons.
        yield (
            self.update_button_states(False),
            self.update_button_states(False),
            "",
            gr.update(value=self.results_table_data),
        )
        try:
            console_output, updated_results_table, quantize_button_update, quantize_and_bench_button_update = self.process_quantization(
                input_dir_val, output_dir_val, quant_types_val
            )
            yield (
                quantize_button_update,
                quantize_and_bench_button_update,
                console_output,
                updated_results_table,
            )
        except Exception as e:
            # Re-enable buttons if an error occurs.
            yield (
                self.update_button_states(True),
                self.update_button_states(True),
                "",
                gr.update(value=self.results_table_data),
            )
            raise


if __name__ == "__main__":
    ui = QuantBenchUI()
    demo = ui.create_ui()
    demo.queue()  # Enables asynchronous execution of the callbacks.
    demo.launch()
