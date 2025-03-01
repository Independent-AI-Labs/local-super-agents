import os
import threading
import time

import gradio as gr

from integration.data.config import DEBUG_LOG
from integration.util.misc_util import select_and_list_directory_contents, list_directory_contents
from quantbench.manage.qb_config import get_available_quant_types
from quantbench.manage.state_manager import StateManager, ProcessStatus
from quantbench.services.logging_service import LoggingService
from quantbench.services.process_service import ProcessService


class QuantBenchUI:
    def __init__(self):
        self.state_manager = StateManager.get_instance()
        self.logger = LoggingService.get_instance()
        self.process_service = ProcessService()

        # Initialize UI components
        self.benchmark_button = None
        self.imatrix_radio = None
        self.input_dir = None
        self.input_dir_button = None
        self.input_dir_content = None
        self.output_console = None
        self.output_dir = None
        self.output_dir_button = None
        self.output_dir_content = None
        self.output_format_radio = None
        self.progress = None
        self.quant_types = None
        self.quantize_and_bench_button = None
        self.quantize_button = None
        self.results_table = None
        self.status_textbox = None
        self.timer = None
        self.log_reader_timer = None
        self.results_timer = None

        # Register as state listener
        self.state_manager.register_listener(self.on_state_change)

    def on_state_change(self, state):
        """Handle state changes."""
        # This method will be called when the state manager notifies listeners
        # Implementation will be handled by UI event handlers
        pass

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
                                self.process_service.get_imatrix_options(),
                                label="Importance Matrix (imatrix.dat)",
                                value=self.process_service.get_imatrix_options()[0],
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
                                label=f"Model Output Directory ({self.process_service.get_output_formats()[0]})",
                                placeholder="Path to output directory...",
                                interactive=False,
                                value=r"",
                            )
                            self.output_format_radio = gr.Radio(
                                self.process_service.get_output_formats(),
                                label="Output Format",
                                value=self.process_service.get_output_formats()[0],
                                interactive=True,
                            )
                            self.output_dir_content = gr.DataFrame(
                                headers=["Name", "Size"],
                                datatype=["str", "str"],
                                max_height=200,
                                interactive=False,
                            )

                    with gr.Row():
                        self.quant_types = gr.CheckboxGroup(
                            label="Quantization Types",
                            choices=get_available_quant_types()
                        )
                    with gr.Row():
                        with gr.Column():
                            self.quantize_button = gr.Button("Convert & Quantize Model", variant="primary")
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
                        headers=self.process_service.get_results_table_headers(),
                        datatype=["str"] * len(self.process_service.get_results_table_headers()),
                        label="Benchmark Results",
                        interactive=False,
                        wrap=True,
                        column_widths=self.process_service.get_results_table_column_widths(),
                    )

            # --- Event Handlers ---
            self.input_dir_button.click(
                fn=self.update_input_directory,
                outputs=[self.input_dir, self.input_dir_content, self.quantize_button],
            )
            self.output_dir_button.click(
                fn=self.update_output_directory,
                outputs=[self.output_dir, self.output_dir_content],
            )
            self.output_format_radio.select(
                fn=self.update_output_dir_label,
                inputs=[self.output_format_radio],
                outputs=[self.output_dir],
            )

            # Make sure this event fires whenever checkbox group changes
            self.quant_types.change(
                fn=self.update_quantize_button_label,
                inputs=[self.quant_types],
                outputs=[self.quantize_button]
            )

            self.quantize_button.click(
                fn=self.run_quantization,
                inputs=[self.input_dir, self.output_dir, self.quant_types, self.imatrix_radio, self.output_format_radio],
                outputs=[
                    self.quantize_button,
                    self.quantize_and_bench_button,
                    self.benchmark_button,
                    self.status_textbox,
                ],
            )
            self.quantize_and_bench_button.click(
                fn=self.run_quantization_and_benchmark,
                inputs=[self.input_dir, self.output_dir, self.quant_types, self.imatrix_radio, self.output_format_radio],
                outputs=[
                    self.quantize_button,
                    self.quantize_and_bench_button,
                    self.benchmark_button,
                    self.status_textbox,
                ],
            )
            self.benchmark_button.click(
                fn=self.run_benchmark,
                inputs=[],
                outputs=[
                    self.quantize_button,
                    self.quantize_and_bench_button,
                    self.benchmark_button,
                    self.status_textbox,
                ],
            )

            # Directory content update timer (every 5 seconds)
            self.timer = gr.Timer(value=5, active=True)
            self.timer.tick(
                fn=self.update_output_dir_content,
                inputs=self.output_dir,
                outputs=self.output_dir_content,
            )

            # Log reader timer (every second)
            self.log_reader_timer = gr.Timer(value=1, active=True)
            self.log_reader_timer.tick(
                fn=self.update_console_log,
                outputs=self.output_console,
            )

            # Results table update timer (every 2 seconds)
            self.results_timer = gr.Timer(value=2, active=True)
            self.results_timer.tick(
                fn=self.update_results_table,
                outputs=self.results_table,
            )

        return demo

    def update_input_directory(self):
        """Update input directory from file selector."""
        selected_dir, dir_content = select_and_list_directory_contents("")
        dir_name = os.path.basename(selected_dir) if selected_dir else ""

        # Update state
        self.state_manager.update_state(input_dir=selected_dir)

        # Update button label based on selected quantization types
        button_label = f"Convert {dir_name}" if not self.state_manager.state.selected_quant_types else f"Quantize {dir_name}"

        return selected_dir, gr.update(value=dir_content), gr.update(value=button_label)

    def update_output_directory(self):
        """Update output directory from file selector."""
        selected_dir, dir_content = select_and_list_directory_contents("")

        # Update state
        self.state_manager.update_state(output_dir=selected_dir)

        return selected_dir, gr.update(value=dir_content)

    def update_output_dir_content(self, output_dir_val):
        """Periodically update output directory content."""
        if output_dir_val and os.path.isdir(output_dir_val):
            try:
                contents = list_directory_contents(output_dir_val)
                return gr.update(value=contents)
            except Exception as e:
                self.logger.log_error(f"Error updating output directory content: {e}")
                return gr.update()
        return gr.update()

    def update_console_log(self):
        """Periodically update console log content."""
        try:
            log_content = self.logger.read_debug_log(DEBUG_LOG, max_lines=100)
            return gr.update(value=log_content)
        except Exception as e:
            return gr.update(value=f"Error reading log: {str(e)}")

    def update_results_table(self):
        """Periodically update results table."""
        return gr.update(value=self.state_manager.state.results_data)

    def update_output_dir_label(self, format_val):
        """Update output directory label based on selected format."""
        self.state_manager.update_state(output_format=format_val)
        return gr.update(label=f"Model Output Directory ({format_val})")

    def update_quantize_button_label(self, quant_types_val):
        """Update quantize button label based on selected quantization types."""
        dir_name = os.path.basename(self.state_manager.state.input_dir) if self.state_manager.state.input_dir else ""

        # Update state
        self.state_manager.update_state(selected_quant_types=quant_types_val)

        # If no quant types selected change button label to Convert
        is_convert_only = not quant_types_val
        button_label = f"Convert {dir_name}" if is_convert_only else f"Quantize {dir_name}"

        self.logger.log_debug(f"Updating button label: {button_label} (quant_types: {quant_types_val}, is_convert_only: {is_convert_only})")

        return gr.update(value=button_label)

    def run_quantization(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val):
        """Run quantization process."""
        # Update state
        self.state_manager.update_state(
            input_dir=input_dir_val,
            output_dir=output_dir_val,
            selected_quant_types=quant_types_val,
            imatrix_option=imatrix_val,
            output_format=output_format_val,
            process_status=ProcessStatus.IDLE,
            error_message="",
            results_data=[]
        )

        # Disable buttons immediately
        yield (
            gr.update(interactive=False),  # quantize_button
            gr.update(interactive=False),  # quantize_and_bench_button
            gr.update(interactive=False),  # benchmark_button
            gr.update(value="Starting process..."),  # status_textbox
        )

        # Validate inputs before starting process
        try:
            # Pre-validate inputs to catch errors before starting the thread
            self.process_service.validate_inputs()
        except Exception as e:
            # Show error message and re-enable buttons
            self.logger.log_error(f"Validation error: {str(e)}")
            gr.Warning(str(e))

            yield (
                gr.update(interactive=True),  # quantize_button
                gr.update(interactive=True),  # quantize_and_bench_button
                gr.update(interactive=True),  # benchmark_button
                gr.update(value=f"Error: {str(e)}"),  # status_textbox
            )
            return

        # Run process in a separate thread
        thread = threading.Thread(
            target=self.process_service.process_quantization_and_benchmark,
            args=(self.progress,)
        )
        thread.daemon = True
        thread.start()

        # Wait a bit to make sure the process starts
        time.sleep(0.5)

        # Start updating status
        while thread.is_alive():
            status = self.state_manager.state.process_status
            status_message = f"Process Status: {status.value.capitalize()}"

            if status == ProcessStatus.ERROR:
                error_message = self.state_manager.state.error_message
                status_message = f"Error: {error_message}"

                yield (
                    gr.update(interactive=True),  # quantize_button
                    gr.update(interactive=True),  # quantize_and_bench_button
                    gr.update(interactive=True),  # benchmark_button
                    gr.update(value=status_message),  # status_textbox
                )
                return

            yield (
                gr.update(interactive=False),  # quantize_button
                gr.update(interactive=False),  # quantize_and_bench_button
                gr.update(interactive=False),  # benchmark_button
                gr.update(value=status_message),  # status_textbox
            )

            time.sleep(0.5)

        # Final status update
        final_status = "Finished" if self.state_manager.state.process_status == ProcessStatus.FINISHED else "Error"

        yield (
            gr.update(interactive=True),  # quantize_button
            gr.update(interactive=True),  # quantize_and_bench_button
            gr.update(interactive=True),  # benchmark_button
            gr.update(value=f"Process {final_status}"),  # status_textbox
        )

    def run_quantization_and_benchmark(self, input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val):
        """Run quantization and benchmark process."""
        # For now, same as run_quantization as benchmarking is integrated
        return self.run_quantization(input_dir_val, output_dir_val, quant_types_val, imatrix_val, output_format_val)

    def run_benchmark(self):
        """Run benchmark process."""
        # Placeholder for separate benchmark implementation
        self.logger.log_user("Separate benchmarking functionality not implemented yet.")

        return (
            gr.update(interactive=True),  # quantize_button
            gr.update(interactive=True),  # quantize_and_bench_button
            gr.update(interactive=True),  # benchmark_button
            gr.update(value="Separate benchmarking not implemented yet."),  # status_textbox
        )


if __name__ == "__main__":
    try:
        ui = QuantBenchUI()
        demo = ui.create_ui()
        demo.queue()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except Exception as e:
        import traceback

        print(f"Error launching UI: {e}")
        print(traceback.format_exc())
