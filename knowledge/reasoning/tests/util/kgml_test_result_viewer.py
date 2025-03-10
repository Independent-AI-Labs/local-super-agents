#!/usr/bin/env python3
"""
KGML Test Results Gradio UI

A Gradio-based user interface for the KGML Test Results Viewer.
This module serves as the view component in the MVC pattern.
"""

import logging
from typing import Dict, List, Tuple

import gradio as gr

# Import the controller
from knowledge.reasoning.tests.util.kgml_test_result_viewer_controller import KGMLTestResultsController

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLViewer")

# Create a controller instance
controller = KGMLTestResultsController()


# ========== UI Event Handlers ==========

def refresh_runs() -> Tuple[List[List], List[Dict], str]:
    """
    Refresh the list of test runs

    Returns:
        Tuple containing visible data for the table, full run data, and any error messages
    """
    return controller.get_test_runs()


def on_run_selected(evt: gr.SelectData, runs_data: List[Dict]) -> Tuple:
    """
    Handle run selection event

    Args:
        evt: Selection event data
        runs_data: Full run data list

    Returns:
        Tuple containing updated UI component values
    """
    return controller.get_run_details(evt.index[0], runs_data)


def on_test_selected(evt: gr.SelectData, tests_data: List[Dict]) -> Tuple:
    """
    Handle test selection event

    Args:
        evt: Selection event data
        tests_data: Full test data list

    Returns:
        Tuple containing updated UI component values
    """
    return controller.get_test_details(evt.index[0], tests_data)


def on_iteration_selected(evt: gr.SelectData, iterations_data: List[Dict]) -> Tuple:
    """
    Handle iteration selection event

    Args:
        evt: Selection event data
        iterations_data: Full iteration data list

    Returns:
        Tuple containing updated UI component values
    """
    return controller.get_iteration_details(evt.index[0], iterations_data)


# ========== UI Creation ==========

def create_ui():
    """Create the Gradio UI"""

    with gr.Blocks(title="KGML Test Results Viewer", theme=gr.themes.Soft(), css="""
        .tab-nav button.selected {
            font-weight: bold;
            border-bottom-width: 3px;
        }
        .status-passed {
            color: green;
            font-weight: bold;
        }
        .status-failed {
            color: red;
            font-weight: bold;
        }
        .status-unknown {
            color: orange;
            font-weight: bold;
        }
        .summary-header {
            margin-top: 0;
            padding-top: 0;
        }
        .selected-row {
            background-color: rgba(63, 81, 181, 0.2) !important;
        }
        .dataframe-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        /* Style for consistent row heights */
        .dataframe tbody tr {
            height: 48px !important;
            line-height: 1.2;
            max-height: 48px;
            overflow: hidden;
        }

        .dataframe tbody tr.selected {
            background-color: rgba(63, 81, 181, 0.2) !important; /* Example: light blue */
        }

        /* Make sure the content doesn't overflow */
        .dataframe td, .dataframe th {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Improve readability of Markdown content */
        .prose {
            max-width: 100%;
            line-height: 1.5;
        }
    """) as app:
        gr.Markdown("# 📊 KGML Test Results Viewer")
        gr.Markdown("Interactive viewer for KGML test data and processing results")

        # Create hidden state variables to store full data with paths
        runs_data_state = gr.State([])
        tests_data_state = gr.State([])
        iterations_data_state = gr.State([])

        # Error output for the entire UI
        error_output = gr.Markdown(visible=True)

        # Top section: Tables for runs, tests, and iterations side by side
        with gr.Row():
            # Runs table (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("## Available Test Runs")
                runs_table = gr.Dataframe(
                    headers=["Rate", "Model", "Date"],
                    row_count=8,
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    column_widths=["20%", "55%", "25%"]
                )
                refresh_button = gr.Button("🔄 Refresh Test Runs", variant="primary", size="sm")

            # Tests table (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("## Tests in Run")
                tests_table = gr.Dataframe(
                    headers=["Status", "Valid", "Test Name"],
                    row_count=8,
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    column_widths=["25%", "20%", "55%"]
                )

            # Iterations table (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("## Test Iterations")
                iterations_table = gr.Dataframe(
                    headers=["Status", "Time"],
                    row_count=8,
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    column_widths=["80%", "20%"]
                )

        # Bottom section: Details tabs
        with gr.Tabs() as tabs:
            # Tab for run analysis
            with gr.Tab("📈 Run Analysis", id="run_analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        run_summary = gr.Markdown(label="Run Summary")

                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.Tab("Test Results"):
                                test_result_chart = gr.Plot(label="Test Results Distribution")

                            with gr.Tab("Response Times"):
                                response_time_chart = gr.Plot(label="Response Times")

            # Tab for test details
            with gr.Tab("🧪 Test Details", id="test_details"):
                test_summary = gr.Markdown(label="Test Summary")

            # Tab for iteration details
            with gr.Tab("🔍 Iteration Details", id="iteration_details"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Request KGML")
                        request_text = gr.Code(
                            label="Request sent to the model",
                            language="sql",  # Use SQL for KGML as it has similar structure
                            interactive=False,
                            lines=21,
                            max_lines=21
                        )

                    with gr.Column():
                        gr.Markdown("### Response KGML")
                        response_text = gr.Code(
                            label="Response received from the model",
                            language="sql",  # Use SQL for KGML as it has similar structure
                            interactive=False,
                            lines=21,
                            max_lines=21
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Execution Result")
                        processing_result_text = gr.Code(
                            label="Result of parsing / executing the KGML",
                            language="json",
                            interactive=False,
                            lines=21,
                            max_lines=21
                        )

                    with gr.Column():
                        gr.Markdown("### Processing Graph", max_height=300)
                        processing_log_chart = gr.Plot(label="Visualization of the processing steps")

        # Set up event handlers
        refresh_button.click(
            fn=refresh_runs,
            outputs=[runs_table, runs_data_state, error_output]
        )

        # Use SelectData events for table selections with state
        runs_table.select(
            fn=on_run_selected,
            inputs=[runs_data_state],
            outputs=[
                tests_table,
                run_summary,
                test_result_chart,
                response_time_chart,
                error_output,
                iterations_table,
                tests_data_state
            ]
        ).then(
            # Use a separate function to select the tab
            lambda: gr.Tabs(selected="run_analysis"),
            inputs=None,
            outputs=tabs
        )

        tests_table.select(
            fn=on_test_selected,
            inputs=[tests_data_state],
            outputs=[
                iterations_table,
                test_summary,
                error_output,
                iterations_data_state
            ]
        ).then(
            # Use a separate function to select the tab
            lambda: gr.Tabs(selected="test_details"),
            inputs=None,
            outputs=tabs
        )

        iterations_table.select(
            fn=on_iteration_selected,
            inputs=[iterations_data_state],
            outputs=[
                request_text,
                response_text,
                processing_result_text,
                processing_log_chart,
                error_output
            ]
        ).then(
            # Use a separate function to select the tab
            lambda: gr.Tabs(selected="iteration_details"),
            inputs=None,
            outputs=tabs
        )

        # Load initial data when the UI starts
        app.load(
            fn=refresh_runs,
            outputs=[runs_table, runs_data_state, error_output]
        )

    return app


# ========== Main Entry Point ==========

def main():
    """Main entry point for the application"""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="KGML Test Results Viewer")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public link for sharing")
    parser.add_argument("--base-dir", type=str, default="kgml_test_logs",
                        help="Base directory for test logs")

    args = parser.parse_args()

    # Update controller with base directory
    global controller
    controller = KGMLTestResultsController(args.base_dir)

    # Create and launch the UI
    app = create_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
