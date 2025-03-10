#!/usr/bin/env python3
"""
KGML Test Results Viewer - Main Application

An interactive web interface for exploring KGML test runs, viewing request-response pairs,
and analyzing test statistics.
"""

import argparse
import logging
from pathlib import Path

from knowledge.reasoning.tests.util.kgml_test_result_ui_util import DEFAULT_BASE_DIR
from knowledge.reasoning.tests.util.kgml_test_result_viewer import create_ui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KGMLViewer")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="KGML Test Results Viewer")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Directory containing test logs (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="Port to run the Gradio server on (default: 7860)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link"
    )

    args = parser.parse_args()

    # Validate logs directory
    logs_path = Path(args.logs_dir)
    if not logs_path.exists():
        logger.warning(f"Logs directory {args.logs_dir} does not exist. Creating it.")
        logs_path.mkdir(parents=True, exist_ok=True)

    # Create and launch the UI
    app = create_ui()
    logger.info(f"Starting KGML Test Results Viewer on port {args.port}")
    app.launch(
        server_port=args.port,
        debug=args.debug,
        share=args.share
    )


if __name__ == "__main__":
    main()
