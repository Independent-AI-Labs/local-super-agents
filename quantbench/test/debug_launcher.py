import logging
import platform
import sys
import threading
import time
import traceback
from contextlib import contextmanager

# Configure basic logging for startup diagnostics
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

startup_logger = logging.getLogger("startup")


class TimeoutException(Exception):
    pass


if platform.system() == "Windows":
    @contextmanager
    def time_limit(seconds):
        timer = threading.Timer(seconds, lambda: _raise_timeout(seconds))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()


    def _raise_timeout(seconds):
        raise TimeoutException(f"Timed out after {seconds} seconds")

else:
    import signal


    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutException(f"Timed out after {seconds} seconds")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


def launch_with_timeout(demo):
    """Try to launch with a timeout to identify hanging"""
    startup_logger.info("Setting up launch in separate thread")

    # Define a function to run in a thread
    def launch_server():
        try:
            startup_logger.info("Thread starting - launching server")
            # Try with inbuilt server first, which should be more reliable for diagnostics
            demo.launch(
                server_name="127.0.0.1",
                server_port=7860,
                favicon_path=None,
                auth=None,
                inbrowser=False,
                share=False,
                debug=True,
                max_threads=1,  # Minimize threading
                show_api=False,  # Hide API
                show_error=True,  # Show detailed errors
                quiet=False,  # Show all messages
            )
            startup_logger.info("Server launch completed in thread")
        except Exception as e:
            startup_logger.error(f"Error in launch thread: {e}")
            startup_logger.error(traceback.format_exc())

    # Create and start the thread
    thread = threading.Thread(target=launch_server)
    thread.daemon = True  # Daemon threads will exit when the main program exits

    startup_logger.info("Starting launch thread")
    thread.start()

    # Wait for the thread for a while
    startup_logger.info("Waiting for server to start...")
    for i in range(30):  # 30 seconds timeout
        if not thread.is_alive():
            startup_logger.info("Launch thread completed")
            break
        startup_logger.info(f"Server starting... ({i + 1}s)")
        time.sleep(1)

    if thread.is_alive():
        startup_logger.error("Server launch appears to be hanging after 30 seconds")
        startup_logger.info("Will continue execution - server may still start in background")

    return thread


def main():
    try:
        startup_logger.info("Starting QuantBench diagnostic launch...")

        # Import with timeout
        startup_logger.info("Importing UI class...")
        try:
            with time_limit(10):
                from quantbench.main_ui import QuantBenchUI
                startup_logger.info("Import successful")
        except TimeoutException:
            startup_logger.error("Import timed out after 10 seconds")
            sys.exit(1)
        except ImportError as e:
            startup_logger.error(f"Import error: {e}")
            startup_logger.error(traceback.format_exc())
            sys.exit(1)

        # Create UI with timeout
        startup_logger.info("Creating UI instance...")
        try:
            with time_limit(10):
                ui = QuantBenchUI()
                startup_logger.info("UI instance created")
        except TimeoutException:
            startup_logger.error("UI creation timed out after 10 seconds")
            sys.exit(1)

        # Build UI with timeout
        startup_logger.info("Building UI...")
        try:
            with time_limit(20):
                demo = ui.create_ui()
                startup_logger.info("UI built successfully")
        except TimeoutException:
            startup_logger.error("UI building timed out after 20 seconds")
            sys.exit(1)

        # Setup queue
        startup_logger.info("Setting up queue...")
        demo.queue()

        # Launch with a timeout in a separate thread
        launch_thread = launch_with_timeout(demo)

        # Keep the main thread running
        startup_logger.info("Main thread continuing. Press Ctrl+C to exit.")
        startup_logger.info("If the server is running, it should be accessible at http://127.0.0.1:7860")

        # Just to keep the main thread alive
        while launch_thread.is_alive():
            time.sleep(1)

    except Exception as e:
        startup_logger.error(f"Unexpected error during startup: {e}")
        startup_logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
