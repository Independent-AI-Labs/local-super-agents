import locale
import os
import socket
import subprocess
import time

from compliance.services.logging_service import DEFAULT_LOGGER
from integration.data.config import CONDA_PATH


def run_command(
        command,
        log_file_path,
        conda_env=None,
        working_dir=None,
        post_init_script=None,
        elevated_external=False
) -> subprocess.Popen:
    """
    Runs a command as a subprocess, optionally with Intel oneAPI and/or Conda environment activation,
    redirects stdout/stderr to `log_file_path`, and optionally runs with elevated permissions.

    Args:
        command (str): The shell command to execute.
        log_file_path (str): Path to the log file for output redirection.
        conda_env (str, optional): Name of the Conda environment to activate.
        working_dir (str, optional): Directory in which to run the command.
        post_init_script (str, optional): Path to a custom initialization script to run before the command.
        elevated_external (bool, optional): If True, run the command with elevated (Administrator) permissions.

    Returns:
        subprocess.Popen: The process object for the command.
    """
    # Ensure the log file directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Open the log file for appending
    log_file = open(log_file_path, 'a', encoding='utf-8')

    try:
        # Build the base_command to chain all setups
        base_command = ""

        # Step 1: Activate oneAPI environment if requested
        # if activate_oneapi:
        #     base_command += f'call "{ONE_API_PATH}" intel64 vs2022 ^&& '

        # Step 2: Activate Conda environment if requested
        if conda_env:
            base_command += f'call "{CONDA_PATH}" activate ^&& '
            base_command += f'conda activate {conda_env} ^&& '

        # Step 3: Run post-init script if provided
        if post_init_script:
            base_command += f'call "{post_init_script}" ^&& '

        # Step 5: Append the actual user command
        base_command += command

        # Log the full command being executed
        DEFAULT_LOGGER.log_debug(f"Running command: {base_command}, cwd={working_dir}")

        # Step 6: Elevate privileges if required
        if elevated_external:
            # Use PowerShell to request elevated permissions with -Verb RunAs
            elevate_command = (
                f'powershell -Command "Start-Process -FilePath \'cmd.exe\' -ArgumentList \'/c {base_command} > \"{log_file_path}\" 2>&1\' -Verb RunAs -WindowStyle Hidden"'
            )
            base_command = elevate_command

        # Launch the process
        process = subprocess.Popen(
            base_command,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=working_dir,
            env=os.environ,
            encoding=locale.getpreferredencoding(False),
            errors="replace"
        )
        return process

    except FileNotFoundError as e:
        DEFAULT_LOGGER.log_debug(f"Executable not found: {e}")
        raise
    except Exception as e:
        print(f"Error running command: {e}")
        raise


def is_port_open(port):
    """
    Returns True if the TCP port is open on localhost, otherwise False.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0


def wait_for_process_output(process, polling_interval=1.0):
    """
    Optionally, a helper to wait for the process to finish.
    Or you can run a background thread that tail-logs if you want real-time updates.
    """
    while True:
        if process.poll() is not None:
            break
        time.sleep(polling_interval)
    # process has ended


def podman_command(
        args,
        log_file_path,
        working_dir=None,
        podman_path=r".\podman.exe"
):
    """
    Runs a podman command with the given arguments.
    Example: podman_command(["machine", "start"], "podman.log")
    """
    command = f'"{podman_path}" {" ".join(args)}'
    return run_command(command, log_file_path, working_dir=working_dir)


def kill_process(process_name=None, pid=None, port=None):
    """
    Kill a process by name, PID, or port. Requires elevated permissions.

    Args:
        process_name (str): Name of the process to terminate.
        pid (int): Process ID to terminate.
        port (int): Port number to identify and terminate the process.

    Raises:
        Exception: If the operation fails or insufficient permissions.
    """
    try:
        if process_name:
            # Kill by process name
            result = subprocess.run(
                f"taskkill /IM {process_name} /F",
                shell=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print(f"Process '{process_name}' terminated successfully.")
        elif pid:
            # Kill by PID
            result = subprocess.run(
                f"taskkill /PID {pid} /F",
                shell=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print(f"Process with PID {pid} terminated successfully.")
        elif port:
            # Kill by port (get PID first)
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().split("\n")
            if not lines:
                print(f"No process found running on port {port}.")
                return

            for line in lines:
                pid = int(line.split()[-1])
                result = subprocess.run(
                    f"taskkill /PID {pid} /F",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)
                print(f"Process on port {port} (PID {pid}) terminated successfully.")
        else:
            raise ValueError("You must specify a process name, PID, or port.")

    except subprocess.CalledProcessError as e:
        # Handle harmless errors (e.g., process already terminated)
        if e.returncode in [128, 5]:  # Exit codes 128 (process unavailable) or 5 (access denied)
            print(f"[WARNING] Taskkill returned exit code {e.returncode}: {e.stderr.strip()}")
        else:
            print(f"[ERROR] Failed to terminate the process: {e.stderr.strip()}")
            raise
