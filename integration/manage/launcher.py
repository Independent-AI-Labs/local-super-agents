import os
import time
from multiprocessing import Process

from integration.data.config import (
    OLLAMA_PATH, OLLAMA_LOG, WEBUI_LOG, LOGS_DIR, DEFAULT_ENV,
    OLLAMA_WORKING_DIR, OLLAMA_PORT, OPEN_WEBUI_PORT, OPEN_WEBUI_EXT_PORT,
    SSL_CERT_FILE, SSL_KEY_FILE, CADDY_PATH, CADDYFILE_PATH, CADDY_LOG, OLLAMA_ENV
)
from integration.desktop.windows import windows_overlay
from integration.desktop.windows.util.shell_util import run_command, is_port_open, kill_process
from integration.manage.util.misc_util import rotate_logs


def safe_terminate(proc, name="process"):
    """
    Terminate a process gracefully. Works whether proc is a subprocess.Popen or a multiprocessing.Process.
    """
    if proc is None:
        return
    try:
        print(f"[INFO] Terminating {name}...")
        proc.terminate()  # Both Popen and Process support terminate()
        proc.wait(timeout=10)
        print(f"[INFO] {name} terminated.")
    except Exception as e:
        print(f"[ERROR] Failed to terminate {name} gracefully: {e}. Attempting to kill it...")
        try:
            proc.kill()
            proc.wait(timeout=5)
            print(f"[INFO] {name} killed.")
        except Exception as kill_e:
            print(f"[ERROR] Could not kill {name}: {kill_e}")


def start_ollama():
    """
    Check if the port is open, if not start ollama.exe serve.
    """
    if is_port_open(OLLAMA_PORT):
        print(f"[INFO] Ollama is already running on port {OLLAMA_PORT}.")
        return None

    # Not running, so start it
    command = f'"{OLLAMA_PATH}" serve'
    post_init_script = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "res", "envs", "ollama_env_setup.bat"))
    return run_command(command, OLLAMA_LOG, conda_env=OLLAMA_ENV, working_dir=OLLAMA_WORKING_DIR,
                       activate_oneapi=True, post_init_script=post_init_script, elevated_external=True)


def start_open_webui():
    """
    Check if the port is open, if not start open-webui serve.
    (we assume "open-webui" is in PATH from conda/pip install)
    """
    if is_port_open(OPEN_WEBUI_PORT):
        print(f"[INFO] open-webui is already running on port {OPEN_WEBUI_PORT}.")
        return None

    command = "open-webui serve --host 127.0.0.1"
    post_init_script = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "res", "envs", "open_webui_env_setup.bat"))
    return run_command(command, WEBUI_LOG, conda_env=DEFAULT_ENV, post_init_script=post_init_script)


def check_update():
    """
    Placeholder for checking an update endpoint.
    Implement your logic here.
    """
    print("[INFO] Checking for updates...")
    # TODO: implement update checking logic here.


def start_windows_overlay():
    """
    Start the main_loop() function of the windows_overlay module in a separate Python process.
    """
    overlay_process = Process(target=windows_overlay.main_loop)
    overlay_process.start()
    return overlay_process


def stop_services():
    """
    Stop ollama and open-webui services if they are running.
    """
    if is_port_open(OLLAMA_PORT):
        print("[INFO] Ollama is running. Stopping it...")
        kill_process(port=OLLAMA_PORT)

    kill_process(process_name="ollama_llama_server.exe")
    kill_process(process_name="ollama.exe")

    if is_port_open(OPEN_WEBUI_PORT):
        print("[INFO] Open-WebUI is running. Stopping it...")
        kill_process(port=OPEN_WEBUI_PORT)


def start_proxy():
    """
    Starts the Caddy proxy server with TLS termination and reverse proxying.
    """
    if is_port_open(OPEN_WEBUI_EXT_PORT):
        print(f"[INFO] Caddy is already running on port {OPEN_WEBUI_EXT_PORT}.")
        return None

    # Generate the Caddyfile dynamically
    caddyfile_content = f"""
:{OPEN_WEBUI_EXT_PORT} {{
    bind 0.0.0.0
    tls {SSL_CERT_FILE} {SSL_KEY_FILE} {{
    }}
    reverse_proxy http://127.0.0.1:{OPEN_WEBUI_PORT}
}}
"""
    # Write the Caddyfile
    with open(CADDYFILE_PATH, 'w') as f:
        f.write(caddyfile_content)

    # Start the Caddy process
    command = f"{CADDY_PATH} run --config {CADDYFILE_PATH}"
    proxy_process = run_command(command, CADDY_LOG)
    return proxy_process


def main():
    # Stop any existing services first.
    stop_services()

    # Rotate logs before starting new services.
    rotate_logs(LOGS_DIR)

    # Check for updates (if implemented).
    check_update()

    # Start the Windows overlay in its own process.
    overlay_process = start_windows_overlay()

    # Start ollama and open-webui. (They are later killed by stop_services().)
    start_ollama()
    start_open_webui()

    # Start the proxy only if the SSL certificate exists.
    proxy_process = None
    if os.path.exists(SSL_CERT_FILE):
        # Wait until open-webui is running.
        while not is_port_open(OPEN_WEBUI_PORT):
            time.sleep(1)
        proxy_process = start_proxy()

    try:
        # Main loop: simply wait until a keyboard interrupt is received.
        while True:
            # Optionally, you can poll proxy_process to see if it has exited unexpectedly.
            if proxy_process:
                proxy_process.poll()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Terminating processes...")
        safe_terminate(overlay_process, "Windows Overlay")
        safe_terminate(proxy_process, "Caddy Proxy")
        print("[INFO] Exiting launcher...")
    finally:
        # In addition, call stop_services() to kill any lingering processes.
        stop_services()


if __name__ == "__main__":
    main()
