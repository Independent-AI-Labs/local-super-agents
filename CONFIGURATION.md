# Configuration Guide for **Local Super-Agents**

This document provides a guide to configuring the Agents application. The application's behavior is largely controlled through environment variables, allowing for flexible customization without modifying the code directly.

## Environment Variables

The following environment variables can be set to customize the application.  When an environment variable is not set, the application falls back to default values, which are often derived or hardcoded within the application.
The following environment variables can be set to customize the application.  When an environment variable is not set, the application falls back to default values, which are often derived or hardcoded within the application.

**To set environment variables:**

The method for setting environment variables depends on your operating system.

*   **Windows:**
    *   **System-wide (for all users):**  Use the System Properties dialog (`Control Panel` -> `System and Security` -> `System` -> `Advanced system settings` -> `Environment Variables`).
    *   **User-specific:** Use the same dialog, but in the "User variables" section.
    *   **For a single command prompt session:** Use the `set` command (e.g., `set INSTALLATION_DIR=C:\path\to\install`). These are temporary and only last for the current command prompt session.

*   **macOS and Linux:**
    *   **For the current shell session:** Use the `export` command in your terminal (e.g., `export INSTALLATION_DIR=/opt/agents`).
    *   **Persistently (for future sessions):**  Modify your shell's configuration file (e.g., `.bashrc`, `.zshrc`, `.bash_profile`). Add `export` commands to this file.
    *   **System-wide (less common for user applications):**  Modify system environment files like `/etc/environment` or files in `/etc/profile.d/`. (Generally not recommended for this application unless you understand the implications).

**Note:** After setting environment variables, you may need to restart your terminal or the application for the changes to take effect.

---

## Configuration Variables Reference

Below is a detailed description of each environment variable used in the Agents application, categorized for easier understanding.

### 1. Installation Path Configuration

These variables control where the application is installed and how to locate its core files.

*   **`INSTALLATION_DIR` (Environment Variable):**
    *   **Description:** Specifies the root directory where the Agents application is installed. Many other paths within the application are derived relative to this directory.
    *   **How to Configure:** Set the `INSTALLATION_DIR` environment variable to the desired installation path.
    *   **Default Value:**
        *   If `INSTALLATION_DIR` is not set, the application first checks for an `install.path` file in the user's home directory under `.agents` (i.e., `~/.agents/install.path`). If this file exists and contains a path, that path is used.
        *   If neither `INSTALLATION_DIR` environment variable nor the `install.path` file is found, the `INSTALL_PATH` will be `None`, which might lead to issues as many paths are dependent on it. **It is highly recommended to set either `INSTALLATION_DIR` or ensure the `install.path` file exists.**

*   **`INSTALL_PATH_FILE` (Python Constant):**
    *   **Description:**  Defines the path to the file where the default installation path is stored. This is a Python constant and is generally **not meant to be changed by the user**. It is used internally by the application to locate the `install.path` file.
    *   **Default Value:**  `~/.agents/install.path` (where `~` represents the user's home directory).

*   **`get_default_install_path()` (Python Function):**
    *   **Description:** This function reads the installation path from the `INSTALL_PATH_FILE` if it exists. This is a Python function and is **not meant to be configured by the user**.

*   **`INSTALL_PATH` (Python Variable):**
    *   **Description:**  This Python variable holds the resolved installation path. It is determined based on the `INSTALLATION_DIR` environment variable or the `install.path` file (via `get_default_install_path()`).
    *   **Configuration:**  Indirectly configured by setting the `INSTALLATION_DIR` environment variable or by creating/modifying the `install.path` file at `~/.agents/install.path`.


### 2. Google Drive API Credentials

These variables are used if the application integrates with Google Drive.

*   **`GDRIVE_CLIENT_ID`:**
    *   **Description:**  Your Google Drive API Client ID. Required for authenticating with Google Drive API.
    *   **How to Configure:** Set the `GDRIVE_CLIENT_ID` environment variable to your Google Drive Client ID.
    *   **Default Value:**  Empty string (`''`). Google Drive integration might not function correctly if this is not set when needed.

*   **`GDRIVE_CLIENT_SECRET`:**
    *   **Description:** Your Google Drive API Client Secret. Required for authenticating with Google Drive API.
    *   **How to Configure:** Set the `GDRIVE_CLIENT_SECRET` environment variable to your Google Drive Client Secret.
    *   **Default Value:** Empty string (`''`). Google Drive integration might not function correctly if this is not set when needed.


### 3. Virtual Machine (VM) Address

*   **`VM_ADDRESS`:**
    *   **Description:** The base URL or IP address of the virtual machine where supporting services (like SearXNG) are running.
    *   **How to Configure:** Set the `VM_ADDRESS` environment variable to the correct address of your VM (including `http://` or `https://` if applicable).
    *   **Default Value:** `http://172.72.72.2`. You will likely need to change this if your VM is on a different address.


### 4. Ollama and Open WebUI Configuration

These variables configure the interaction with Ollama (for large language models) and Open WebUI (a web interface for LLMs).

*   **`OLLAMA_PORT`:**
    *   **Description:** The port on which the Ollama service is running.
    *   **How to Configure:** Set the `OLLAMA_PORT` environment variable to the desired port number.
    *   **Default Value:** `11434`.

*   **`OLLAMA_BASE_URL`:**
    *   **Description:** The base URL for accessing the Ollama API.
    *   **How to Configure:** Set the `OLLAMA_BASE_URL` environment variable.  Generally, you should only need to change this if you are running Ollama on a different host or port than the default.
    *   **Default Value:** `http://127.0.0.1:11434` (derived from `OLLAMA_PORT`).

*   **`OPEN_WEBUI_PORT`:**
    *   **Description:** The port on which the Open WebUI service is running.
    *   **How to Configure:** Set the `OPEN_WEBUI_PORT` environment variable to the desired port number.
    *   **Default Value:** `8080`.

*   **`OPEN_WEBUI_BASE_URL`:**
    *   **Description:** The base URL for accessing the Open WebUI web interface.
    *   **How to Configure:** Set the `OPEN_WEBUI_BASE_URL` environment variable. Generally, you should only need to change this if you are running Open WebUI on a different host or port than the default.
    *   **Default Value:** `http://127.0.0.1:8080` (derived from `OPEN_WEBUI_PORT`).

*   **`OPEN_WEBUI_EMAIL`:**
    *   **Description:**  The email address for logging into Open WebUI, if authentication is enabled.
    *   **How to Configure:** Set the `OPEN_WEBUI_EMAIL` environment variable to your Open WebUI email.
    *   **Default Value:** Empty string (`''`).

*   **`OPEN_WEBUI_PASSWORD`:**
    *   **Description:** The password for logging into Open WebUI, if authentication is enabled.
    *   **How to Configure:** Set the `OPEN_WEBUI_PASSWORD` environment variable to your Open WebUI password.
    *   **Default Value:** Empty string (`''`).

*   **`OPEN_WEBUI_EXT_PORT`:**
    *   **Description:** The external port used to serve HTTPS traffic for Open WebUI (likely for SSL termination via Caddy).
    *   **How to Configure:**  **Generally, you should not need to change this unless you have specific networking requirements.**  Set the `OPEN_WEBUI_EXT_PORT` environment variable to the desired external HTTPS port.
    *   **Default Value:** `8443`.

*   **`WEBUI_SSL_CERT_FILE`:**
    *   **Description:** Path to the SSL certificate file used for HTTPS for Open WebUI.
    *   **How to Configure:** Set the `WEBUI_SSL_CERT_FILE` environment variable to the full path to your SSL certificate file.  By default, it is expected to be located within the installation directory.
    *   **Default Value:**  `{INSTALL_PATH}\certs\certificate.crt` (relative to `INSTALL_PATH`).

*   **`WEBUI_SSL_KEY_FILE`:**
    *   **Description:** Path to the SSL private key file used for HTTPS for Open WebUI.
    *   **How to Configure:** Set the `WEBUI_SSL_KEY_FILE` environment variable to the full path to your SSL key file. By default, it is expected to be located within the installation directory.
    *   **Default Value:** `{INSTALL_PATH}\certs\private.key` (relative to `INSTALL_PATH`).


### 5. SearXNG Configuration

*   **`SEARXNG_BASE_URL`:**
    *   **Description:** The base URL for accessing the SearXNG search engine instance.
    *   **How to Configure:** Set the `SEARXNG_BASE_URL` environment variable to the base URL of your SearXNG instance (including `http://` or `https://`).
    *   **Default Value:** `{AGENTS_VM_ADDRESS}:8888` (derived from `VM_ADDRESS`). You will likely need to adjust this if SearXNG is not running at this default address relative to your VM.


### 6. Chat ID

*   **`CHAT_ID`:**
    *   **Description:**  An identifier for a specific chat session or context. The exact usage depends on the application's functionality, but it might be used to group or identify conversations.
    *   **How to Configure:** Set the `CHAT_ID` environment variable to the desired chat identifier.
    *   **Default Value:** Empty string (`''`).


### 7. Default Environments and Language Models

*   **`DEFAULT_ENV` (Used for LSA Environment):**
    *   **Description:** Specifies the default environment for some features, potentially "LSA" (likely referring to a specific application environment or configuration).
    *   **How to Configure:** Set the `DEFAULT_ENV` environment variable to the desired default environment string.
    *   **Default Value:** `'lsa'`.

*   **`OLLAMA_ENV` (Used for LLM Environment):**
    *   **Description:** Specifies the default environment related to Ollama and LLMs.
    *   **How to Configure:** Set the `OLLAMA_ENV` environment variable to the desired default environment string for LLM related features.
    *   **Default Value:** `'llm'`.

*   **`DEFAULT_LLM`:**
    *   **Description:**  Specifies the default Large Language Model (LLM) to be used by the application (e.g., for Ollama).
    *   **How to Configure:** Set the `DEFAULT_LLM` environment variable to the identifier of the desired default LLM model (e.g., `'internlm_7b'`, `'llama2'`, etc., depending on your Ollama setup).
    *   **Default Value:** `'internlm_7b'`.


### 8. Cache and Logs Directories

*   **`CACHE_DIR`:**
    *   **Description:**  Specifies the directory to be used for caching data. Caching can improve performance by storing frequently accessed information.
    *   **How to Configure:** Set the `CACHE_DIR` environment variable to the desired path for the cache directory.
    *   **Default Value:** Empty string (`''`). In this case, the application might use a default temporary cache location, or caching might be disabled depending on implementation.

*   **`LOGS_DIR`:**
    *   **Description:** Specifies the directory where application log files will be stored. Logs are useful for debugging and monitoring the application.
    *   **How to Configure:** Set the `LOGS_DIR` environment variable to the desired path for the logs directory.
    *   **Default Value:** `{INSTALL_PATH}\.agents\logs` (relative to `INSTALL_PATH`). If `INSTALL_PATH` is not properly set, the logs directory might not be correctly located.


### 9. Desktop Overlay FPS

*   **`DESKTOP_OVERLAY_TARGET_FPS`:**
    *   **Description:**  The target frames per second (FPS) for the desktop overlay feature (if implemented). Higher FPS values may result in smoother overlays but might consume more system resources.
    *   **How to Configure:** Set the `DESKTOP_OVERLAY_TARGET_FPS` environment variable to the desired FPS value (as an integer).
    *   **Default Value:** `240`.

*   **`DESKTOP_OVERLAY_TARGET_FRAME_TIME` (Python Constant):**
    *   **Description:**  Calculated frame time in seconds, derived from `DESKTOP_OVERLAY_TARGET_FPS`. This is a Python constant and **not meant to be configured directly by the user.**
    *   **Calculation:** `1.0 / float(DESKTOP_OVERLAY_TARGET_FPS)`.


### 10. Paths to Executables

These variables define the paths to various external executable files that the application depends on. **Ensure these paths are correct for your installation.**  Incorrect paths will prevent the application from finding and using these tools.

*   **`ONE_API_PATH`:**
    *   **Description:** Path to the `setvars.bat` batch file from Intel oneAPI. This is likely needed for Intel-specific optimizations or libraries.
    *   **How to Configure:** Set the `ONE_API_PATH` environment variable to the full path to the `setvars.bat` file in your Intel oneAPI installation.
    *   **Default Value:** `{INSTALL_PATH}\Program Files (x86)\Intel\oneAPI\setvars.bat` (relative to `INSTALL_PATH`).

*   **`CONDA_PATH`:**
    *   **Description:** Path to the `conda.bat` batch file from Miniconda or Anaconda. Conda is likely used for managing Python environments and dependencies.
    *   **How to Configure:** Set the `CONDA_PATH` environment variable to the full path to the `conda.bat` file in your Miniconda/Anaconda installation.
    *   **Default Value:** `{INSTALL_PATH}\tools\miniconda3\condabin\conda.bat` (relative to `INSTALL_PATH`).

*   **`CADDY_PATH`:**
    *   **Description:** Path to the `caddy.exe` executable. Caddy is a web server, likely used for serving the Open WebUI or other web components.
    *   **How to Configure:** Set the `CADDY_PATH` environment variable to the full path to the `caddy.exe` file.
    *   **Default Value:** `{INSTALL_PATH}\tools\caddy-win\caddy.exe` (relative to `INSTALL_PATH`).

*   **`CADDYFILE_PATH`:**
    *   **Description:** Path to the `Caddyfile` configuration file for Caddy.
    *   **How to Configure:** Set the `CADDYFILE_PATH` environment variable to the full path to your `Caddyfile`.
    *   **Default Value:** `{INSTALL_PATH}\tools\caddy-win\caddyfile` (relative to `INSTALL_PATH`).  **Note:** This default value seems incorrect as it's using `CADDY_PATH` again, likely a typo and should be `CADDYFILE_PATH`.

*   **`CHROME_PATH`:**
    *   **Description:** Path to the Chrome browser executable (`chrome-agents.exe`, possibly a custom build or renamed Chrome for agent purposes).
    *   **How to Configure:** Set the `CHROME_PATH` environment variable to the full path to the `chrome-agents.exe` file.
    *   **Default Value:** `{INSTALL_PATH}\tools\chrome-win\chrome-agents.exe` (relative to `INSTALL_PATH`).

*   **`OLLAMA_PATH`:**
    *   **Description:** Path to the `ollama.exe` executable.
    *   **How to Configure:** Set the `OLLAMA_PATH` environment variable to the full path to the `ollama.exe` file.
    *   **Default Value:** `{INSTALL_PATH}\agents\ollama\ollama.exe` (relative to `INSTALL_PATH`).

*   **`LLAMACPP_QUANTIZE_PATH`:**
    *   **Description:** Path to the `llama-quantize.exe` executable from llama.cpp. This tool is likely used to quantize language models for reduced size and potentially faster inference.
    *   **How to Configure:** Set the `LLAMACPP_QUANTIZE_PATH` environment variable to the full path to the `llama-quantize.exe` file.
    *   **Default Value:** `{INSTALL_PATH}\build\llama.cpp\build\bin\Release\llama-quantize.exe` (relative to `INSTALL_PATH`).


### 11. Working Directories

These variables specify the working directories for certain processes.

*   **`OLLAMA_WORKING_DIR`:**
    *   **Description:** The working directory for the Ollama service. This might be where Ollama stores its data, models, or temporary files.
    *   **How to Configure:** Set the `OLLAMA_WORKING_DIR` environment variable to the desired working directory for Ollama.
    *   **Default Value:** `{INSTALL_PATH}\agents\ollama` (relative to `INSTALL_PATH`).

*   **`LLAMACPP_WORKING_DIR`:**
    *   **Description:** The working directory for llama.cpp related processes (like quantization).
    *   **How to Configure:** Set the `LLAMACPP_WORKING_DIR` environment variable to the desired working directory for llama.cpp tools.
    *   **Default Value:** `{INSTALL_PATH}\build\llama.cpp` (relative to `INSTALL_PATH`).


### 12. Log Files Paths

These variables define the paths to various log files generated by the application and its components. **Generally, you don't need to change these directly unless you want to consolidate logs into a different location by modifying `LOGS_DIR`.**

*   **`LOGS_DIR` (Environment Variable - See Section 8)**: The base directory for all logs.

*   **`MONITOR_LOG`:** Path to the monitor process log file. Default: `{LOGS_DIR}\monitor.log`
*   **`CADDY_LOG`:** Path to the Caddy web server log file. Default: `{LOGS_DIR}\caddy.log`
*   **`OLLAMA_LOG`:** Path to the Ollama service log file. Default: `{LOGS_DIR}\ollama.log`
*   **`WEBUI_LOG`:** Path to the Open WebUI log file. Default: `{LOGS_DIR}\open_webui.log`
*   **`SEARXNG_LOG`:** Path to the SearXNG log file. Default: `{LOGS_DIR}\searxng.log`
*   **`VAULTWARDEN_LOG`:** Path to the Vaultwarden (password manager) log file. Default: `{LOGS_DIR}\vaultwarden.log`
*   **`OVERLAY_LOG`:** Path to the Windows overlay process log file. Default: `{LOGS_DIR}\windows_overlay.log`
*   **`TEMP_LOG`:** Path to a temporary log file. Default: `{LOGS_DIR}\temp.log`

*   **`ALL_LOGS` (Python Constant):** A list containing the paths of all the log files defined above. This is a Python constant and **not meant to be configured by the user**.


### 13. NopeCHA API Key

*   **`NOPECHA_KEY`:**
    *   **Description:** API key for the NopeCHA service, likely used for bypassing CAPTCHAs in web automation tasks.
    *   **How to Configure:** Set the `NOPECHA_KEY` environment variable to your NopeCHA API key.
    *   **Default Value:** Empty string (`''`).  Features relying on NopeCHA might not function without this key.


### 14. Default Modelfile

*   **`DEFAULT_MODELFILE`:**
    *   **Description:** Path to the default Modelfile. Modelfiles are used by Ollama to define and customize language models.
    *   **How to Configure:**  While not explicitly defined as configurable via environment variable in the code, you *could* potentially override this in your own code if needed. However, it's generally expected to use the default path.  If you need to use a different default Modelfile, consider adjusting your Ollama workflows or models directly.
    *   **Default Value:** `{INSTALL_PATH}\.agents\default.Modelfile` (relative to `INSTALL_PATH`).


---

## Important Notes

*   **Restart Services:** After changing environment variables that affect running services (like Ollama, Open WebUI, Caddy, SearXNG), you will likely need to restart those services for the changes to take effect. Refer to the application's documentation or service management scripts for instructions on restarting services.
*   **Path Separators:** On Windows, paths often use backslashes (`\`). In environment variables or configuration files, you may need to use double backslashes (`\\`) or forward slashes (`/`) in paths to avoid issues with escaping. The code provided uses raw strings (`fr''`) for Windows paths, which helps with backslash handling in Python.
*   **`INSTALLATION_DIR` is Key:** Correctly setting the `INSTALLATION_DIR` environment variable (or ensuring the `install.path` file exists) is crucial, as many other paths and configurations are derived from it.
*   **Log Files for Troubleshooting:** If you encounter issues, check the log files located in the directory specified by `LOGS_DIR`. These logs can provide valuable information for debugging problems.

This guide should help you understand and configure the Agents application effectively. If you have further questions or need more advanced customization, please consult the application's more detailed documentation or developer resources.