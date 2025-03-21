import os
from pathlib import Path

# TODO Replace with Windows registry.
# Define the default path file
INSTALL_PATH_FILE = os.path.join(Path.home(), '.agents', 'install.path')


# Read the default installation path from the file if it exists
def get_default_install_path():
    if os.path.exists(INSTALL_PATH_FILE):
        with open(INSTALL_PATH_FILE, 'r') as f:
            return f.read().strip()
    return None  # Return None if the file doesn't exist


INSTALL_PATH = os.getenv('INSTALLATION_DIR', get_default_install_path())

GDRIVE_CLIENT_ID = os.getenv('GDRIVE_CLIENT_ID', '')
GDRIVE_CLIENT_SECRET = os.getenv('GDRIVE_CLIENT_SECRET', '')

AGENTS_VM_ADDRESS = os.getenv('VM_ADDRESS', 'http://172.72.72.2')

OLLAMA_INTEL = os.getenv('OLLAMA_INTEL', False)
OLLAMA_PORT = os.getenv('OLLAMA_PORT', 11434)
OLLAMA_BASE_URL = os.getenv('OPEN_WEBUI_BASE_URL', f'http://127.0.0.1:{OLLAMA_PORT}')
OPEN_WEBUI_PORT = os.getenv('OPEN_WEBUI_PORT', 8080)
OPEN_WEBUI_BASE_URL = os.getenv('OPEN_WEBUI_BASE_URL', f'http://127.0.0.1:{OPEN_WEBUI_PORT}')
OPEN_WEBUI_EMAIL = os.getenv('OPEN_WEBUI_EMAIL', 'v@test.com')
OPEN_WEBUI_PASSWORD = os.getenv('OPEN_WEBUI_PASSWORD', 'test')

OPEN_WEBUI_EXT_PORT = 8443  # The external port to serve HTTPS traffic
WEBUI_SSL_CERT_FILE = os.getenv('WEBUI_SSL_CERT_FILE', fr'{INSTALL_PATH}\certs\certificate.crt')
WEBUI_SSL_KEY_FILE = os.getenv('WEBUI_SSL_KEY_FILE', fr'{INSTALL_PATH}\certs\private.key')

SEARXNG_BASE_URL = os.getenv('SEARXNG_BASE_URL', f'{AGENTS_VM_ADDRESS}:8888')
CHAT_ID = os.getenv('CHAT_ID', '')

OLLAMA_ENV = os.getenv('OLLAMA_ENV', 'llm')
DEFAULT_ENV = os.getenv('DEFAULT_ENV', 'lsa')
DEFAULT_LLM = os.getenv('DEFAULT_LLM', 'qwen2.5-coder:14b')
DEFAULT_PIPELINE = os.getenv('DEFAULT_PIPELINE', 'web_search_pipeline')

CACHE_DIR = os.getenv('CACHE_DIR', '')
LOGS_DIR = os.getenv('LOGS_DIR', fr'{INSTALL_PATH}\.agents\logs')
INDEXED_DATA_DIR = os.getenv('INDEXED_DATA_DIR', fr'{INSTALL_PATH}\.agents\indexed_data')

DESKTOP_OVERLAY_TARGET_FPS = 240
DESKTOP_OVERLAY_TARGET_FRAME_TIME = 1.0 / float(DESKTOP_OVERLAY_TARGET_FPS)

# Adjust paths as needed
ONE_API_PATH = os.getenv('ONE_API_PATH', fr'{INSTALL_PATH}\Program Files (x86)\Intel\oneAPI\setvars.bat')
CONDA_PATH = os.getenv('CONDA_PATH', fr'{INSTALL_PATH}\tools\miniconda3\condabin\conda.bat')
CADDY_PATH = os.getenv('CADDY_PATH', fr'{INSTALL_PATH}\tools\caddy-win\caddy.exe')
CADDYFILE_PATH = os.getenv('CADDY_PATH', fr'{INSTALL_PATH}\tools\caddy-win\caddyfile')
CHROME_PATH = os.getenv('CHROME_PATH', fr'{INSTALL_PATH}\tools\chrome-win\chrome-agents.exe')
OLLAMA_PATH = os.getenv('OLLAMA_PATH', fr'{INSTALL_PATH}\agents\ollama\ollama.exe')
LLAMACPP_IMATRIX_PATH = os.getenv('LLAMACPP_IMATRIX_PATH', fr'{INSTALL_PATH}\build\llama.cpp\build\bin\Release\llama-imatrix.exe')
LLAMACPP_QUANTIZE_PATH = os.getenv('LLAMACPP_QUANTIZE_PATH', fr'{INSTALL_PATH}\build\llama.cpp\build\bin\Release\llama-quantize.exe')
OLLAMA_WORKING_DIR = os.getenv('OLLAMA_WORKING_DIR', fr'{INSTALL_PATH}\agents\ollama')
LLAMACPP_WORKING_DIR = os.getenv('LLAMACPP_WORKING_DIR', fr'{INSTALL_PATH}\build\llama.cpp')

EXTENSIONS_PATH = os.getenv('EXTENSIONS_PATH', fr'{INSTALL_PATH}\agents\res\integration\third_party\extensions')

DEBUG_LOG = os.path.join(LOGS_DIR, 'debug.log')
USER_LOG = os.path.join(LOGS_DIR, 'user.log')
MONITOR_LOG = os.path.join(LOGS_DIR, 'monitor.log')
CADDY_LOG = os.path.join(LOGS_DIR, 'caddy.log')
OLLAMA_LOG = os.path.join(LOGS_DIR, 'ollama.log')
WEBUI_LOG = os.path.join(LOGS_DIR, 'open_webui.log')
SEARXNG_LOG = os.path.join(LOGS_DIR, 'searxng.log')
VAULTWARDEN_LOG = os.path.join(LOGS_DIR, 'vaultwarden.log')
OVERLAY_LOG = os.path.join(LOGS_DIR, 'windows_overlay.log')
TEMP_LOG = os.path.join(LOGS_DIR, 'temp.log')

ALL_LOGS = [MONITOR_LOG, CADDY_LOG, OLLAMA_LOG, WEBUI_LOG, SEARXNG_LOG, VAULTWARDEN_LOG, OVERLAY_LOG, TEMP_LOG]

NOPECHA_KEY = os.getenv('NOPECHA_KEY', '')
DEFAULT_MODELFILE = fr'{INSTALL_PATH}\.agents\default.Modelfile'
KGML_SYSTEM_PROMPT = fr'{INSTALL_PATH}\agents\knowledge\reasoning\prompts\KGML'

WEB_SEARCH_MODEL = os.getenv('WEB_SEARCH_MODEL', 'qwen2.5-coder:14b')
KGML_MODEL = os.getenv('KGML_MODEL', 'qwen2.5-coder:14b')
PYTHON_INSTR_MODEL = os.getenv('PYTHON_INSTR_MODEL', 'qwen2.5-coder:14b')
TOP_N_WEB_SEARCH_RESULTS = os.getenv('TOP_N_WEB_SEARCH_RESULTS', 5)
DEFAULT_WEB_PAGE_TIMEOUT = os.getenv('DEFAULT_WEB_PAGE_TIMEOUT', 5)
MAX_WEB_SCRAPE_WORKERS_PER_SEARCH = os.getenv('MAX_WEB_SCRAPE_WORKERS_PER_SEARCH', 8)
