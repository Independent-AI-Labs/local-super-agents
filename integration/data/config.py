import os

NOPECHA_KEY = os.getenv('NOPECHA_KEY', '')

GDRIVE_CLIENT_ID = os.getenv('GDRIVE_CLIENT_ID', '')
GDRIVE_CLIENT_SECRET = os.getenv('GDRIVE_CLIENT_SECRET', '')

AGENTS_VM_ADDRESS = os.getenv('VM_ADDRESS', 'http://172.72.72.2')

OLLAMA_PORT = os.getenv('OLLAMA_PORT', 11434)
OPEN_WEBUI_PORT = os.getenv('OPEN_WEBUI_PORT', 8080)
OPEN_WEBUI_BASE_URL = os.getenv('OPEN_WEBUI_BASE_URL', f'http://127.0.0.1:{OPEN_WEBUI_PORT}')
OPEN_WEBUI_EMAIL = os.getenv('OPEN_WEBUI_EMAIL', '')
OPEN_WEBUI_PASSWORD = os.getenv('OPEN_WEBUI_PASSWORD', '')

OPEN_WEBUI_EXT_PORT = 8443  # The external port to serve HTTPS traffic
SSL_CERT_FILE = os.getenv('SSL_CERT_FILE', r'C:\certs\certificate.crt')
SSL_KEY_FILE = os.getenv('SSL_KEY_FILE', r'C:\certs\private.key')

SEARXNG_BASE_URL = os.getenv('SEARXNG_BASE_URL', f'{AGENTS_VM_ADDRESS}:8888')
CHAT_ID = os.getenv('CHAT_ID', '')
 
OLLAMA_ENV = os.getenv('DEFAULT_ENV', 'llm')
DEFAULT_ENV = os.getenv('DEFAULT_ENV', 'lsa')
DEFAULT_LLM = os.getenv('DEFAULT_LLM', 'internlm_7b')

CACHE_DIR = os.getenv('CACHE_DIR', '')
LOGS_DIR = os.getenv('LOGS_DIR', r"C:\agents\logs")

DESKTOP_OVERLAY_TARGET_FPS = 240
DESKTOP_OVERLAY_TARGET_FRAME_TIME = 1.0 / float(DESKTOP_OVERLAY_TARGET_FPS)

# Adjust paths as needed
ONE_API_PATH = os.getenv('ONE_API_PATH', r'C:\Program Files (x86)\Intel\oneAPI\setvars.bat')
CONDA_PATH = os.getenv('CONDA_PATH', r'C:\tools\miniconda3\condabin\conda.bat')
CADDY_PATH = os.getenv('CADDY_PATH', r"C:\tools\caddy-win\caddy.exe")
CADDYFILE_PATH = os.getenv('CADDY_PATH', r"C:\tools\caddy-win\caddyfile")
CHROME_PATH = os.getenv('CHROME_PATH', r"C:\tools\chrome-win\chrome.exe")
OLLAMA_PATH = os.getenv('OLLAMA_PATH', r"C:\agents\ollama\ollama.exe")
OLLAMA_WORKING_DIR = os.getenv('OLLAMA_WORKING_DIR', r"C:\agents\ollama")

MONITOR_LOG = os.path.join(LOGS_DIR, "monitor.log")
CADDY_LOG = os.path.join(LOGS_DIR, "caddy.log")
OLLAMA_LOG = os.path.join(LOGS_DIR, "ollama.log")
WEBUI_LOG = os.path.join(LOGS_DIR, "open_webui.log")
SEARXNG_LOG = os.path.join(LOGS_DIR, "searxng.log")
VAULTWARDEN_LOG = os.path.join(LOGS_DIR, "vaultwarden.log")
OVERLAY_LOG = os.path.join(LOGS_DIR, "windows_overlay.log")

ALL_LOGS = [MONITOR_LOG, CADDY_LOG, OLLAMA_LOG, WEBUI_LOG, SEARXNG_LOG, VAULTWARDEN_LOG, OVERLAY_LOG]
