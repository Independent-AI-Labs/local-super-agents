import socket
import requests
import sys
import time
import platform
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("network_diagnostic")

def check_port_available(port, host='127.0.0.1'):
    """Check if a port is available"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return False  # Port is in use
    except (socket.timeout, ConnectionRefusedError):
        return True  # Port is available

def check_internet_connection():
    """Check if there's a connection to the internet"""
    try:
        requests.get('https://www.google.com', timeout=3)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

def ping_host(host):
    """Ping a host and return True if it responds"""
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '1', host]
    try:
        return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except:
        return False

def check_gradio_hosts():
    """Check connectivity to Gradio and HuggingFace hosts"""
    hosts = [
        'api.gradio.app',
        'huggingface.co',
    ]
    
    results = {}
    for host in hosts:
        ping_result = ping_host(host)
        http_result = False
        try:
            requests.head(f'https://{host}', timeout=3)
            http_result = True
        except:
            pass
        
        results[host] = {
            'ping': ping_result,
            'http': http_result
        }
    
    return results

def run_diagnostics():
    """Run network diagnostics"""
    logger.info("Starting network diagnostics...")
    
    # Check if ports are available
    port_check = check_port_available(7860)
    logger.info(f"Port 7860 available: {port_check}")
    
    # Check internet connection
    internet = check_internet_connection()
    logger.info(f"Internet connection: {internet}")
    
    # Check specific host connectivity
    if internet:
        logger.info("Checking Gradio and HuggingFace host connectivity...")
        host_results = check_gradio_hosts()
        for host, result in host_results.items():
            logger.info(f"  {host}: Ping: {result['ping']}, HTTP: {result['http']}")
    
    # Check local network listeners
    logger.info("Local network listeners on port 7860:")
    if platform.system().lower() == 'windows':
        os.system('netstat -ano | findstr :7860')
    else:
        os.system('netstat -tuln | grep 7860')
    
    # Check firewall status
    logger.info("Checking firewall status...")
    if platform.system().lower() == 'windows':
        os.system('netsh advfirewall show allprofiles state')
    else:
        os.system('sudo ufw status')
    
    logger.info("Network diagnostics complete.")

if __name__ == "__main__":
    run_diagnostics()
