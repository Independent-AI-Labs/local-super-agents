# Ollama Setup for Intel Arc Cards

This temporary repository provides a minimal Windows setup for running Ollama on Intel Arc cards. It uses batch scripts to automate the installation of all required components.

## Included Scripts

- **pre_install.bat**  
  Unzips and installs pre-requisites including:
  - Visual Studio Build Tools
  - Intel oneAPI Base Kit
  - Anaconda
  - Git

- **init_python_env.bat**  
  Sets up the Python environment by:
  - Activating Miniconda
  - Creating and activating a `llm` conda environment with Python 3.11
  - Installing required Python packages (including `ipex-llm[cpp]`)
  - Initializing Ollama

## Overview

1. **Pre-requisites Installation**:  
   The `pre_install.bat` script extracts required packages, installs essential development tools, and cleans up the installer files.

2. **Environment Setup**:  
   The `init_python_env.bat` script activates the conda environment and installs Ollama-related packages, setting up the simplest Ollama configuration for Intel Arc cards.

*Note: These scripts will later be integrated into a graphical installer for Windows.*

Happy installing!