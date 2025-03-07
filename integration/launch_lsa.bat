@echo off
REM Read the installation path from the file (assumes the file contains a single line)
for /f "usebackq delims=" %%A in ("%USERPROFILE%\.agents\install.path") do set "install_path=%%A"

REM Optional: Check if install_path was read successfully
if "%install_path%"=="" (
    echo Error: install_path is empty. Please check %USERPROFILE%\.agents\install.path
    pause
    exit /b 1
)

REM Check if the agents ollama directory exists and is not empty (Intel install)
if exist "%install_path%\agents\ollama\*" (
    set OLLAMA_INTEL=true
    echo OLLAMA_INTEL set to true.
)

REM Activate Conda environment
call "%install_path%\tools\miniconda3\condabin\conda.bat" activate
call conda activate lsa

REM Navigate to agents directory
cd "%install_path%\agents\"
set "PYTHONPATH=%install_path%\agents"

REM Launch the Python script
python integration\manage\launcher.py
