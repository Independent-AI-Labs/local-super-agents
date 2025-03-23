@echo off
echo Creating and setting up vibecheck conda environment...

REM Create a new conda environment named vibecheck with Python 3.10
call conda create -n vibecheck python=3.11 -y
if %ERRORLEVEL% neq 0 (
    echo Failed to create conda environment.
    exit /b %ERRORLEVEL%
)

REM Activate the vibecheck environment
call conda activate vibecheck
if %ERRORLEVEL% neq 0 (
    echo Failed to activate vibecheck environment.
    exit /b %ERRORLEVEL%
)

REM Install dependencies from requirements.txt
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install requirements.
    exit /b %ERRORLEVEL%
)

REM Change to the knowledge/retrieval directory and run setup
cd ..\knowledge\retrieval\
if %ERRORLEVEL% neq 0 (
    echo Failed to change directory to ..\knowledge\retrieval\
    exit /b %ERRORLEVEL%
)

REM Run the setup script in the retrieval directory
call run_to_setup.bat
if %ERRORLEVEL% neq 0 (
    echo Failed to run setup script in retrieval directory.
    exit /b %ERRORLEVEL%
)

REM Return to the original directory
cd ..\..\vibecheck\

echo.
echo VibeCheck environment setup completed successfully!
echo To activate the environment, run: conda activate vibecheck
echo To start the application use the appropriate run script!
