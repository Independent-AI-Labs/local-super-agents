@echo off
echo Starting VibeCheck application...

REM Activate the vibecheck conda environment
call conda activate vibecheck
if %ERRORLEVEL% neq 0 (
    echo Failed to activate vibecheck environment.
    echo Please make sure you've run setup.bat first.
    exit /b %ERRORLEVEL%
)

REM Run the VibeCheck application
echo Running VibeCheck app...
python -m vibecheck.app

REM Capture the exit code
set VIBECHECK_EXIT=%ERRORLEVEL%

REM Return the exit code
if %VIBECHECK_EXIT% neq 0 (
    echo VibeCheck application exited with error code: %VIBECHECK_EXIT%
    exit /b %VIBECHECK_EXIT%
)

echo VibeCheck application closed.
