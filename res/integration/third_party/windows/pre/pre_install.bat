@echo off

echo Starting pre-requisite installation...
echo.

:: Unzip the pre-requisites.
echo Extracting pre-requisite files...
if exist "%install_path%\agents\res\integration\third_party\windows\pre\pre.7z" (
    "%ProgramFiles%\7-Zip\7z.exe" x "%install_path%\agents\res\integration\third_party\windows\pre\pre.7z" -o"%install_path%\agents\res\integration\third_party\windows\pre" -y
    ren "%install_path%\agents\res\integration\third_party\windows\pre\pre.7z.dir" "pre"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to extract pre-requisite packages pre.7z.
        pause
        exit /b
    )
    echo [OK] Extraction completed.
) else (
    echo [ERROR] Pre-requisites not found.
    pause
    exit /b
)
echo.

:: Define the path to the extracted installers
set "PRE_PATH=%install_path%\agents\res\integration\third_party\windows\pre\pre"

:: Install Anaconda
echo Installing Anaconda...
"%PRE_PATH%\Anaconda3-2024.10-1-Windows-x86_64.exe" /InstallationType=JustMe /RegisterPython=0 /S /D=%install_path%\tools\miniconda3
echo [OK] Anaconda installed.
echo.

:: Install Git
echo Installing Git...
"%PRE_PATH%\Git-2.48.0-rc2-64-bit.exe" /VERYSILENT /NORESTART
echo [OK] Git installed.
echo.

:: Remove installer files
echo Cleaning up installation files...
del /Q "%PRE_PATH%\*.exe"
del /Q "%PRE_PATH%\pre"
del /Q "%install_path%\agents\res\integration\third_party\windows\pre\pre.7z"
echo [OK] Installer files removed.
echo.

echo All installations completed successfully!