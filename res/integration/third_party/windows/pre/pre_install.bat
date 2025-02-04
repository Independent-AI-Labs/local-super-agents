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

:: Install Visual Studio Build Tools
echo Installing Visual Studio Build Tools...
"%PRE_PATH%\vs_BuildTools.exe" --quiet --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CoreBuildTools --add Microsoft.VisualStudio.Component.VC.143.x86.x64 --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.CMake.Tools --add Microsoft.VisualStudio.Component.Windows11SDK.26100
echo [OK] Visual Studio Build Tools installed.
echo.

:: Install Intel BaseKit. This is included only in case there are issues with the automated install via ipex-llm.
:: ipex-llm has been set to a specific version that is confirmed to include the required oneAPI binaries, but still.
:: echo Installing Intel oneAPI Base Kit...
:: "%PRE_PATH%\w_BaseKit_p_2024.2.1.101_offline" -a --silent --eula accept
:: echo [OK] Intel oneAPI Base Kit installed.
:: echo.

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