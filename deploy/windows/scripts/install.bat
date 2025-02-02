@echo off
setlocal

set "GUEST_MODE=false"

:: Function to check for administrative privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Administrative permissions required. Please run this script as an administrator.
    pause
    exit /b
)


if %GUEST_MODE% neq true (
	:: Enable Hyper-V feature
	echo Enabling Hyper-V feature...
	powershell -Command "Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -All -LimitAccess"
	if %errorlevel% neq 0 (
		echo Failed to enable Hyper-V. Please ensure your system supports Hyper-V and try again.
		pause
		exit /b
	)
	echo Hyper-V enabled successfully.
	echo.
	pause
)


:: Prompt user for installation path with default to C:\
set "install_path=C:"
set /p "install_path=Enter the installation path [default: C:\]: "
if "%install_path%"=="" set "install_path=C:"

:: Confirm the installation path
echo You have chosen to install to: %install_path%
echo.
pause

:: Navigate to the installation path
cd /d "%install_path%"
if %errorlevel% neq 0 (
    echo Failed to navigate to %install_path%. Please ensure the path exists.
    pause
    exit /b
)

:: Write the installation path to C:\.agents\install.path
if not exist C:\.agents mkdir C:\.agents
echo %install_path% > C:\.agents\install.path
echo Installation path recorded in C:\.agents\install.path
echo.
pause


:: Silent install of 7z2409-x64.exe
echo Installing 7-Zip...
if exist "7z2409-x64.exe" (
    ".\7z2409-x64.exe" /S
    if %errorlevel% neq 0 (
        echo Failed to install 7-Zip.
        pause
        exit /b
    )
    echo 7-Zip installed successfully.
) else (
    echo 7z2409-x64.exe not found.
    pause
    exit /b
)
echo.
pause


:: Create directories 'agents' and 'certs'
echo Creating directories 'agents' and 'certs'...
mkdir "%install_path%\agents"
mkdir "%install_path%\certs"
echo Directories created.
echo.
pause


:: Extract integration.7z into 'agents' directory
echo Extracting integration.7z into 'agents' directory...
if exist "integration.7z" (
    "%ProgramFiles%\7-Zip\7z.exe" x "integration.7z" -o"%install_path%\agents" -y
    if %errorlevel% neq 0 (
        echo Failed to extract integration.7z.
        pause
        exit /b
    )
    echo Extraction completed.
) else (
    echo integration.7z not found in %install_path%.
    pause
    exit /b
)
echo.
pause


:: Extract web.7z
echo Extracting web.7z...
if exist "%install_path%\agents\integration\res\third_party\windows\run\web.7z" (
    "%ProgramFiles%\7-Zip\7z.exe" x "%install_path%\agents\integration\res\third_party\windows\run\web.7z" -o"%install_path%\tools" -y
    if %errorlevel% neq 0 (
        echo Failed to extract web.7z.
        pause
        exit /b
    )
    echo Extraction completed.
) else (
    echo web.7z not found.
    pause
    exit /b
)
echo.
pause


if %GUEST_MODE% neq true (
	:: Extract HyperV.7z.001-00X into 'vm' directory
	echo Extracting HyperV.7z.001-00X into 'vm' directory...
	if exist "%install_path%\agents\integration\res\third_party\windows\vm\HyperV.7z.001" (
		"%ProgramFiles%\7-Zip\7z.exe" x "%install_path%\agents\integration\res\third_party\windows\vm\HyperV.7z.001" -o"%install_path%\agents\integration\res\third_party\windows\vm" -y
		if %errorlevel% neq 0 (
			echo Failed to extract HyperV.7z.001-00X.
			pause
			exit /b
		)
		echo Extraction completed.
	) else (
		echo HyperV.7z.001 not found.
		pause
		exit /b
	)
	echo.
	pause

	:: Run vm_import.ps1
	echo Running vm_import.ps1...
	if exist "%install_path%\agents\integration\res\third_party\windows\vm\vm_import.ps1" (
		powershell -ExecutionPolicy Bypass -File "%install_path%\agents\integration\res\third_party\windows\vm\vm_import.ps1"
		if %errorlevel% neq 0 (
			echo vm_import.ps1 encountered an error.
			pause
			exit /b
		)
		echo vm_import.ps1 executed successfully.
	) else (
		echo vm_import.ps1 not found.
		pause
		exit /b
	)
	echo.
	pause

	:: Remove HyperV directory
	echo Removing HyperV directory...
	rmdir /s /q "%install_path%\agents\integration\res\third_party\windows\vm\HyperV"
	echo HyperV directory removed.
	echo.
	pause
)


:: Run pre_install.bat
echo Running pre_install.bat...
if exist "%install_path%\agents\integration\res\third_party\windows\pre\pre_install.bat" (
    call "%install_path%\agents\integration\res\third_party\windows\pre\pre_install.bat"
    if %errorlevel% neq 0 (
        echo pre_install.bat encountered an error.
        pause
        exit /b
    )
    echo pre_install.bat executed successfully.
) else (
    echo pre_install.bat not found.
    pause
    exit /b
)
echo.
pause


:: Run init_python_env.bat
echo Running init_python_env.bat...
if exist "%install_path%\agents\integration\res\third_party\windows\envs\init_python_env.bat" (
    call "%install_path%\agents\integration\res\third_party\windows\envs\init_python_env.bat"
    if %errorlevel% neq 0 (
        echo init_python_env.bat encountered an error.
        pause
        exit /b
    )
    echo init_python_env.bat executed successfully.
) else (
    echo init_python_env.bat not found.
    pause
    exit /b
)
echo.
pause


echo
::contentReference[oaicite:0]{index=0}
