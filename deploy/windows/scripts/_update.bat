@echo off
setlocal

REM Read the installation path from the file (assumes the file contains a single line)
for /f "usebackq delims=" %%A in ("%USERPROFILE%\.agents\install.path") do set "install_path=%%A"

:: Function to check for administrative privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Administrative permissions required. Please run this script as an administrator.
    pause
	echo.
    exit /b
)


if %GUEST_MODE% neq true (
	echo Before we proceed, we need to enable Windows Virtualization.
REM	pause
	echo.
	
	:: Enable Hyper-V feature
	echo Enabling Hyper-V feature...
	powershell -Command "Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -All -LimitAccess"
	if %errorlevel% neq 0 (
		echo Failed to enable Hyper-V. Please ensure your system supports Hyper-V and try again.
		pause
		echo.
		exit /b
	)
	echo Hyper-V enabled successfully.
	echo.
REM	pause
	echo.
)

:: Verify install_path
if "%install_path%"=="" (
    echo ERROR: Application not installed! Please run install.bat instead!
    pause
	echo.
    exit /b
)

:: Navigate to the installation path
cd /d "%install_path%"
if %errorlevel% neq 0 (
    echo Failed to navigate to %install_path%. Please ensure the path exists.
    pause
    exit /b
)

:: Confirm the installation path
echo Detected existing installation: [%install_path%]
echo.
REM pause
echo.

:: Ensure directory exists
if not exist %USERPROFILE%\.agents mkdir %USERPROFILE%\.agents

:: Copy integration into 'agents' directory
echo Deploying integration...
xcopy "%~dp0integration\" "%install_path%\agents\integration\" /E /H /C /I /Y

:: Copy res into 'agents' directory
echo Deploying resources...
xcopy "%~dp0res\" "%install_path%\agents\res\" /E /H /C /I /Y


:: TODO Archive extractions.


if %GUEST_MODE% neq true (
    :: Extract AgentsVM.7z into 'vm' directory
    echo Extracting AgentsVM.7z into 'vm' directory...
    if exist "%install_path%\agents\res\integration\third_party\windows\vm\AgentsVM.7z" (
        "%ProgramFiles%\7-Zip\7z.exe" x "%install_path%\agents\res\integration\third_party\windows\vm\AgentsVM.7z" -o"%install_path%\agents\res\integration\third_party\windows\vm" -y
        if %errorlevel% neq 0 (
            echo Failed to extract AgentsVM.7z.
            pause
            echo.
            exit /b
        )
        echo Extraction completed.

        :: Rename extracted folder from AgentsVM.7z.dir to AgentsVM
        if exist "%install_path%\agents\res\integration\third_party\windows\vm\AgentsVM.7z.dir" (
            ren "%install_path%\agents\res\integration\third_party\windows\vm\AgentsVM.7z.dir" "AgentsVM"
            echo Renamed extracted folder to 'AgentsVM'.
        ) else (
            echo Extracted folder AgentsVM.7z.dir not found.
        )

        :: Delete the original .7z file
        del /q "%install_path%\agents\res\integration\third_party\windows\vm\AgentsVM.7z"
        echo Deleted AgentsVM.7z.
    ) else (
        echo AgentsVM.7z not found.
        pause
        echo.
        exit /b
    )
    echo.


	:: Run vm_import.ps1
	echo Running vm_import.ps1...
	if exist "%install_path%\agents\res\integration\third_party\windows\vm\vm_import.ps1" (
		powershell -ExecutionPolicy Bypass -File "%install_path%\agents\res\integration\third_party\windows\vm\vm_import.ps1"
		if %errorlevel% neq 0 (
			echo vm_import.ps1 encountered an error.
			pause
			echo.
			exit /b
		)
		echo vm_import.ps1 executed successfully.
	) else (
		echo vm_import.ps1 not found.
		pause
		echo.
		exit /b
	)
	echo.

	:: Remove AgentsVM directory
	echo Removing AgentsVM directory...
	rmdir /s /q "%install_path%\agents\res\integration\third_party\windows\vm\AgentsVM"
	echo AgentsVM directory removed.
	echo.
)


:: TODO 3rd-party app updates.


:: Run init_python_env.bat
echo Running update_python_env.bat...
if exist "%install_path%\agents\res\integration\third_party\windows\envs\update_python_env.bat" (
    call "%install_path%\agents\res\integration\third_party\windows\envs\update_python_env.bat"
    if %errorlevel% neq 0 (
        echo update_python_env.bat encountered an error.
        pause
		echo.
        exit /b
    )
    echo update_python_env.bat executed successfully.
) else (
    echo update_python_env.bat not found.
    pause
	echo.
    exit /b
)
echo.
REM pause
echo.


:: TODO Create additional shortcuts here.


echo.
echo Update completed!
echo.
pause

echo

