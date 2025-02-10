@echo off
setlocal

set "GUEST_MODE=true"

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


:: Prompt user for installation path with default to User dir.
set "install_path=%USERPROFILE%"
set /p "install_path=Enter installation path (press enter for %USERPROFILE%) "
if "%install_path%"=="" set "install_path=%USERPROFILE%"

:: Confirm the installation path
echo You have chosen to install to: [%install_path%]
echo.
REM pause
echo.

:: Navigate to the installation path
cd /d "%install_path%"
if %errorlevel% neq 0 (
    echo Failed to navigate to %install_path%. Please ensure the path exists.
    pause
    exit /b
)

:: Ensure directory exists
if not exist %USERPROFILE%\.agents mkdir %USERPROFILE%\.agents

:: Write the installation path to the file WITHOUT trailing spaces/newline
(
    for /f "delims=" %%A in ("%install_path%") do @echo %%A
) > %USERPROFILE%\.agents\install.path

echo Installation path recorded in C:\.agents\install.path
echo.
REM pause
echo.

:: Silent install of 7z2409-x64.exe
echo Installing 7-Zip...
if exist "%~dp07z2409-x64.exe" (
    "%~dp07z2409-x64.exe" /S
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
REM pause
echo.


:: Create directories 'agents' and 'certs'
echo Creating directories 'agents' and 'certs'...
mkdir "%install_path%\agents"
mkdir "%install_path%\certs"
echo Directories created.
echo.
REM pause
echo.

:: Copy integration into 'agents' directory
echo Deploying integration...
xcopy "%~dp0integration\" "%install_path%\agents\integration\" /E /H /C /I /Y

:: Copy res into 'agents' directory
echo Deploying resources...
xcopy "%~dp0res\" "%install_path%\agents\res\" /E /H /C /I /Y


:: Extract web.7z
echo Extracting web.7z...
if exist "%install_path%\agents\res\integration\third_party\windows\run\web.7z" (
    "%ProgramFiles%\7-Zip\7z.exe" x "%install_path%\agents\res\integration\third_party\windows\run\web.7z" -o"%install_path%\tools" -y
    xcopy /E /I /Y "%install_path%\tools\web.7z.dir\*" "%install_path%\tools\"
    rd /S /Q "%install_path%\tools\web.7z.dir"
    if %errorlevel% neq 0 (
        echo Failed to extract web.7z.
        pause
		vecho.
        exit /b
    )
    echo Extraction completed.

    :: Delete the original .7z file
    del /q "%install_path%\agents\res\integration\third_party\windows\run\web.7z"
    echo Deleted web.7z.
) else (
    echo web.7z not found.
    pause
	echo.
    exit /b
)
echo.
REM pause
echo.


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


:: Run pre_install.bat
echo Running pre_install.bat...
if exist "%install_path%\agents\res\integration\third_party\windows\pre\pre_install.bat" (
    call "%install_path%\agents\res\integration\third_party\windows\pre\pre_install.bat"
    if %errorlevel% neq 0 (
        echo pre_install.bat encountered an error.
        pause
		echo.
        exit /b
    )
    echo pre_install.bat executed successfully.
) else (
    echo pre_install.bat not found.
    pause
	echo.
    exit /b
)
echo.
REM pause
echo.


:: Run init_python_env.bat
echo Running init_python_env.bat...
if exist "%install_path%\agents\res\integration\third_party\windows\envs\init_python_env.bat" (
    call "%install_path%\agents\res\integration\third_party\windows\envs\init_python_env.bat"
    if %errorlevel% neq 0 (
        echo init_python_env.bat encountered an error.
        pause
		echo.
        exit /b
    )
    echo init_python_env.bat executed successfully.
) else (
    echo init_python_env.bat not found.
    pause
	echo.
    exit /b
)
echo.
REM pause
echo.


@echo off
echo Creating Desktop shortcuts...

REM Create “Chatbot” shortcut:
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\Open WebUI AI Assistant.lnk'); $s.TargetPath='%install_path%\tools\brave-win\brave-agents.exe'; $s.Arguments='--disable-gpu --app="http://127.0.0.1:8080" --user-data-dir="%install_path%\.tools\bravedata"'; $s.IconLocation='%install_path%\agents\res\integration\graphics\icons\chatbot.ico'; $s.Save()"

if %GUEST_MODE% neq true (
    REM Create “MagnifyingGlass” shortcut:
    powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\Private Browsing.lnk'); $s.TargetPath='%install_path%\tools\brave-win\brave-agents.exe'; $s.Arguments='--disable-gpu --new-window "http://172.72.72.2:8888" --user-data-dir="%install_path%\.tools\bravedata"'; $s.IconLocation='%install_path%agents\res\integration\graphics\icons\magnifying-glass.ico'; $s.Save()"

    REM Create “Padlock” shortcut:
    powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\Password Vault.lnk'); $s.TargetPath='%install_path%\tools\brave-win\brave-agents.exe'; $s.Arguments='--disable-gpu --app="https://172.72.72.2:8000" --user-data-dir="%install_path%\.tools\bravedata"'; $s.IconLocation='%install_path%\agents\res\integration\graphics\icons\padlock.ico'; $s.Save()"
)

REM Create “Run Integration” shortcut:
REM This shortcut runs the batch file via CMD (/c runs the command then exits)
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\Start AI Services as Admin.lnk'); $s.TargetPath='%COMSPEC%'; $s.Arguments='/c \"%install_path%\agents\integration\run.bat"'; $s.WorkingDirectory='%install_path%\agents\integration'; $s.WindowStyle=7; $s.Save()"

echo.
echo Shortcuts created. Run the AI Services as Admin before launching the AI Assistant!
echo This will be replaced by an automated Windows service in the next release.
echo.
pause

echo
::contentReference[oaicite:0]{index=0}
