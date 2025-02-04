@echo off
setlocal enabledelayedexpansion

:: List of specific directories to remove
set DIRS=%USERPROFILE%\.agents
set DIRS=!DIRS! %USERPROFILE%\agents
set DIRS=!DIRS! %USERPROFILE%\certs
set DIRS=!DIRS! %USERPROFILE%\.tools

:: Remove explicitly listed directories
for %%D in (!DIRS!) do (
    if exist "%%D" (
        echo Removing: %%D
        rmdir /s /q "%%D"
    ) else (
        echo Skipping: %%D (not found)
    )
)

:: Remove directories matching *-win pattern inside %USERPROFILE%\tools
for /d %%D in ("%USERPROFILE%\tools\*-win") do (
    if exist "%%D" (
        echo Removing: %%D
        rmdir /s /q "%%D"
    )
)

:: Delete each shortcut file individually
if exist "%USERPROFILE%\Desktop\Open WebUI AI Assistant.lnk" (
    echo Deleting: "%USERPROFILE%\Desktop\Open WebUI AI Assistant.lnk"
    del /f "%USERPROFILE%\Desktop\Open WebUI AI Assistant.lnk"
)

if exist "%USERPROFILE%\Desktop\Private Browsing.lnk" (
    echo Deleting: "%USERPROFILE%\Desktop\Private Browsing.lnk"
    del /f "%USERPROFILE%\Desktop\Private Browsing.lnk"
)

if exist "%USERPROFILE%\Desktop\Password Vault.lnk" (
    echo Deleting: "%USERPROFILE%\Desktop\Password Vault.lnk"
    del /f "%USERPROFILE%\Desktop\Password Vault.lnk"
)

if exist "%USERPROFILE%\Desktop\Start AI Services as Admin.lnk" (
    echo Deleting: "%USERPROFILE%\Desktop\Start AI Services as Admin.lnk"
    del /f "%USERPROFILE%\Desktop\Start AI Services as Admin.lnk"
)

echo Cleanup complete.
endlocal
pause
