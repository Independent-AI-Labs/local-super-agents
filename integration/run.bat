REM Read the installation path from the file (assumes the file contains a single line)
for /f "usebackq delims=" %%A in ("%USERPROFILE%\.agents\install.path") do set "install_path=%%A"

REM Optional: Check if install_path was read successfully
if "%install_path%"=="" (
    echo Error: install_path is empty. Please check C:\.agents\install.path.
    pause
    exit /b 1
)


call "%install_path%\tools\miniconda3\condabin\conda.bat" activate
call conda activate lsa

cd "%install_path%\agents\"
set "PYTHONPATH=%install_path%\agents"
python integration\manage\launcher.py