:: Verify install_path
if "%install_path%"=="" (
    echo ERROR: install_path is not set. Setting default...
    set install_path=C:
)

call "%install_path%\tools\miniconda3\condabin\conda.bat" activate

call conda activate lsa

pip install --upgrade -r "%install_path%\agents\integration\requirements.txt"
