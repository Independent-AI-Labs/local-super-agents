:: Verify install_path
if "%install_path%"=="" (
    echo ERROR: install_path is not set. Setting default...
    set install_path=C:
)


call "%install_path%\tools\miniconda3\condabin\conda.bat" activate

call conda create -n llm python=3.11 libuv -y
call conda activate llm

pip install --pre ipex-llm[cpp]==2.2.0b20250123

mkdir "%install_path%\agents\ollama"
cd "%install_path%\agents\ollama"

init-ollama

call conda create -n lsa python=3.12
call conda activate lsa

pip install -r "%install_path%\agents\mvp\requirements.txt"