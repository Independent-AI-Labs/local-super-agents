:: Verify install_path
if "%install_path%"=="" (
    echo ERROR: install_path is not set. Setting default...
    set install_path=C:
)


call "%install_path%\tools\miniconda3\condabin\conda.bat" activate

call conda create -n llm python=3.11 libuv -y
call conda activate llm

pip install --pre --upgrade ipex-llm[cpp] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

mkdir "%install_path%\agents\ollama"
cd "%install_path%\agents\ollama"

init-ollama
