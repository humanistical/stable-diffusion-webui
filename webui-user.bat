@echo off

set PYTHON="C:\Users\tjsul\AppData\Local\Microsoft\WindowsApps\python3.10.exe"
set GIT=C:\Users\tjsul\local\Git\bin\git.exe
set VENV_DIR=C:\Users\tjsul\Repos\stable-diffusion-webui\venv
set COMMANDLINE_ARGS=--xformers --no-half --no-half-vae --update-check --update-all-extensions
set ACCELERATE=True
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512

call webui.bat
