@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title HanziStyleForge Fusion 2.2 - Installer

echo [1/7] Checking Python...
where py >nul 2>nul
if %errorlevel%==0 (
  set "BASEPY=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 goto :no_python
  set "BASEPY=python"
)
%BASEPY% -c "import sys; ok=(3,10)<=sys.version_info[:2]<=(3,14); print('Python:',sys.version); raise SystemExit(0 if ok else 2)"
if errorlevel 1 goto :bad_python

echo [2/7] Creating isolated environment...
if not exist ".venv\Scripts\python.exe" %BASEPY% -m venv .venv
if not exist ".venv\Scripts\python.exe" goto :failed
set "PY=.venv\Scripts\python.exe"

echo [3/7] Updating pip...
%PY% -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

echo [4/7] Installing font and image dependencies...
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [5/7] Installing official PyTorch CUDA 13.0 wheels...
%PY% -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
%PY% -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :fallback_cuda128
%PY% -c "import torch; print('PyTorch:',torch.__version__); print('CUDA runtime:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); raise SystemExit(0 if torch.cuda.is_available() else 3)"
if errorlevel 1 goto :fallback_cuda128
goto :selftest

:fallback_cuda128
echo CUDA 13.0 installation or verification failed.
echo Falling back to the official PyTorch 2.11 CUDA 12.8 build...
%PY% -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
%PY% -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
%PY% -c "import torch; print('PyTorch:',torch.__version__); print('CUDA runtime:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); raise SystemExit(0 if torch.cuda.is_available() else 3)"
if errorlevel 1 goto :cuda_failed

:selftest
echo [6/7] Running internal self-test...
%PY% selftest.py
if errorlevel 1 goto :failed

echo [7/7] Installation completed successfully.
echo Next: copy fonts\target.ttf and refs\ref.otf, then run verify_project.bat.
pause
exit /b 0

:no_python
echo Python was not found. Install 64-bit Python 3.10 through 3.14, then run this file again.
pause
exit /b 1

:bad_python
echo This project requires 64-bit Python 3.10 through 3.14.
pause
exit /b 1

:cuda_failed
echo PyTorch was installed, but CUDA is unavailable. Update the NVIDIA driver and run this installer again.
pause
exit /b 1

:failed
echo Installation failed. Review the error above.
pause
exit /b 1
