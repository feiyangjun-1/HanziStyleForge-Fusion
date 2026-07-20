@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :not_installed
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-status
pause
exit /b 0
:not_installed
echo Run install_cuda130.bat first.
pause
exit /b 1
