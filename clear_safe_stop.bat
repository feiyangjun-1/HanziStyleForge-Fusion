@echo off
setlocal
cd /d "%~dp0"
if exist STOP_AFTER_CHECKPOINT del /q STOP_AFTER_CHECKPOINT
echo Safe-stop request cleared. Run run_months_resilient.bat to resume.
pause
