@echo off
setlocal
cd /d "%~dp0"
type nul > STOP_AFTER_CHECKPOINT
echo Safe-stop request created. The running job will stop only after the next durable checkpoint.
pause
