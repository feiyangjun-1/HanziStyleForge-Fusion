@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title HanziStyleForge Fusion - Months Mode
if not exist ".venv\Scripts\python.exe" goto :not_installed
if exist "STOP_AFTER_CHECKPOINT" del /q "STOP_AFTER_CHECKPOINT" >nul 2>nul
set /a ATTEMPT=0
:restart
set /a ATTEMPT+=1
echo.
echo ============================================================
echo HanziStyleForge Fusion months run - attempt !ATTEMPT!
echo All major stages and every generated glyph are checkpointed.
echo ============================================================
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-auto-months
set RC=!errorlevel!
if !RC! EQU 0 goto :done
if !RC! EQU 75 goto :safe_stopped
if !RC! EQU 76 goto :style_collapse
if !RC! EQU 130 goto :user_stopped
if !ATTEMPT! GEQ 9999 goto :failed
echo.
echo Process exited with code !RC!. Retrying in 60 seconds...
timeout /t 60 /nobreak >nul
goto :restart
:done
echo.
echo Complete. See build\target-HanziStyleForge-Fusion.ttf
pause
exit /b 0
:safe_stopped
echo.
echo Safely stopped after a durable checkpoint.
echo Run this file again to resume. The completed request will be cleared automatically.
pause
exit /b 75
:style_collapse
echo.
echo Training stopped by the target-style collapse guard.
echo Review DIFFUSION_STYLE_COLLAPSE_DETECTED.json before changing configuration.
pause
exit /b 76
:user_stopped
echo.
echo Stopped by user. Run this file again to resume from the latest checkpoint.
pause
exit /b 130
:not_installed
echo Run install_cuda130.bat first.
pause
exit /b 1
:failed
echo Retry limit reached. Correct the persistent error, then run this file again.
pause
exit /b 1
