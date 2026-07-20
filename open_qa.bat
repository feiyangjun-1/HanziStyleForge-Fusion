@echo off
cd /d "%~dp0"
if exist "work_hanzistyleforge_fusion_months\qa\index.html" start "" "work_hanzistyleforge_fusion_months\qa\index.html"
if not exist "work_hanzistyleforge_fusion_months\qa\index.html" echo QA report is not available yet.
pause
