@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" annotation_service.py stop
  ".venv\Scripts\python.exe" panel_service.py stop
)
