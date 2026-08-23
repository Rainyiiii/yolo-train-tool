@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo MyAutoTrain Python environment was not found. Run 一键安装并启动.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" annotation_service.py start --share
if errorlevel 1 pause
