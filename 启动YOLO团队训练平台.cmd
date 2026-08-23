@echo off
setlocal
cd /d "%~dp0"
if exist "Desktop\YOLOTeamTrainingPlatform.exe" (
  start "" "Desktop\YOLOTeamTrainingPlatform.exe"
  exit /b 0
)
if not exist ".venv\Scripts\python.exe" (
  echo 尚未完成安装，正在启动一键安装程序...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_and_start.ps1"
  exit /b %errorlevel%
)
".venv\Scripts\python.exe" panel_service.py start
if errorlevel 1 pause
