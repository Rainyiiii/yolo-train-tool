@echo off
setlocal
cd /d "%~dp0"

set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "LOG_DIR=%~dp0workspace\logs"
set "LOG_FILE=%LOG_DIR%\launcher.log"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%PS_EXE%" (
  echo Windows PowerShell was not found:
  echo %PS_EXE%
  pause
  exit /b 1
)

if not exist "%PROJECT_PYTHON%" (
  echo First use detected. Starting the one-click installer...
  "%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_and_start.ps1"
  exit /b %ERRORLEVEL%
)

"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_panel.ps1" > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo YOLO Team Training Platform failed to start.
  echo Log: %LOG_FILE%
  echo.
  type "%LOG_FILE%"
  echo.
  pause
  exit /b %EXIT_CODE%
)

echo YOLO Team Training Platform is running at http://127.0.0.1:8989/
exit /b 0
