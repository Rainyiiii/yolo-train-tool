@echo off
setlocal
cd /d "%~dp0"

set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "LOG_DIR=%~dp0logs"
set "LOG_FILE=%LOG_DIR%\stop_launcher.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%PS_EXE%" (
  echo Windows PowerShell was not found:
  echo %PS_EXE%
  pause
  exit /b 1
)

"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_train_panel.ps1" > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo MyAutoTrain failed to stop.
  echo Log: %LOG_FILE%
  echo.
  type "%LOG_FILE%"
  echo.
  pause
  exit /b %EXIT_CODE%
)

echo MyAutoTrain has stopped.
exit /b 0
