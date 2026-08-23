@echo off
setlocal
cd /d "%~dp0"
title MyAutoTrain Installer

echo MyAutoTrain is preparing this computer...
echo This may take several minutes on first use.
echo.

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_and_start.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Installation did not finish. See logs\install.log
  pause
)
exit /b %EXIT_CODE%
