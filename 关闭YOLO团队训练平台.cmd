@echo off
setlocal
cd /d "%~dp0"
wscript.exe "%~dp0关闭YOLO团队训练平台.vbs"
exit /b %ERRORLEVEL%
