@echo off
setlocal
cd /d "%~dp0"
wscript.exe "%~dp0启动YOLO团队训练平台.vbs"
exit /b %ERRORLEVEL%
