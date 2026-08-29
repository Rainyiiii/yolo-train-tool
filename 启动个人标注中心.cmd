@echo off
setlocal
cd /d "%~dp0"
wscript.exe "%~dp0启动个人标注中心.vbs"
exit /b %ERRORLEVEL%
