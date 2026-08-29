@echo off
setlocal
cd /d "%~dp0"
wscript.exe "%~dp0关闭协作标注中心.vbs"
exit /b %ERRORLEVEL%
