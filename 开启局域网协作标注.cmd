@echo off
setlocal
cd /d "%~dp0"
wscript.exe "%~dp0开启局域网协作标注.vbs"
exit /b %ERRORLEVEL%
