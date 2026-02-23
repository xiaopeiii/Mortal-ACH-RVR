@echo off
setlocal
cd /d "%~dp0"
python mortal.py %*
exit /b %ERRORLEVEL%
