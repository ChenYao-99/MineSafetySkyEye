@echo off
REM Silent backend restart (for scripts / automation).
REM Same as the desktop launcher, but without the 25s wait and without pause.
REM
REM Usage: double-click, or run  start_backend_5100.bat
REM
REM IMPORTANT: launch via "start" so the process gets its own console.
REM Running it in the background of an existing shell (e.g. nohup ... &)
REM lets its std handles die with that shell, which on Windows makes
REM werkzeug hit a socket-inheritance fault (WinError 10038).
REM
REM NOTE: keep this file ASCII-only. cmd.exe reads .bat as the OEM codepage
REM (GBK on this machine), so UTF-8 Chinese comments become mojibake and get
REM executed as bogus commands. Same reason the desktop launcher has none.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1" -NoBrowser
