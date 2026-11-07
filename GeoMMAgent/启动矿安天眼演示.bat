@echo off
title KAT Demo Launcher
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1"
if errorlevel 1 pause
