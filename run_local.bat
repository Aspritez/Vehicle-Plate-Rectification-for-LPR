@echo off
:: Thin CMD wrapper around run_local.ps1.
:: Double-click this file in Windows Explorer to start the app.
powershell -ExecutionPolicy Bypass -File "%~dp0run_local.ps1"
pause
