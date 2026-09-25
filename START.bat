@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title License Plate Recognition System

set PYTHONIOENCODING=utf-8

echo.
echo ====================================================================
echo    Thai License Plate Recognition System (SIFT + Homography + OCR)
echo ====================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    echo Please install Python 3.10+ and make sure to check "Add Python to PATH".
    pause
    exit /b 1
)

echo [OK] Python found.
echo.

REM Check / Install dependencies
echo [*] Checking dependencies...
pip install -r "%~dp0backend\requirements.txt" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing dependencies...
    pip install -r "%~dp0backend\requirements.txt"
)
echo [OK] Dependencies ready.
echo.

REM Start backend server
echo [*] Starting Backend Server...
echo ====================================================================
echo.
echo Access the Web UI:   http://localhost:8000
echo API Documentation:   http://localhost:8000/docs
echo.
echo ====================================================================
echo.

cd /d "%~dp0"
start "License Plate Backend" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

ping -n 3 127.0.0.1 >nul

echo Opening browser...
start http://localhost:8000

echo.
echo Server is running in the backend window.
echo Press any key to exit this launcher window...
pause >nul
