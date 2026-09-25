@echo off
setlocal
chcp 65001 >nul
title License Plate Backend Server

set PYTHONIOENCODING=utf-8

echo License Plate Recognition System - Backend Server
echo.
echo Installing dependencies (if needed)...
pip install -r "%~dp0backend\requirements.txt"

echo.
echo Starting FastAPI server...
echo.
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

pause
