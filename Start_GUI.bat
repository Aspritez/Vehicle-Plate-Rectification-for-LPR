@echo off
title CCTV License Plate AI Dashboard
echo Starting CCTV License Plate Dashboard...
python gui_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error.
    pause
)
