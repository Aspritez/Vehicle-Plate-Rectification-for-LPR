@echo off
title CCTV License Plate AI Test Studio
echo Starting CCTV License Plate Test Studio...
python test_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error.
    pause
)
