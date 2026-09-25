#!/bin/bash

echo "License Plate Recognition System - Backend Server"
echo ""
echo "Installing dependencies (if needed)..."
pip install -q -r backend/requirements.txt

echo ""
echo "Starting FastAPI server..."
echo ""
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
