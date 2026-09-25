#!/usr/bin/env python3
"""
License Plate Recognition System - One-Click Startup
Starts backend and opens web UI automatically
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def print_header():
    """Print beautiful header"""
    print("\n")
    print("╔" + "═" * 66 + "╗")
    print("║" + " " * 66 + "║")
    print("║" + "  🚗 Thai License Plate Recognition System".center(66) + "║")
    print("║" + "  SIFT + Homography + OCR".center(66) + "║")
    print("║" + " " * 66 + "║")
    print("╚" + "═" * 66 + "╝")
    print()

def check_python():
    """Check Python version"""
    print("✓ Python version:", sys.version.split()[0])
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ required!")
        input("Press Enter to exit...")
        sys.exit(1)

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing/Checking dependencies...")

    req_file = Path(__file__).parent / "backend" / "requirements.txt"

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError:
        print("⚠️  Installing dependencies (this may take 1-2 minutes)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])

def start_backend():
    """Start the FastAPI backend"""
    print("\n🚀 Starting Backend Server...\n")
    print("━" * 68)
    print()

    # Start uvicorn in background
    os.chdir(Path(__file__).parent)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]

    try:
        process = subprocess.Popen(cmd)
        time.sleep(3)  # Give server time to start
        return process
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

def open_browser():
    """Open the web UI in default browser"""
    print("\n" + "━" * 68)
    print()
    print("✅ Backend started successfully!\n")
    print("📍 Access the Web UI:")
    print("   🌐 http://localhost:8000\n")
    print("📚 API Documentation:")
    print("   🔗 http://localhost:8000/docs\n")
    print("━" * 68 + "\n")

    time.sleep(1)
    print("🌐 Opening browser in 2 seconds...\n")
    time.sleep(2)

    try:
        webbrowser.open("http://localhost:8000")
        print("✓ Browser opened!")
    except Exception as e:
        print(f"⚠️  Could not open browser: {e}")
        print("   Please manually visit: http://localhost:8000")

def print_usage():
    """Print usage tips"""
    print()
    print("💡 Quick Start Guide:")
    print("   1. Upload a car image with Thai license plate")
    print("   2. Watch the pipeline visualization")
    print("   3. See recognition results instantly\n")
    print("📝 Try these:")
    print("   • Clear, front-facing plates work best")
    print("   • Good lighting improves accuracy")
    print("   • Use 'Edit Manually' if OCR needs correction\n")
    print("ℹ️  To stop:")
    print("   • Press Ctrl+C in the backend window")
    print("   • Or close the backend window\n")

def main():
    """Main startup routine"""
    print_header()

    # Check Python
    check_python()

    # Install dependencies
    install_dependencies()

    # Start backend
    process = start_backend()

    # Open browser
    open_browser()

    # Print usage
    print_usage()

    print("━" * 68)
    print("\n⏳ Server is running. Press Ctrl+C to stop.\n")

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n\n⛔ Stopping server...")
        process.terminate()
        process.wait()
        print("✓ Server stopped\n")

if __name__ == "__main__":
    main()
