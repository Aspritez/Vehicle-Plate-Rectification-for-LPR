#!/bin/bash

# Thai License Plate Recognition System - One-Click Startup
# Works on Linux and Mac

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     🚗 Thai License Plate Recognition System                  ║"
    echo "║     SIFT + Homography + OCR                                   ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Check Python
check_python() {
    echo -e "${BLUE}Checking Python installation...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Error: Python 3 not found!${NC}"
        echo "   Please install Python 3.10+ first"
        echo "   macOS: brew install python"
        echo "   Linux: sudo apt-get install python3 python3-pip"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
}

# Install dependencies
install_deps() {
    echo -e "\n${BLUE}📦 Installing/Checking dependencies...${NC}"

    if ! python3 -m pip install -q -r backend/requirements.txt 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Installing dependencies (this may take 1-2 minutes)...${NC}"
        python3 -m pip install -r backend/requirements.txt
    fi

    echo -e "${GREEN}✓ Dependencies ready${NC}"
}

# Start backend
start_backend() {
    echo -e "\n${BLUE}🚀 Starting Backend Server...${NC}\n"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Start server in background
    cd "$(dirname "$0")"
    python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!

    # Wait for server to start
    sleep 3

    return $BACKEND_PID
}

# Open browser
open_browser() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${GREEN}✅ Backend started successfully!${NC}\n"
    echo "📍 Access the Web UI:"
    echo -e "   ${BLUE}🌐 http://localhost:8000${NC}\n"
    echo "📚 API Documentation:"
    echo -e "   ${BLUE}🔗 http://localhost:8000/docs${NC}\n"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    sleep 1
    echo "🌐 Opening browser in 2 seconds..."
    sleep 2

    # Try to open browser
    if command -v open &> /dev/null; then
        # macOS
        open http://localhost:8000
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open http://localhost:8000 &
    else
        echo -e "${YELLOW}⚠️  Could not open browser automatically${NC}"
        echo "   Please manually visit: http://localhost:8000"
    fi
}

# Print usage tips
print_tips() {
    echo ""
    echo "💡 Quick Start Guide:"
    echo "   1. Upload a car image with Thai license plate"
    echo "   2. Watch the pipeline visualization"
    echo "   3. See recognition results instantly"
    echo ""
    echo "📝 Try these:"
    echo "   • Clear, front-facing plates work best"
    echo "   • Good lighting improves accuracy"
    echo "   • Use 'Edit Manually' if OCR needs correction"
    echo ""
    echo "ℹ️  To stop the server:"
    echo "   Press Ctrl+C below"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${GREEN}⏳ Server is running. Press Ctrl+C to stop.${NC}\n"
}

# Main
main() {
    print_header
    check_python
    install_deps
    start_backend
    BACKEND_PID=$!
    open_browser
    print_tips

    # Wait for backend to finish or Ctrl+C
    wait $BACKEND_PID 2>/dev/null

    echo -e "\n${YELLOW}⛔ Server stopped${NC}\n"
}

# Make sure script is executable
if [ "$0" != "bash" ] && [ "$0" != "sh" ]; then
    chmod +x "$0"
fi

# Run main
main
