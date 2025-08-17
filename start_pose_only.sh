#!/bin/bash
echo "=========================================="
echo "🤸 MALEO - POSE DETECTION ONLY LAUNCHER"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 tidak ditemukan!"
    echo "💡 Install Python3 terlebih dahulu"
    echo "   sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 tidak ditemukan!"
    echo "💡 Install pip3 terlebih dahulu"
    echo "   sudo apt install python3-pip"
    exit 1
fi

echo "🔍 Checking dependencies..."

# Check and install OpenCV
python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ OpenCV tidak terinstall"
    echo "📦 Installing OpenCV..."
    pip3 install opencv-python
fi

# Check and install Flask
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Flask tidak terinstall"
    echo "📦 Installing Flask..."
    pip3 install flask
fi

# Check and install MediaPipe
python3 -c "import mediapipe" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ MediaPipe tidak terinstall"
    echo "📦 Installing MediaPipe..."
    pip3 install mediapipe
fi

echo "✅ All dependencies ready!"
echo ""
echo "🚀 Starting Pose Detection Server..."
echo "📍 Server will run on: http://localhost:5001"
echo "🎯 Mode: Pose Detection Only"
echo ""
echo "Press Ctrl+C to stop server"
echo "=========================================="

python3 pose_only_server.py

echo ""
echo "👋 Server stopped"
