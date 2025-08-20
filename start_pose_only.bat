@echo off
echo ==========================================
echo 🤸 MALEO - POSE DETECTION ONLY LAUNCHER
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python tidak ditemukan!
    echo 💡 Install Python terlebih dahulu
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 🔍 Checking dependencies...

python -c "import cv2" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ OpenCV tidak terinstall
    echo 📦 Installing OpenCV...
    pip install opencv-python
)

python -c "import flask" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Flask tidak terinstall
    echo 📦 Installing Flask...
    pip install flask
)

python -c "import mediapipe" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ MediaPipe tidak terinstall
    echo 📦 Installing MediaPipe...
    pip install mediapipe
)

echo ✅ All dependencies ready!
echo.
echo 🚀 Starting Pose Detection Server...
echo 📍 Server will run on: http://localhost:5001
echo 🎯 Mode: Pose Detection Only
echo.
echo Press Ctrl+C to stop server
echo ==========================================

python pose_only_server.py

echo.
echo 👋 Server stopped
pause
