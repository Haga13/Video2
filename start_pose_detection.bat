@echo off
cls
echo ============================================
echo        🤸 POSE DETECTION LAUNCHER
echo ============================================
echo.
echo Starting pose detection web server...
echo Make sure your webcam is connected!
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found!
    echo    Please install Python and add it to PATH
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check dependencies
echo Checking dependencies...
python -c "import cv2, flask, mediapipe" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Some packages are missing. Installing...
    echo.
    pip install flask opencv-python mediapipe numpy
    if errorlevel 1 (
        echo.
        echo ❌ Failed to install dependencies!
        echo    Please install manually: pip install flask opencv-python mediapipe numpy
        pause
        exit /b 1
    )
    echo.
    echo ✅ Dependencies installed successfully!
)

echo ✅ All dependencies available
echo.
echo 🚀 Starting server...
echo ============================================
echo.

REM Start the server
python pose_server.py

echo.
echo 👋 Server stopped.
pause
