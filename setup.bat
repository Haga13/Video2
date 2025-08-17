@echo off
echo =============================================
echo VIDEO STREAMING SETUP - WINDOWS
echo =============================================
echo.

echo [1/3] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python tidak ditemukan!
    echo Silakan install Python dari https://python.org
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Gagal menginstall dependencies!
    echo Mencoba dengan pip upgrade...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
)

echo.
echo [3/3] Testing camera...
python camera_test.py

echo.
echo =============================================
echo SETUP SELESAI!
echo =============================================
echo.
echo Untuk menjalankan server streaming:
echo   python video_server.py
echo.
echo Untuk test kamera:
echo   python camera_test.py
echo.
echo Untuk menjalankan client desktop:
echo   python video_client.py
echo.
pause
