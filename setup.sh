#!/bin/bash

echo "============================================="
echo "VIDEO STREAMING SETUP - LINUX/MAC"
echo "============================================="
echo

echo "[1/3] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 tidak ditemukan!"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "MacOS: brew install python3"
    exit 1
fi

echo
echo "[2/3] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Gagal menginstall dependencies!"
    echo "Mencoba dengan pip upgrade..."
    python3 -m pip install --upgrade pip
    pip3 install -r requirements.txt
fi

echo
echo "[3/3] Testing camera..."
python3 camera_test.py

echo
echo "============================================="
echo "SETUP SELESAI!"
echo "============================================="
echo
echo "Untuk menjalankan server streaming:"
echo "  python3 video_server.py"
echo
echo "Untuk test kamera:"
echo "  python3 camera_test.py"
echo
echo "Untuk menjalankan client desktop:"
echo "  python3 video_client.py"
echo

read -p "Tekan Enter untuk melanjutkan..."
