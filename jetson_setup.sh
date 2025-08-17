#!/bin/bash
# Jetson Nano Setup and Run Script
# This script helps setup and run the video streaming server on Jetson Nano

echo "==================================="
echo "Video Streaming Server - Jetson Setup"
echo "==================================="

# Check if running on Jetson
if grep -q "tegra" /proc/cpuinfo; then
    echo "✅ Running on NVIDIA Jetson platform"
else
    echo "⚠️  Not detected as Jetson platform, but continuing..."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Check current directory structure
echo ""
echo "Checking directory structure..."
echo "Current directory: $(pwd)"

# Check for required files
files_to_check=(
    "video_server - Final, with AI.py"
    "yolov8n-pose.pt"
    "Gun Detection/GunModel.pt"
    "Grenade Detection/best.pt"
    "templates/index.html"
)

echo ""
echo "Checking required files:"
for file in "${files_to_check[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        size_mb=$((size / 1024 / 1024))
        echo "✅ $file (${size_mb} MB)"
    else
        echo "❌ $file - NOT FOUND"
    fi
done

# Check for Python packages
echo ""
echo "Checking Python packages..."
packages=("opencv-python" "flask" "ultralytics" "numpy")

for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ $package installed"
    else
        echo "❌ $package NOT installed"
        echo "   Install with: pip3 install $package"
    fi
done

echo ""
echo "Setup complete! To run the server:"
echo "python3 'video_server - Final, with AI.py'"
echo ""
echo "Or run this script with 'start' parameter:"
echo "./jetson_setup.sh start"

# If 'start' parameter is provided, run the server
if [[ "$1" == "start" ]]; then
    echo ""
    echo "Starting video streaming server..."
    python3 "video_server - Final, with AI.py"
fi
