#!/bin/bash
# Install Flask-CORS for better Jetson Nano compatibility

echo "🔧 Installing Flask-CORS untuk kompatibilitas Jetson Nano..."

# Update pip
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install Flask-CORS
echo "🌐 Installing Flask-CORS..."
pip3 install flask-cors

echo "✅ Flask-CORS installed successfully!"
echo "🚀 Sekarang jalankan: python3 pose_server.py"
