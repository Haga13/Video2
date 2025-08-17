# Model Paths Configuration

## Current Model Structure

Aplikasi ini menggunakan path relatif sehingga dapat dipindahkan ke sistem lain (termasuk Jetson Nano) tanpa perlu mengubah konfigurasi path.

### Struktur Folder yang Diperlukan:
```
project_root/
├── video_server - Final, with AI.py
├── yolov8n-pose.pt                 # YOLOv8 Pose detection model (6.5 MB)
├── Gun Detection/
│   └── GunModel.pt                 # Gun detection model (21.5 MB)
├── Grenade Detection/
│   └── best.pt                     # Grenade detection model
├── templates/
│   └── index.html
├── requirements.txt
├── jetson_setup.sh                 # Setup script for Jetson Nano
└── MODEL_PATHS.md                  # This file
```

### Path Configuration:
- **Gun Model**: `./Gun Detection/GunModel.pt` (relative path)
- **Grenade Model**: `./Grenade Detection/best.pt` (relative path)
- **Pose Model**: `./yolov8n-pose.pt` (akan didownload otomatis jika tidak ada)

### Deployment ke Jetson Nano:

#### 1. Transfer Files:
```bash
# Copy seluruh folder ke Jetson Nano
scp -r /path/to/project/ jetson@jetson-ip:/home/jetson/video_streaming/
```

#### 2. Setup Environment:
```bash
cd /home/jetson/video_streaming/
chmod +x jetson_setup.sh
./jetson_setup.sh
```

#### 3. Install Dependencies:
```bash
pip3 install -r requirements.txt
```

#### 4. Run Application:
```bash
python3 "video_server - Final, with AI.py"
# atau
./jetson_setup.sh start
```

### Features:
- **Cross-platform compatibility**: Windows, Linux, Jetson Nano
- **Automatic model detection**: Checks for local models first
- **Fallback mechanism**: Downloads YOLOv8 pose model if not found locally
- **Debug information**: Detailed logging for troubleshooting
- **Performance optimization**: Mode toggle to reduce computational load

### Mode Toggle Feature:
- **Weapon Mode**: Deteksi senjata dan granat (gun + grenade detection)
- **Pose Mode**: Deteksi pose manusia
- Toggle via web interface atau API endpoint `/toggle_mode`
- Mengurangi beban komputasi dengan hanya menjalankan satu model group

### API Endpoints:
- `GET /status` - Status server dan AI models
- `GET /toggle_mode` - Switch antara weapon/pose detection
- `GET /toggle_detection` - Enable/disable AI detection
- `POST /ai_settings` - Update confidence thresholds

### Troubleshooting:
1. **Model tidak ditemukan**: Check struktur folder dan permissions
2. **Performance issues di Jetson**: Gunakan mode toggle, jalankan hanya satu model
3. **Camera issues**: Pastikan camera device tersedia dan tidak digunakan aplikasi lain
4. **Memory issues**: Reduce input resolution atau model size

### System Requirements:
- **Minimum**: 4GB RAM, CUDA-compatible GPU (optional)
- **Jetson Nano**: 4GB model recommended
- **Camera**: USB camera atau CSI camera
- **Network**: WiFi atau Ethernet untuk web interface
