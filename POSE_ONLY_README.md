# MALEO - POSE DETECTION ONLY

## 🎯 Overview
Server khusus untuk pose detection saja, tanpa fitur gun detection atau grenade detection. Lebih ringan dan fokus hanya pada deteksi gerakan tubuh manusia.

## ✨ Fitur:
- ✅ **Real-time Pose Detection** menggunakan MediaPipe
- ✅ **33 Pose Landmarks** tracking
- ✅ **Multi-person Detection** support
- ✅ **FPS Monitoring** real-time
- ✅ **Web Interface** responsive
- ✅ **Sensor Data Simulation**
- ✅ **Auto Camera Detection**
- ❌ **No Gun/Grenade Detection** (removed)
- ❌ **No Mode Toggle** (always pose mode)

## 🚀 Quick Start

### Metode 1: Launcher (Recommended)
```bash
# Windows
start_pose_only.bat

# Linux/Mac
chmod +x start_pose_only.sh
./start_pose_only.sh
```

### Metode 2: Manual
```bash
# Install dependencies
pip install flask opencv-python mediapipe

# Run server
python pose_only_server.py
```

## 🌐 Access
- **Local**: http://localhost:5001
- **Network**: http://[YOUR_IP]:5001

## 📱 Interface Features

### 🎥 Video Display
- Fullscreen video stream
- Real-time pose landmarks overlay
- FPS counter display
- Pose detection status

### 📊 Information Panels
- **MALEO Branding** - Top center
- **Sensor Data** - Bottom left (temperature, humidity)
- **Pose Status** - Top right (status, FPS, mode)

### 🎨 Visual Elements
- Green pose landmarks on detected persons
- "🤸 POSE DETECTED" text when person found
- Live indicator dot (green)
- Professional dark theme

## 🔧 Technical Specifications

### 📹 Camera Settings
- Resolution: 640x480
- FPS: 30
- Auto-detection: Camera index 0

### 🤖 MediaPipe Configuration
- Model complexity: 1 (balanced)
- Min detection confidence: 0.5
- Min tracking confidence: 0.5
- Static image mode: False

### 🌐 Server Configuration
- Port: 5001 (to avoid conflicts)
- Host: 0.0.0.0 (accessible from network)
- CORS: Enabled for compatibility

## 📂 File Structure
```
d:\AI\
├── pose_only_server.py       # Main server file
├── start_pose_only.bat       # Windows launcher
├── templates/
│   └── pose_only.html        # Web interface
└── POSE_ONLY_README.md       # This documentation
```

## 🛠️ Dependencies
```
flask>=2.0.0
opencv-python>=4.5.0
mediapipe>=0.10.0
```

## 💡 Usage Tips

### 🎯 Best Performance
1. **Stand 1-2 meters** from camera
2. **Good lighting** recommended
3. **Avoid busy backgrounds**
4. **Face the camera** directly

### 🔍 Troubleshooting
- **No pose detected**: Check lighting and distance
- **Low FPS**: Close other applications
- **Camera not found**: Check camera permissions
- **MediaPipe errors**: Restart server

## 🆚 Differences from Full Server

| Feature | Full Server | Pose Only |
|---------|-------------|-----------|
| Port | 5000 | 5001 |
| Gun Detection | ✅ | ❌ |
| Grenade Detection | ✅ | ❌ |
| Mode Toggle | ✅ | ❌ |
| Pose Detection | ✅ | ✅ |
| File Size | ~35KB | ~15KB |
| Memory Usage | Higher | Lower |
| CPU Usage | Higher | Lower |

## 🎪 Demo Commands

### Start Server
```bash
python pose_only_server.py
```

### Test Endpoints
```bash
# Get status
curl http://localhost:5001/status

# Get sensor data
curl http://localhost:5001/get_sensor_data
```

### Check Performance
```bash
# Monitor with htop (Linux)
htop -p $(pgrep -f pose_only_server)

# Task Manager (Windows)
# Look for python.exe process
```

## ⚡ Performance Optimizations
- Single model loading (pose only)
- Efficient frame processing
- Queue-based frame management
- Threaded camera capture
- Optimized MediaPipe settings

## 🔮 Future Enhancements
- [ ] Multiple camera support
- [ ] Pose gesture recognition
- [ ] Pose statistics logging
- [ ] Custom pose alerts
- [ ] Export pose data
- [ ] Mobile-optimized interface

## 🤝 Support
Jika ada masalah:
1. Check console output untuk error messages
2. Verify camera dan dependencies
3. Test dengan `curl` commands
4. Check browser console (F12)

## 📄 License
Same as main project - for educational and development purposes.

---
**MALEO Pose Detection Only - Focused, Fast, Reliable** 🎯
