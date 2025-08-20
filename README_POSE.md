# 🤸 Pose Detection Web Interface

## 🚀 Quick Start (3 Cara Mudah!)

### 1️⃣ Windows - Double Click (Termudah!)
```
🖱️ Double-click: start_pose_detection.bat
```

### 2️⃣ Python Command
```bash
python pose_server.py
```

### 3️⃣ Manual Run
```bash
cd d:\AI
python pose_server.py
```

## 🌐 Menggunakan Interface

1. **Server sudah berjalan?** Lihat output:
   ```
   ✅ MediaPipe available - Pose detection ready!
   🌐 Server running at: http://localhost:5000
   ```

2. **Buka browser** → http://localhost:5000

3. **Klik tombol** "SWITCH TO POSE MODE"

4. **Berpose di depan kamera!** 🕺💃

## ✨ Yang Akan Anda Lihat

### 📹 Video Stream
- Live video dari webcam Anda
- FPS counter di pojok kanan atas
- Status "POSE DETECTION: ON/OFF"

### 🤸 Pose Detection Active
- Skeleton lines pada tubuh yang terdeteksi
- Text "✅ POSE DETECTED" ketika ada pose
- 33 keypoints tracking

### 🎮 Controls
- Tombol "SWITCH TO POSE MODE" untuk aktifkan
- Tombol "SWITCH TO WEAPON MODE" untuk nonaktifkan
- Status real-time di panel kiri bawah

## 🛠️ Requirements

### Dependencies:
```bash
pip install flask opencv-python mediapipe numpy
```

### System:
- Python 3.7+
- Webcam
- Browser modern

## 📊 Status Indicators

| Status | Meaning |
|--------|---------|
| ✅ POSE DETECTED | Ada pose yang terdeteksi |
| POSE DETECTION: ON | Aktif, tapi belum detect pose |
| POSE DETECTION: OFF | Tidak aktif |

## 🔧 Troubleshooting

### Camera tidak muncul:
- Pastikan webcam terhubung
- Tutup aplikasi lain yang menggunakan camera
- Restart browser

### MediaPipe error:
```bash
pip uninstall mediapipe
pip install mediapipe
```

### Port sudah digunakan:
- Tutup aplikasi lain di port 5000
- Atau edit `port = 5000` di pose_server.py

## 📱 Network Access

Server dapat diakses dari perangkat lain di jaringan yang sama:
- Cek IP di output server
- Akses: `http://[IP]:5000`
- Contoh: `http://192.168.1.73:5000`

## 🎯 Features

✅ **Real-time pose detection**  
✅ **Web-based interface**  
✅ **Mobile friendly**  
✅ **Start/Stop controls**  
✅ **FPS monitoring**  
✅ **Error handling**  
✅ **Network accessible**  

---

## 🎉 Ready to Use!

**File penting:**
- `pose_server.py` - Server utama
- `start_pose_detection.bat` - Windows launcher
- `templates/index.html` - Web interface

**Langkah mudah:**
1. Double-click `start_pose_detection.bat`
2. Buka http://localhost:5000
3. Klik "SWITCH TO POSE MODE"
4. Mulai berpose! 🤸‍♂️
