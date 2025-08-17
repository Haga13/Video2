# Video Streaming dengan Latency Rendah

Sistem video streaming real-time menggunakan Python dengan latency rendah yang dapat diakses melalui jaringan lokal.

## 📋 Fitur

- ✅ **Latency Rendah**: Buffer minimal untuk delay yang minimal
- 🌐 **Akses Jaringan**: Dapat diakses dari perangkat lain di jaringan yang sama
- ⚙️ **Pengaturan Fleksibel**: FPS dan kualitas video dapat disesuaikan
- 📱 **Multi-Platform**: Bekerja di Windows, Mac, dan Linux
- 🎮 **Interface Web**: GUI yang user-friendly melalui browser
- 📸 **Screenshot**: Fitur untuk mengambil screenshot
- 🔄 **Real-time Control**: Kontrol streaming secara real-time

## 🛠️ Instalasi

### 1. Install Dependencies

```bash
pip install flask opencv-python pillow requests numpy
```

### 2. Download Files

Download semua file dalam folder yang sama:
- `video_server.py` - Server streaming
- `video_client.py` - Client desktop (opsional)
- `camera_test.py` - Tool untuk test kamera
- `templates/index.html` - Web interface

## 🚀 Cara Penggunaan

### 1. Test Kamera Terlebih Dahulu

```bash
python camera_test.py
```

Program ini akan:
- Scan semua kamera yang tersedia
- Test apakah kamera dapat berfungsi
- Memberikan preview real-time

### 2. Jalankan Server Streaming

```bash
python video_server.py
```

Server akan menampilkan informasi seperti:
```
==========================================
VIDEO STREAMING SERVER
==========================================
Server berjalan di:
  Local:    http://localhost:5000
  Network:  http://192.168.1.100:5000

Untuk mengakses dari perangkat lain di jaringan yang sama,
gunakan: http://192.168.1.100:5000
==========================================
```

### 3. Akses Streaming

#### Melalui Browser (Recommended)
- **Lokal**: Buka `http://localhost:5000`
- **Jaringan**: Buka `http://[IP_ADDRESS]:5000` dari perangkat lain

#### Melalui Desktop Client (Opsional)
```bash
python video_client.py
```

## 🎮 Kontrol dan Pengaturan

### Web Interface
- **Frame Rate**: 15, 24, 30, 60 FPS
- **Kualitas Video**: 60%, 80%, 90%, 95%
- **Fullscreen**: Mode layar penuh
- **Refresh**: Restart streaming
- **Real-time Status**: Monitor FPS dan kualitas

### Keyboard Shortcuts
- `Ctrl + R`: Refresh stream
- `Ctrl + F`: Toggle fullscreen

## 📡 Akses dari Jaringan Lain

1. **Pastikan dalam jaringan yang sama**
   - WiFi yang sama, atau
   - LAN yang sama

2. **Dapatkan IP Address server**
   - Windows: `ipconfig`
   - Mac/Linux: `ifconfig`
   - Atau lihat output dari `video_server.py`

3. **Akses dari perangkat lain**
   - Smartphone: Buka browser, masuk ke `http://[IP]:5000`
   - Laptop lain: Sama seperti di atas
   - Tablet: Sama seperti di atas

## ⚡ Optimasi Latency

### Pengaturan Server
- Buffer size minimal (1 frame)
- Codec MJPEG untuk kecepatan
- Threading untuk parallel processing
- Queue management untuk frame terbaru

### Pengaturan Jaringan
- Gunakan koneksi WiFi 5GHz
- Pastikan bandwidth cukup
- Tutup aplikasi yang menggunakan bandwidth tinggi
- Gunakan kabel Ethernet jika memungkinkan

### Pengaturan Video
- **Untuk koneksi cepat**: 30-60 FPS, 90-95% quality
- **Untuk koneksi sedang**: 24-30 FPS, 80% quality  
- **Untuk koneksi lambat**: 15 FPS, 60% quality

## 🔧 Troubleshooting

### Kamera Tidak Terdeteksi
```bash
# Test kamera
python camera_test.py

# Pastikan tidak ada aplikasi lain yang menggunakan kamera
# Restart komputer jika perlu
```

### Tidak Bisa Akses dari Jaringan Lain
1. Check firewall settings
2. Pastikan port 5000 tidak diblokir
3. Verifikasi IP address dengan `ipconfig`

### Latency Tinggi
1. Kurangi FPS (15-24)
2. Turunkan kualitas video (60-80%)
3. Tutup aplikasi lain
4. Gunakan koneksi kabel

### Error Dependencies
```bash
# Install semua dependencies
pip install --upgrade flask opencv-python pillow requests numpy

# Jika masih error, coba dengan conda
conda install flask opencv pillow requests numpy
```

## 📱 Contoh Penggunaan

### Scenario 1: Monitoring Rumah
- Setup: PC/Laptop dengan webcam
- Akses: Smartphone dari ruangan lain
- Pengaturan: 24 FPS, 80% quality

### Scenario 2: Video Conference
- Setup: Laptop sebagai server
- Akses: Multiple devices dalam meeting room
- Pengaturan: 30 FPS, 90% quality

### Scenario 3: Security Camera
- Setup: PC dengan IP camera
- Akses: 24/7 monitoring dari tablet
- Pengaturan: 15 FPS, 60% quality (hemat bandwidth)

## 🛡️ Keamanan

⚠️ **Penting**: Sistem ini untuk jaringan lokal saja!

- Tidak ada autentikasi
- Data tidak dienkripsi
- Jangan expose ke internet public
- Gunakan hanya dalam jaringan terpercaya

## 📈 Performance

### Spesifikasi Minimum
- **CPU**: Dual-core 2.0GHz
- **RAM**: 2GB available
- **Network**: 10 Mbps untuk 30 FPS
- **Camera**: USB 2.0 webcam

### Spesifikasi Recommended
- **CPU**: Quad-core 2.5GHz+
- **RAM**: 4GB+ available  
- **Network**: 100 Mbps ethernet/WiFi 5GHz
- **Camera**: USB 3.0 HD webcam

## 📄 Lisensi

Free to use untuk personal dan educational purposes.

## 🤝 Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan fitur baru!
