# JETSON NANO - TOGGLE BUTTON TROUBLESHOOTING

## Masalah: Toggle button tidak berfungsi di Jetson Nano

### 🔍 Kemungkinan Penyebab:

1. **Browser Compatibility** - Browser lama di Jetson tidak support modern JavaScript
2. **CORS Issues** - Cross-Origin Request Blocking
3. **Network Latency** - Koneksi lambat
4. **JavaScript Execution** - Browser tidak execute JavaScript dengan baik

### ✅ Solusi yang Telah Diterapkan:

1. **CORS Headers Manual**
   ```python
   @app.after_request
   def after_request(response):
       response.headers.add('Access-Control-Allow-Origin', '*')
       response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
       response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
       return response
   ```

2. **XMLHttpRequest Fallback**
   - Modern fetch() dengan fallback ke XMLHttpRequest
   - Kompatibel dengan browser lama

3. **Visual Debugging**
   - Debug info panel di bawah button
   - Visual feedback saat button ditekan
   - Console logging yang detail

4. **Enhanced Server Logging**
   - Log setiap request toggle_mode
   - Detail response data
   - Error handling yang lebih baik

### 🚀 Cara Test di Jetson Nano:

1. **Jalankan Server:**
   ```bash
   python3 pose_server.py
   ```

2. **Buka Browser:**
   ```
   http://localhost:5000
   ```

3. **Check Debug Info:**
   - Lihat panel kuning di bawah button
   - Monitor console browser (F12)
   - Monitor terminal server untuk log

4. **Test Button:**
   - Klik toggle button
   - Perhatikan visual feedback
   - Check debug info berubah
   - Pastikan server log muncul

### 🔧 Troubleshooting Steps:

**Step 1: Check JavaScript Console**
```
F12 → Console → Lihat error messages
```

**Step 2: Check Server Logs**
```
Terminal server harus show:
🔄 Toggle mode request received from [IP]
✅ Mode toggled successfully to: [mode]
📤 Sending response: {...}
```

**Step 3: Manual Test**
```bash
# Test endpoint langsung
curl http://localhost:5000/toggle_mode
```

**Step 4: Install CORS Support**
```bash
chmod +x install_jetson_cors.sh
./install_jetson_cors.sh
```

### 📱 Alternative: Manual Refresh

Jika toggle masih tidak work, user bisa:
1. Refresh halaman untuk mode switch
2. Check video stream untuk pose detection visual
3. Monitor server terminal untuk konfirmasi

### 🎯 Expected Behavior:

✅ **Working Button:**
- Visual feedback saat click
- Debug info update
- Server log muncul
- Mode UI berubah
- Video detection berubah

❌ **Not Working:**
- No visual feedback
- Debug info stuck
- No server log
- UI tidak update
- Video tidak berubah

### 🌟 Tips:

1. **Gunakan Chromium Browser** di Jetson jika available
2. **Clear browser cache** sebelum test
3. **Disable browser security** untuk testing:
   ```bash
   chromium-browser --disable-web-security --disable-features=VizDisplayCompositor
   ```

4. **Check network connectivity**:
   ```bash
   ping localhost
   curl -I http://localhost:5000
   ```
