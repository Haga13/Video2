# Jetson Nano Performance Configuration
# Konfigurasi khusus untuk mengurangi lag di Jetson Nano

## Recommended Settings:

### 1. Performance Mode
```bash
curl http://localhost:5000/performance_mode/fast
```

### 2. Video Settings (Lower quality for better FPS)
```bash
curl http://localhost:5000/settings/15/60  # 15 FPS, 60% quality
```

### 3. Environment Variables (Add to ~/.bashrc)
```bash
# CUDA Optimization
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648

# Memory optimization  
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp"

# Limit CPU usage
export OMP_NUM_THREADS=2
```

### 4. System Optimization Commands:

```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Disable GUI for headless operation (more resources)
sudo systemctl set-default multi-user.target

# Increase swap space (if needed)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 5. Power Management:
```bash
# Check current power mode
sudo nvpmodel -q

# Set to max performance (10W mode)
sudo nvpmodel -m 0

# If overheating, use 5W mode
sudo nvpmodel -m 1
```

### 6. Monitoring Commands:
```bash
# Monitor GPU usage
sudo tegrastats

# Check temperature
cat /sys/class/thermal/thermal_zone*/temp

# Monitor memory
free -h
```

## Expected Results on Jetson Nano:

✅ **POSE MODE**: 8-12 FPS (Good performance)
⚠️ **WEAPON MODE**: 4-8 FPS (Acceptable for monitoring)
❌ **BOTH MODES**: 2-4 FPS (Not recommended)

## Deployment Tips:

1. **Start with POSE mode** - less computational load
2. **Use Fast performance mode** - `/performance_mode/fast`
3. **Monitor temperature** - use cooling fan if needed
4. **Lower video quality** if needed for better FPS
5. **Consider USB 3.0 camera** for better performance than CSI
