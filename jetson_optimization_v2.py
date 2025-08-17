# JETSON NANO PERFORMANCE OPTIMIZATION
# Konfigurasi khusus untuk balance antara layar 7 inch dan performa Jetson

# Jetson-specific resolution recommendations
JETSON_DISPLAY_CONFIGS = {
    "jetson_7inch_performance": {
        "name": "Jetson 7-Inch Performance Mode",
        "camera_width": 512,      # Compromise: bigger than 480, smaller than 640
        "camera_height": 384,     # 4:3 aspect ratio maintained
        "jpeg_quality": 65,       # Lower quality for performance
        "frame_skip": 3,          # Process every 3rd frame
        "ai_input_size": 320,     # Reduced AI input (vs default 416)
        "estimated_fps": "6-12 FPS",
        "recommended": True
    },
    
    "jetson_7inch_balanced": {
        "name": "Jetson 7-Inch Balanced Mode", 
        "camera_width": 640,      # Full 7-inch resolution
        "camera_height": 480,
        "jpeg_quality": 60,       # Lower quality
        "frame_skip": 4,          # Process every 4th frame  
        "ai_input_size": 320,     # Reduced AI input
        "estimated_fps": "4-8 FPS",
        "recommended": False
    },
    
    "jetson_7inch_quality": {
        "name": "Jetson 7-Inch Quality Mode",
        "camera_width": 640,
        "camera_height": 480, 
        "jpeg_quality": 70,
        "frame_skip": 2,          # Process every 2nd frame
        "ai_input_size": 416,     # Standard AI input
        "estimated_fps": "3-6 FPS", 
        "recommended": False
    }
}

# Thermal management settings
THERMAL_CONFIG = {
    "temperature_check_interval": 5,   # Check temp every 5 seconds
    "thermal_limit": 65,               # Start throttling at 65°C  
    "emergency_limit": 75,             # Emergency shutdown at 75°C
    "cooling_actions": [
        "reduce_fps",                  # First: reduce FPS
        "increase_frame_skip",         # Second: skip more frames
        "reduce_quality",              # Third: reduce JPEG quality
        "disable_ai"                   # Last resort: disable AI
    ]
}

# Memory optimization
MEMORY_CONFIG = {
    "max_buffer_frames": 1,        # Minimal buffering
    "gc_interval": 30,             # Garbage collect every 30 seconds
    "memory_limit_mb": 512,        # Max memory for video processing
    "swap_usage_limit": 25         # Max 25% swap usage
}

# Network optimization for Jetson  
NETWORK_CONFIG = {
    "stream_buffer_size": 1024,    # Smaller network buffer
    "compression_level": 9,        # Max compression
    "adaptive_bitrate": True,      # Adjust quality based on network
    "max_clients": 2,              # Limit concurrent clients
    "timeout_seconds": 30          # Connection timeout
}

def get_jetson_optimal_config():
    """Get the most optimal configuration for Jetson Nano with 7-inch display"""
    return {
        "display": JETSON_DISPLAY_CONFIGS["jetson_7inch_performance"],
        "thermal": THERMAL_CONFIG,
        "memory": MEMORY_CONFIG, 
        "network": NETWORK_CONFIG,
        "additional_optimizations": {
            "disable_debug_logging": True,
            "use_gpu_acceleration": True,
            "enable_nvpmodel_max": True,
            "jetson_clocks": True
        }
    }

# Performance monitoring
MONITORING_CONFIG = {
    "monitor_cpu_usage": True,
    "monitor_gpu_usage": True, 
    "monitor_temperature": True,
    "monitor_memory": True,
    "monitor_fps": True,
    "log_performance_data": True,
    "performance_log_interval": 60  # Log every minute
}

if __name__ == "__main__":
    config = get_jetson_optimal_config()
    print("🚀 JETSON NANO OPTIMAL CONFIGURATION FOR 7-INCH DISPLAY")
    print("="*60)
    print(f"📺 Resolution: {config['display']['camera_width']}x{config['display']['camera_height']}")
    print(f"📊 Quality: {config['display']['jpeg_quality']}%")  
    print(f"⏭️  Frame Skip: Every {config['display']['frame_skip']} frames")
    print(f"🤖 AI Input: {config['display']['ai_input_size']}px")
    print(f"🎯 Expected FPS: {config['display']['estimated_fps']}")
    print(f"🌡️  Thermal Limit: {config['thermal']['thermal_limit']}°C")
    print(f"💾 Memory Limit: {config['memory']['memory_limit_mb']}MB")
