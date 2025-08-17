# Configuration for 7-inch display optimization
# Konfigurasi khusus untuk layar 7 inch

# Common 7-inch display resolutions
DISPLAY_CONFIGS = {
    "7inch_800x480": {
        "name": "7 Inch - 800x480 (Standard)",
        "camera_width": 640,
        "camera_height": 480,
        "display_width": 800,
        "display_height": 480,
        "aspect_ratio": "4:3",
        "recommended": True
    },
    "7inch_1024x600": {
        "name": "7 Inch - 1024x600 (Widescreen)",
        "camera_width": 640,
        "camera_height": 480,
        "display_width": 1024,
        "display_height": 600,
        "aspect_ratio": "16:10",
        "recommended": False
    },
    "7inch_480x320": {
        "name": "7 Inch - 480x320 (Low Resolution)",
        "camera_width": 480,
        "camera_height": 320,
        "display_width": 480,
        "display_height": 320,
        "aspect_ratio": "3:2",
        "recommended": False
    }
}

# CSS styling adjustments for 7-inch displays
UI_SCALING = {
    "overlay_title_size": 28,      # Main title font size
    "overlay_subtitle_size": 16,   # Subtitle font size
    "date_font_size": 14,          # Date display font size
    "panel_font_size": 10,         # Control panel font size
    "panel_padding": 6,            # Panel padding
    "panel_margin": 10,            # Panel margin from edges
    "button_padding": 6            # Button padding
}

# Performance settings optimized for 7-inch displays
PERFORMANCE_CONFIG = {
    "frame_skip": 2,               # Process every 2nd frame
    "jpeg_quality": 70,            # JPEG compression quality (70% for balance)
    "buffer_size": 1,              # Camera buffer size (minimal latency)
    "fps_target": 20,              # Target FPS for smooth playback on small screen
}

# Touch interface settings (if using touchscreen)
TOUCH_CONFIG = {
    "button_min_size": 44,         # Minimum touch target size (44px standard)
    "touch_friendly_spacing": 8,   # Spacing between touch elements
    "swipe_enabled": True,         # Enable swipe gestures for mode switching
    "long_press_duration": 800     # Long press duration in ms
}

def get_optimal_config(display_type="7inch_800x480"):
    """
    Get optimal configuration for specified 7-inch display
    
    Args:
        display_type: Type of 7-inch display configuration
    
    Returns:
        dict: Complete configuration dictionary
    """
    if display_type not in DISPLAY_CONFIGS:
        display_type = "7inch_800x480"  # Default fallback
    
    config = {
        "display": DISPLAY_CONFIGS[display_type],
        "ui_scaling": UI_SCALING,
        "performance": PERFORMANCE_CONFIG,
        "touch": TOUCH_CONFIG
    }
    
    return config

def apply_config_to_server(app, config):
    """
    Apply 7-inch display configuration to Flask server
    
    Args:
        app: Flask application instance
        config: Configuration dictionary from get_optimal_config()
    """
    # This would be called from the main server file
    print(f"Applying 7-inch display configuration: {config['display']['name']}")
    print(f"Camera resolution: {config['display']['camera_width']}x{config['display']['camera_height']}")
    print(f"Display resolution: {config['display']['display_width']}x{config['display']['display_height']}")
    print(f"Performance: {config['performance']['fps_target']} FPS target, Quality: {config['performance']['jpeg_quality']}%")

if __name__ == "__main__":
    # Test configuration
    config = get_optimal_config("7inch_800x480")
    print("7-Inch Display Configuration:")
    print("="*40)
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
