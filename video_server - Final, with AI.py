#!/usr/bin/env python3
"""
Enhanced Video Streaming Server with AI Detection
Menggunakan Flask, OpenCV, dan YOLOv8 untuk streaming video dengan deteksi senjata, granat, dan pose

Compatible with Windows and Linux (Jetson Nano)
Uses relative paths for cross-platform compatibility
"""

import cv2
import threading
import time
from flask import Flask, Response, render_template, jsonify, request
import socket
import queue
import numpy as np
import base64
from ultralytics import YOLO
import os
import math
import platform

# Print system information for debugging
print(f"Running on: {platform.system()} {platform.release()}")
print(f"Python version: {platform.python_version()}")
print(f"Working directory: {os.getcwd()}")

app = Flask(__name__)

def get_model_path(relative_path):
    """
    Get absolute path for model files using relative paths
    This ensures compatibility across different operating systems
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, relative_path)
    return os.path.normpath(model_path)

def check_model_exists(model_path, model_name):
    """Check if model exists and print debug information"""
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"✅ {model_name} found: {model_path} ({file_size:.1f} MB)")
        return True
    else:
        print(f"❌ {model_name} not found: {model_path}")
        # Try to list directory contents for debugging
        dir_path = os.path.dirname(model_path)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.pt', '.onnx', '.engine'))]
            if files:
                print(f"   Available model files in directory: {files}")
            else:
                print(f"   No model files found in: {dir_path}")
        else:
            print(f"   Directory does not exist: {dir_path}")
        return False

class PoseDetector:
    def __init__(self, model_size='n', confidence_threshold=0.5, input_size=416):
        """
        Initialize YOLOv8 Pose detector
        
        Args:
            model_size: 'n' (nano/fastest), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
            confidence_threshold: minimum confidence for pose detection (0.0-1.0)
            input_size: input image size for the model (reduced for performance)
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size  # Reduced from 640 to 416 for better performance
        self.pose_model = None
        self.pose_enabled = True
        
        # Load YOLOv8 pose model
        model_name = f"yolov8{model_size}-pose.pt"
        model_path = get_model_path(model_name)
        
        print(f"Loading YOLOv8 Pose model...")
        
        try:
            # Try to load from local directory first
            if check_model_exists(model_path, "YOLOv8 Pose model"):
                self.pose_model = YOLO(model_path)
            else:
                # If not found locally, try to download from ultralytics
                print(f"Downloading {model_name} from ultralytics...")
                self.pose_model = YOLO(model_name)
            print("✅ YOLOv8 Pose model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLOv8 pose model: {e}")
            self.pose_model = None
            return
        
        # COCO pose keypoints (17 points)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for drawing (COCO format)
        self.skeleton_connections = [
            # Face
            [0, 1], [0, 2], [1, 3], [2, 4],
            # Arms
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            # Body
            [5, 11], [6, 12], [11, 12],
            # Legs
            [11, 13], [13, 15], [12, 14], [14, 16]
        ]
        
        # Colors for different body parts
        self.colors = {
            'face': (255, 255, 0),      # Yellow
            'arms': (0, 255, 0),        # Green
            'body': (255, 0, 0),        # Red
            'legs': (0, 0, 255),        # Blue
            'keypoints': (0, 255, 255)  # Cyan
        }
    
    def detect_poses(self, image):
        """Detect poses in image using YOLOv8"""
        if self.pose_model is None or not self.pose_enabled:
            return []
        
        try:
            results = self.pose_model(image, imgsz=self.input_size, verbose=False)
            poses = []
            
            for result in results:
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    boxes_data = result.boxes.data if result.boxes is not None else None
                    
                    for i, person_keypoints in enumerate(keypoints_data):
                        bbox_confidence = boxes_data[i][4].item() if boxes_data is not None and i < len(boxes_data) else 1.0
                        
                        if bbox_confidence >= self.confidence_threshold:
                            kpts = person_keypoints.cpu().numpy()
                            bbox = boxes_data[i][:4].cpu().numpy() if boxes_data is not None and i < len(boxes_data) else None
                            
                            pose_data = {
                                'person_id': i,
                                'keypoints': kpts,
                                'bbox_confidence': bbox_confidence,
                                'bbox': bbox,
                                'visible_keypoints': np.sum(kpts[:, 2] > 0.5)
                            }
                            
                            poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            print(f"Pose detection error: {e}")
            return []
    
    def draw_poses(self, image, poses, draw_skeleton=True, draw_keypoints=True):
        """Draw detected poses on image"""
        annotated_image = image.copy()
        
        for pose in poses:
            keypoints = pose['keypoints']
            
            # Draw skeleton connections
            if draw_skeleton:
                for connection in self.skeleton_connections:
                    kpt1_idx, kpt2_idx = connection
                    
                    if (keypoints[kpt1_idx][2] > 0.5 and keypoints[kpt2_idx][2] > 0.5):
                        pt1 = (int(keypoints[kpt1_idx][0]), int(keypoints[kpt1_idx][1]))
                        pt2 = (int(keypoints[kpt2_idx][0]), int(keypoints[kpt2_idx][1]))
                        
                        # Choose color based on body part
                        if kpt1_idx <= 4 or kpt2_idx <= 4:  # Face
                            color = self.colors['face']
                        elif kpt1_idx <= 10 or kpt2_idx <= 10:  # Arms
                            color = self.colors['arms']
                        elif kpt1_idx <= 12 or kpt2_idx <= 12:  # Body
                            color = self.colors['body']
                        else:  # Legs
                            color = self.colors['legs']
                        
                        cv2.line(annotated_image, pt1, pt2, color, 2)
            
            # Draw keypoints
            if draw_keypoints:
                for i, (x, y, visibility) in enumerate(keypoints):
                    if visibility > 0.5:
                        if i <= 4:  # Face keypoints
                            color = self.colors['face']
                            radius = 3
                        else:  # Body keypoints
                            color = self.colors['keypoints']
                            radius = 4
                        
                        cv2.circle(annotated_image, (int(x), int(y)), radius, color, -1)
        
        return annotated_image

class AIDetector:
    def __init__(self):
        # Paths to the AI models (using relative paths)
        self.gun_model_path = get_model_path("Gun Detection/GunModel.pt")
        self.grenade_model_path = get_model_path("Grenade Detection/best.pt")
        
        # Load models
        self.gun_model = None
        self.grenade_model = None
        self.models_loaded = False
        
        # Detection settings
        self.gun_confidence_threshold = 0.75
        self.grenade_confidence_threshold = 0.85
        self.detection_enabled = True
        
        # Mode toggle: 'weapon' or 'pose'
        self.detection_mode = 'weapon'  # Default to weapon detection
        
        # Alternating weapon detection untuk mengurangi load
        self.weapon_frame_counter = 0
        self.weapon_detection_interval = 3  # Alternate between gun and grenade every 3 frames
        
        # Initialize pose detector
        self.pose_detector = PoseDetector()
        
        self.load_models()
    
    def load_models(self):
        """Load YOLOv8 models for gun and grenade detection"""
        try:
            print("Loading AI detection models...")
            print(f"Script running from: {os.path.dirname(os.path.abspath(__file__))}")
            
            # Load gun detection model
            if check_model_exists(self.gun_model_path, "Gun detection model"):
                self.gun_model = YOLO(self.gun_model_path)
                print("✅ Gun detection model loaded successfully")
            else:
                print("⚠️ Gun detection model not available")
            
            # Load grenade detection model
            if check_model_exists(self.grenade_model_path, "Grenade detection model"):
                self.grenade_model = YOLO(self.grenade_model_path)
                print("✅ Grenade detection model loaded successfully")
            else:
                print("⚠️ Grenade detection model not available")
            
            self.models_loaded = (self.gun_model is not None) or (self.grenade_model is not None)
            
            if self.models_loaded:
                print("🤖 Weapon Detection system ready!")
                if self.gun_model: print("   - Gun detection: READY")
                if self.grenade_model: print("   - Grenade detection: READY")
            else:
                print("⚠️ Running without weapon detection models")
                print("   Only pose detection will be available")
                
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
            self.models_loaded = False
    
    def detect_objects(self, frame):
        """Run detection on frame and return annotated frame"""
        if not self.detection_enabled:
            return frame
        
        detections = []
        poses = []
        
        try:
            # Check detection mode and run appropriate detection
            if self.detection_mode == 'pose':
                # Only run pose detection
                if self.pose_detector.pose_enabled and self.pose_detector.pose_model is not None:
                    poses = self.pose_detector.detect_poses(frame)
                    frame = self.pose_detector.draw_poses(frame, poses, 
                                                         draw_skeleton=True, 
                                                         draw_keypoints=True)
            
            elif self.detection_mode == 'weapon':
                # Optimized weapon detection - alternate between gun and grenade
                if self.models_loaded:
                    self.weapon_frame_counter += 1
                    
                    # Alternate between gun and grenade detection untuk mengurangi load
                    if self.weapon_frame_counter % self.weapon_detection_interval == 0:
                        current_weapon_model = 'grenade'
                    else:
                        current_weapon_model = 'gun'
                    
                    # Gun detection
                    if current_weapon_model == 'gun' and self.gun_model is not None:
                        gun_results = self.gun_model(frame, imgsz=416, stream=True, verbose=False)  # Reduced size + disable verbose
                        for result in gun_results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    conf = float(box.conf[0])
                                    cls_id = int(box.cls[0])
                                    
                                    if conf >= self.gun_confidence_threshold:
                                        label = self.gun_model.names[cls_id]
                                        detections.append({
                                            'type': 'gun',
                                            'bbox': (x1, y1, x2, y2),
                                            'confidence': conf,
                                            'label': label,
                                            'color': (0, 0, 255)  # Red for guns
                                        })
                    
                    # Grenade detection
                    elif current_weapon_model == 'grenade' and self.grenade_model is not None:
                        grenade_results = self.grenade_model(frame, imgsz=416, stream=True, verbose=False)  # Reduced size + disable verbose
                        for result in grenade_results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    conf = float(box.conf[0])
                                    cls_id = int(box.cls[0])
                                    
                                    if conf >= self.grenade_confidence_threshold:
                                        label = self.grenade_model.names[cls_id]
                                        detections.append({
                                            'type': 'grenade',
                                            'bbox': (x1, y1, x2, y2),
                                            'confidence': conf,
                                            'label': label,
                                            'color': (0, 255, 0)  # Green for grenades
                                        })
                
                # Draw weapon detections on frame
                annotated_frame = self.draw_detections(frame, detections)
                frame = annotated_frame
            
            # Add detection info overlay
            frame = self.draw_info_overlay(frame, len(poses), len(detections))
            
            return frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            label = detection['label']
            color = detection['color']
            det_type = detection['type']
            
            # Draw bounding box with thicker line for visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text with type identifier
            text = f"{det_type.upper()}: {label} ({conf:.2f})"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Draw text directly without background
            cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale, color, thickness)
        
        return frame
    
    def draw_info_overlay(self, frame, pose_count, weapon_count):
        """Clean interface - no overlay display"""
        # Return frame without any overlay modifications
        return frame

class VideoStreamer:
    def __init__(self, camera_index=0, fps=30, quality=50):
        self.camera_index = camera_index
        self.fps = fps
        self.quality = quality  # Adjusted quality for 480p resolution
        self.frame_queue = queue.Queue(maxsize=1)  # Reduce buffer for lower latency
        self.cap = None
        self.running = False
        # Standard definition resolution for balanced performance
        self.frame_width = 320  # Standard width (480p)
        self.frame_height = 240  # Standard height (480p)
        
        # Skip frame processing for performance
        self.frame_skip_counter = 0
        self.process_every_nth_frame = 2  # Process every 2nd frame only
        
        # Initialize AI detector
        self.detector = AIDetector()
        
        # Cache for last processed frame
        self.last_processed_frame = None
        
    def start_camera(self):
        """Inisialisasi kamera dengan pengaturan optimal untuk latency rendah"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Pengaturan untuk latency rendah
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
        
        # Pengaturan codec jika tersedia
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        if not self.cap.isOpened():
            raise RuntimeError("Tidak dapat membuka kamera")
            
        print(f"Kamera berhasil diinisialisasi: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        
    def capture_frames(self):
        """Thread untuk mengambil frame dari kamera dan menjalankan deteksi AI dengan optimisasi"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Frame skipping untuk mengurangi load processing
                self.frame_skip_counter += 1
                
                # Hanya proses AI detection setiap N frame
                if self.frame_skip_counter >= self.process_every_nth_frame:
                    if self.detector.detection_enabled:
                        processed_frame = self.detector.detect_objects(frame)
                        self.last_processed_frame = processed_frame
                    else:
                        self.last_processed_frame = frame
                    self.frame_skip_counter = 0
                else:
                    # Gunakan frame terakhir yang sudah diproses untuk mengurangi lag
                    processed_frame = self.last_processed_frame if self.last_processed_frame is not None else frame
                
                # Gunakan frame yang sudah diproses atau frame asli
                final_frame = self.last_processed_frame if self.last_processed_frame is not None else frame
                
                # Bersihkan queue lama untuk mengurangi latency
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(final_frame)
                except queue.Full:
                    pass
            
            time.sleep(1/(self.fps * 1.5))  # Slight adjustment for performance
    
    def get_frame(self):
        """Mendapatkan frame terbaru"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def encode_frame(self, frame):
        """Encode frame ke JPEG dengan kualitas yang dapat disesuaikan"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        return encoded_img.tobytes()
    
    def start(self):
        """Memulai streaming"""
        self.running = True
        self.start_camera()
        
        # Mulai thread untuk capture frame
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("Video streamer dengan AI detection dimulai")
    
    def stop(self):
        """Menghentikan streaming"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("Video streamer dihentikan")

# Instance global streamer
streamer = VideoStreamer()

# Global variable untuk menyimpan data sensor
sensor_data = {
    'temperature': 0.0,
    'humidity': 0.0,
    'timestamp': time.time()
}

def generate_frames():
    """Generator untuk streaming frame"""
    while True:
        frame = streamer.get_frame()
        if frame is not None:
            # Encode frame ke JPEG
            buffer = streamer.encode_frame(frame)
            
            # Format untuk HTTP streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')
        else:
            time.sleep(0.01)  # Tunggu sebentar jika tidak ada frame

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint untuk streaming video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Status server"""
    return jsonify({
        'status': 'running' if streamer.running else 'stopped',
        'fps': streamer.fps,
        'quality': streamer.quality,
        'resolution': f"{streamer.frame_width}x{streamer.frame_height}",
        'ai_detection': {
            'enabled': streamer.detector.detection_enabled,
            'detection_mode': streamer.detector.detection_mode,
            'models_loaded': streamer.detector.models_loaded,
            'gun_model_loaded': streamer.detector.gun_model is not None,
            'grenade_model_loaded': streamer.detector.grenade_model is not None,
            'gun_confidence': streamer.detector.gun_confidence_threshold,
            'grenade_confidence': streamer.detector.grenade_confidence_threshold,
            'pose_enabled': streamer.detector.pose_detector.pose_enabled,
            'pose_model_loaded': streamer.detector.pose_detector.pose_model is not None
        }
    })

@app.route('/settings/<int:fps>/<int:quality>')
def update_settings(fps, quality):
    """Update pengaturan streaming"""
    streamer.fps = max(1, min(60, fps))
    streamer.quality = max(10, min(100, quality))
    return jsonify({
        'fps': streamer.fps,
        'quality': streamer.quality,
        'message': 'Pengaturan berhasil diupdate'
    })

@app.route('/ai_settings', methods=['POST'])
def update_ai_settings():
    """Update pengaturan AI detection"""
    try:
        data = request.get_json()
        
        if 'detection_enabled' in data:
            streamer.detector.detection_enabled = bool(data['detection_enabled'])
        
        if 'gun_confidence' in data:
            confidence = float(data['gun_confidence'])
            streamer.detector.gun_confidence_threshold = max(0.1, min(1.0, confidence))
        
        if 'grenade_confidence' in data:
            confidence = float(data['grenade_confidence'])
            streamer.detector.grenade_confidence_threshold = max(0.1, min(1.0, confidence))
        
        if 'pose_enabled' in data:
            streamer.detector.pose_detector.pose_enabled = bool(data['pose_enabled'])
        
        if 'pose_confidence' in data:
            confidence = float(data['pose_confidence'])
            streamer.detector.pose_detector.confidence_threshold = max(0.1, min(1.0, confidence))
        
        return jsonify({
            'status': 'success',
            'detection_enabled': streamer.detector.detection_enabled,
            'gun_confidence': streamer.detector.gun_confidence_threshold,
            'grenade_confidence': streamer.detector.grenade_confidence_threshold,
            'pose_enabled': streamer.detector.pose_detector.pose_enabled,
            'pose_confidence': streamer.detector.pose_detector.confidence_threshold,
            'message': 'AI settings berhasil diupdate'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/toggle_detection')
def toggle_detection():
    """Toggle AI detection on/off"""
    streamer.detector.detection_enabled = not streamer.detector.detection_enabled
    status_text = "enabled" if streamer.detector.detection_enabled else "disabled"
    
    return jsonify({
        'detection_enabled': streamer.detector.detection_enabled,
        'message': f'AI detection {status_text}'
    })

@app.route('/toggle_pose')
def toggle_pose():
    """Toggle pose detection on/off"""
    streamer.detector.pose_detector.pose_enabled = not streamer.detector.pose_detector.pose_enabled
    status_text = "enabled" if streamer.detector.pose_detector.pose_enabled else "disabled"
    
    return jsonify({
        'pose_enabled': streamer.detector.pose_detector.pose_enabled,
        'message': f'Pose detection {status_text}'
    })

@app.route('/toggle_mode')
def toggle_mode():
    """Toggle between weapon and pose detection modes"""
    if streamer.detector.detection_mode == 'weapon':
        streamer.detector.detection_mode = 'pose'
        mode_name = "Pose Detection"
    else:
        streamer.detector.detection_mode = 'weapon'
        mode_name = "Weapon Detection"
    
    return jsonify({
        'detection_mode': streamer.detector.detection_mode,
        'mode_name': mode_name,
        'message': f'Switched to {mode_name} mode'
    })

@app.route('/performance_mode/<mode>')
def set_performance_mode(mode):
    """Set performance mode: 'fast', 'balanced', 'quality'"""
    if mode == 'fast':
        streamer.process_every_nth_frame = 3
        streamer.detector.weapon_detection_interval = 5
        message = "Fast mode: Lower quality, higher FPS"
    elif mode == 'balanced':
        streamer.process_every_nth_frame = 2
        streamer.detector.weapon_detection_interval = 3
        message = "Balanced mode: Medium quality and FPS"
    elif mode == 'quality':
        streamer.process_every_nth_frame = 1
        streamer.detector.weapon_detection_interval = 1
        message = "Quality mode: Best quality, lower FPS"
    else:
        return jsonify({'error': 'Invalid mode. Use: fast, balanced, quality'}), 400
    
    return jsonify({
        'performance_mode': mode,
        'frame_skip': streamer.process_every_nth_frame,
        'weapon_interval': streamer.detector.weapon_detection_interval,
        'message': message
    })

@app.route('/display_config/<display_type>')
def set_display_config(display_type):
    """Configure display settings for 7-inch screens"""
    display_configs = {
        'hd_1080x720': {
            'width': 1080, 'height': 720, 
            'name': 'HD 1080x720 (High Quality)',
            'quality': 85
        },
        'standard_640x480': {
            'width': 640, 'height': 480, 
            'name': 'Standard 640x480 (Balanced)',
            'quality': 75
        },
        '7inch_800x480': {
            'width': 640, 'height': 480, 
            'name': '7 Inch 800x480 (Recommended)',
            'quality': 75
        },
        '7inch_1024x600': {
            'width': 640, 'height': 480, 
            'name': '7 Inch 1024x600 (Widescreen)',
            'quality': 70
        },
        '7inch_compact': {
            'width': 480, 'height': 360, 
            'name': '7 Inch Compact (High Performance)',
            'quality': 80
        },
        # NEW: Jetson-optimized configurations
        'jetson_7inch_performance': {
            'width': 512, 'height': 384,
            'name': 'Jetson 7-Inch Performance (RECOMMENDED for Jetson)',
            'quality': 65,
            'frame_skip': 3,
            'ai_input_size': 320
        },
        'jetson_7inch_balanced': {
            'width': 640, 'height': 480,
            'name': 'Jetson 7-Inch Balanced',  
            'quality': 60,
            'frame_skip': 4,
            'ai_input_size': 320
        },
        'default': {
            'width': 640, 'height': 480, 
            'name': 'Standard 480p Resolution',
            'quality': 75
        }
    }
    
    if display_type not in display_configs:
        display_type = 'default'
    
    config = display_configs[display_type]
    
    # Apply new resolution (would require camera restart in production)
    try:
        if streamer.cap and streamer.cap.isOpened():
            streamer.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            streamer.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
            streamer.frame_width = config['width']
            streamer.frame_height = config['height']
            streamer.quality = config['quality']
            
            # Apply Jetson-specific optimizations if available
            if 'frame_skip' in config:
                streamer.process_every_nth_frame = config['frame_skip']
            if 'ai_input_size' in config:
                if hasattr(streamer.detector, 'pose_detector'):
                    streamer.detector.pose_detector.input_size = config['ai_input_size']
        
        return jsonify({
            'display_type': display_type,
            'config': config,
            'current_resolution': f"{streamer.frame_width}x{streamer.frame_height}",
            'frame_skip': getattr(streamer, 'process_every_nth_frame', 2),
            'message': f"Display configured for {config['name']}"
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to apply display config: {str(e)}'
        }), 500

@app.route('/set_fps/<int:new_fps>')
def set_fps(new_fps):
    """Set FPS dengan validasi dan warning"""
    global streamer
    
    # Validasi FPS range
    if new_fps < 5 or new_fps > 120:
        return jsonify({
            'error': 'FPS harus antara 5-120',
            'current_fps': streamer.fps
        }), 400
    
    # Warning untuk FPS tinggi
    warnings = []
    if new_fps > 60:
        warnings.append("⚠️ FPS >60: Sangat berat untuk CPU/GPU")
    elif new_fps > 45:
        warnings.append("⚠️ FPS >45: Konsumsi bandwidth tinggi")
    elif new_fps > 30:
        warnings.append("⚠️ FPS >30: Penggunaan CPU meningkat")
    
    # Set FPS baru
    old_fps = streamer.fps
    streamer.fps = new_fps
    
    # Update kamera jika sedang berjalan
    if streamer.cap and streamer.cap.isOpened():
        streamer.cap.set(cv2.CAP_PROP_FPS, new_fps)
    
    # Estimasi impact
    cpu_impact = min(100, (new_fps / 30) * 50)  # Base 50% pada 30fps
    bandwidth_impact = (new_fps / 30) * 100  # Persentase increase
    
    return jsonify({
        'status': 'success',
        'old_fps': old_fps,
        'new_fps': new_fps,
        'warnings': warnings,
        'estimated_impact': {
            'cpu_usage_percent': f"{cpu_impact:.0f}%",
            'bandwidth_increase': f"{bandwidth_impact:.0f}%",
            'memory_usage': f"{(new_fps/30)*60:.0f}MB"
        },
        'recommendation': get_fps_recommendation(new_fps),
        'message': f'FPS changed from {old_fps} to {new_fps}'
    })

def get_fps_recommendation(fps):
    """Get recommendation based on FPS"""
    if fps <= 15:
        return "💡 Low FPS: Cocok untuk monitoring hemat battery"
    elif fps <= 30:
        return "✅ Optimal: Balance antara smoothness dan performance"
    elif fps <= 45:
        return "⚡ High FPS: Smooth tapi konsumsi resource tinggi"
    elif fps <= 60:
        return "🔥 Very High: Butuh hardware kuat, bandwidth besar"
    else:
        return "🚀 Extreme: Hanya untuk hardware premium"

@app.route('/sensor_data', methods=['POST'])
def receive_sensor_data():
    """Endpoint untuk menerima data sensor dari ESP"""
    global sensor_data
    try:
        data = request.get_json()
        if data:
            sensor_data['temperature'] = float(data.get('temperature', 0))
            sensor_data['humidity'] = float(data.get('humidity', 0))
            sensor_data['timestamp'] = time.time()
            
            return jsonify({
                'status': 'success',
                'message': 'Data sensor berhasil diterima'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Data tidak valid'
            }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_sensor_data')
def get_sensor_data():
    """Endpoint untuk mengambil data sensor terbaru"""
    global sensor_data
    return jsonify({
        'temperature': sensor_data['temperature'],
        'humidity': sensor_data['humidity'],
        'timestamp': sensor_data['timestamp'],
        'last_update': time.time() - sensor_data['timestamp']
    })

def get_local_ip():
    """Mendapatkan IP address lokal"""
    try:
        # Koneksi ke server eksternal untuk mendapatkan IP lokal
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    try:
        # Mulai streamer
        streamer.start()
        
        # Dapatkan IP lokal
        local_ip = get_local_ip()
        port = 5000
        
        print(f"\n{'='*60}")
        print(f"ENHANCED VIDEO STREAMING SERVER WITH AI DETECTION")
        print(f"{'='*60}")
        print(f"Server berjalan di:")
        print(f"  Local:    http://localhost:{port}")
        print(f"  Network:  http://{local_ip}:{port}")
        print(f"\nFitur yang tersedia:")
        print(f"  🎥 Video streaming dengan latency rendah")
        print(f"  🤖 AI detection untuk senjata dan granat")
        print(f"  🤸 Pose detection dengan skeleton tracking")
        print(f"  📊 Sensor data monitoring")
        print(f"  ⚙️  Real-time settings adjustment")
        print(f"\nAPI Endpoints:")
        print(f"  GET  /status           - Status server dan AI models")
        print(f"  GET  /toggle_detection - Toggle weapon detection")
        print(f"  GET  /toggle_pose      - Toggle pose detection")
        print(f"  GET  /toggle_mode      - Switch between weapon/pose modes")
        print(f"  GET  /performance_mode/<mode> - Set performance (fast/balanced/quality)")
        print(f"  POST /ai_settings      - Update AI confidence thresholds")
        print(f"  GET  /settings/<fps>/<quality> - Update video settings")
        print(f"\nUntuk mengakses dari perangkat lain di jaringan yang sama,")
        print(f"gunakan: http://{local_ip}:{port}")
        print(f"{'='*60}\n")
        
        # Jalankan Flask server
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nMenghentikan server...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        streamer.stop()
        cv2.destroyAllWindows()