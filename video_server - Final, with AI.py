#!/usr/bin/env python3
"""
Enhanced Video Streaming Server with AI Detection
Menggunakan Flask, OpenCV, dan YOLOv8 untuk streaming video dengan deteksi senjata, granat, dan pose
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

app = Flask(__name__)

class PoseDetector:
    def __init__(self, model_size='n', confidence_threshold=0.5, input_size=640):
        """
        Initialize YOLOv8 Pose detector
        
        Args:
            model_size: 'n' (nano/fastest), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
            confidence_threshold: minimum confidence for pose detection (0.0-1.0)
            input_size: input image size for the model
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.pose_model = None
        self.pose_enabled = True
        
        # Load YOLOv8 pose model
        model_name = f"yolov8{model_size}-pose.pt"
        print(f"Loading YOLOv8 Pose model: {model_name}")
        
        try:
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
        # Paths to the AI models
        self.gun_model_path = r"C:\Users\ADMIN\Documents\UNHAN\UNHAN\Gun Detection\GunModel.pt"
        self.grenade_model_path = r"C:\Users\ADMIN\Documents\UNHAN\UNHAN\Grenade Detection\best.pt"
        
        # Load models
        self.gun_model = None
        self.grenade_model = None
        self.models_loaded = False
        
        # Detection settings
        self.gun_confidence_threshold = 0.75
        self.grenade_confidence_threshold = 0.85
        self.detection_enabled = True
        
        # Initialize pose detector
        self.pose_detector = PoseDetector()
        
        self.load_models()
    
    def load_models(self):
        """Load YOLOv8 models for gun and grenade detection"""
        try:
            print("Loading AI detection models...")
            
            # Load gun detection model
            if os.path.exists(self.gun_model_path):
                self.gun_model = YOLO(self.gun_model_path)
                print("✅ Gun detection model loaded successfully")
            else:
                print(f"❌ Gun model not found at: {self.gun_model_path}")
            
            # Load grenade detection model
            if os.path.exists(self.grenade_model_path):
                self.grenade_model = YOLO(self.grenade_model_path)
                print("✅ Grenade detection model loaded successfully")
            else:
                print(f"❌ Grenade model not found at: {self.grenade_model_path}")
            
            self.models_loaded = (self.gun_model is not None) or (self.grenade_model is not None)
            
            if self.models_loaded:
                print("🤖 AI Detection system ready!")
            else:
                print("⚠️ No weapon detection models loaded")
                
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
            # Pose detection
            if self.pose_detector.pose_enabled and self.pose_detector.pose_model is not None:
                poses = self.pose_detector.detect_poses(frame)
                frame = self.pose_detector.draw_poses(frame, poses, 
                                                     draw_skeleton=True, 
                                                     draw_keypoints=True)
            
            # Only run weapon detection if models are loaded
            if self.models_loaded:
                # Gun detection
                if self.gun_model is not None:
                    gun_results = self.gun_model(frame, stream=True)
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
                if self.grenade_model is not None:
                    grenade_results = self.grenade_model(frame, stream=True)
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
            
            # Add detection info overlay
            annotated_frame = self.draw_info_overlay(annotated_frame, len(poses), len(detections))
            
            return annotated_frame
            
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
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_info_overlay(self, frame, pose_count, weapon_count):
        """Draw detection information overlay"""
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"People detected: {pose_count}", (20, 35),
                   font, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Weapons detected: {weapon_count}", (20, 60),
                   font, 0.6, (255, 0, 0) if weapon_count > 0 else (0, 255, 0), 2)
        
        status_text = "AI: ON" if self.detection_enabled else "AI: OFF"
        cv2.putText(frame, status_text, (20, 85),
                   font, 0.6, (0, 255, 0) if self.detection_enabled else (0, 0, 255), 2)
        
        pose_status = "Pose: ON" if self.pose_detector.pose_enabled else "Pose: OFF"
        cv2.putText(frame, pose_status, (20, 110),
                   font, 0.6, (0, 255, 0) if self.pose_detector.pose_enabled else (0, 0, 255), 2)
        
        return frame

class VideoStreamer:
    def __init__(self, camera_index=0, fps=30, quality=80):
        self.camera_index = camera_index
        self.fps = fps
        self.quality = quality
        self.frame_queue = queue.Queue(maxsize=2)  # Buffer kecil untuk latency rendah
        self.cap = None
        self.running = False
        self.frame_width = 640
        self.frame_height = 480
        
        # Initialize AI detector
        self.detector = AIDetector()
        
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
        """Thread untuk mengambil frame dari kamera dan menjalankan deteksi AI"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Run AI detection on frame
                if self.detector.detection_enabled:
                    frame = self.detector.detect_objects(frame)
                
                # Bersihkan queue lama untuk mengurangi latency
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            
            time.sleep(1/self.fps)
    
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
        print(f"  GET  /status           - Status server dan AI")
        print(f"  GET  /toggle_detection - Toggle weapon detection")
        print(f"  GET  /toggle_pose      - Toggle pose detection")
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