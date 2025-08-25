
#!/usr/bin/env python3
"""
AI Video Server with Detection
Server video streaming dengan AI detection untuk gun, grenade, pose detection, dan face mesh
"""

import cv2
import threading
import time
from flask import Flask, Response, render_template, jsonify
import socket
import queue
from ultralytics import YOLO
import os
import mediapipe as mp

print(f"🤖 AI Video Server Starting...")
print(f"Working directory: {os.getcwd()}")


# ===== PATCH BUZZER START =====
import serial
import threading
import time


# Variabel global untuk data sensor
buzzer_state = "OFF"
sensor_detect = False
sensor_temperature = 0.0
sensor_humidity = 0.0

def read_serial():
    global buzzer_state, sensor_detect, sensor_temperature, sensor_humidity, sensor_data
    try:
        ser = serial.Serial('COM3', 115200, timeout=1)  # Ganti COM & baudrate sesuai ESP32
        time.sleep(2)
        while True:
            line = ser.readline().decode().strip()
            if not line:
                continue
            # Format: detect*temperature*humidity
            parts = line.split('*')
            if len(parts) == 3:
                detect_str, temp_str, hum_str = parts
                # detect: 'true'/'false' (string)
                sensor_detect = detect_str.lower() == 'true'
                buzzer_state = "ON" if sensor_detect else "OFF"
                try:
                    sensor_temperature = float(temp_str)
                except:
                    sensor_temperature = -99.0
                try:
                    sensor_humidity = float(hum_str)
                except:
                    sensor_humidity = -99.0
                # Update global sensor_data agar endpoint tetap backward compatible
                sensor_data['temperature'] = sensor_temperature
                sensor_data['humidity'] = sensor_humidity
                sensor_data['timestamp'] = time.time()
    except Exception as e:
        print("Serial error:", e)

serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

app = Flask(__name__)

def get_model_path(relative_path):
    """Get absolute path for model files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, relative_path)
    return os.path.normpath(model_path)

def check_model_exists(model_path, model_name):
    """Check if model exists and print debug information"""
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ {model_name} found: {model_path} ({file_size:.1f} MB)")
        return True
    else:
        print(f"❌ {model_name} not found: {model_path}")
        return False

class AIDetector:
    """AI Object Detection Class"""
    
    def __init__(self):
        self.detection_enabled = True
        self.detection_mode = 'weapon'  # 'weapon', 'pose', 'off'
        
        # YOLO Models
        self.gun_model = None
        self.grenade_model = None
        self.pose_model = None
        
        # MediaPipe Models
        self.mp_face_mesh = None
        self.face_mesh = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        
        self.models_loaded = False
        
        # Detection settings
        self.gun_confidence_threshold = 0.5
        self.grenade_confidence_threshold = 0.5
        self.pose_confidence_threshold = 0.5
        self.face_mesh_confidence_threshold = 0.5
        
        self.load_models()
    
    def load_models(self):
        """Load all AI models"""
        try:
            print("Loading AI Models...")
            
            # Load gun detection model
            gun_model_path = get_model_path("Gun Detection/GunModel.pt")
            if check_model_exists(gun_model_path, "Gun detection model"):
                self.gun_model = YOLO(gun_model_path)
                print("✅ Gun detection model loaded successfully")
            
            # Load grenade detection model
            grenade_model_path = get_model_path("Grenade Detection/best.pt")
            if check_model_exists(grenade_model_path, "Grenade detection model"):
                self.grenade_model = YOLO(grenade_model_path)
                print("✅ Grenade detection model loaded successfully")
            
            # Load pose detection model
            pose_model_path = get_model_path("yolov8n-pose.pt")
            if check_model_exists(pose_model_path, "Pose detection model"):
                self.pose_model = YOLO(pose_model_path)
                print("✅ Pose detection model loaded successfully")
            
            # Initialize MediaPipe Face Mesh
            print("Loading MediaPipe Face Mesh...")
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=self.face_mesh_confidence_threshold,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Face Mesh loaded successfully")
            
            self.models_loaded = True
            print("🤖 AI Detection system ready!")
            
        except Exception as e:
            print(f"❌ Error loading AI models: {e}")
            self.models_loaded = False
    
    def detect_objects(self, frame):
        """Main detection method"""
        if not self.detection_enabled or not self.models_loaded:
            return frame
        
        try:
            # Weapon detection (gun + grenade)
            if self.detection_mode == 'weapon':
                if self.gun_model:
                    frame = self.detect_guns(frame)
                if self.grenade_model:
                    frame = self.detect_grenades(frame)
            
            # Pose detection with Face Mesh
            elif self.detection_mode == 'pose':
                if self.pose_model:
                    frame = self.detect_poses(frame)
                if self.face_mesh:
                    frame = self.detect_face_mesh(frame)
            
            return frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame
    
    def detect_guns(self, frame):
        """Gun detection"""
        try:
            results = self.gun_model(frame, imgsz=640, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf[0].item()
                        if conf >= self.gun_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Draw label
                            label = f"Gun: {conf:.2f}"
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w+10, y1), (0, 0, 255), -1)
                            cv2.putText(frame, label, (x1+5, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            print(f"Gun detection error: {e}")
        
        return frame
    
    def detect_grenades(self, frame):
        """Grenade detection"""
        try:
            results = self.grenade_model(frame, imgsz=640, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf[0].item()
                        if conf >= self.grenade_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                            
                            # Draw label
                            label = f"Grenade: {conf:.2f}"
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w+10, y1), (255, 0, 0), -1)
                            cv2.putText(frame, label, (x1+5, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            print(f"Grenade detection error: {e}")
        
        return frame
    
    def detect_poses(self, frame):
        """Pose detection"""
        try:
            results = self.pose_model(frame, imgsz=640, verbose=False)
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    
                    for person_kpts in keypoints:
                        # Draw keypoints
                        for i, (x, y, conf) in enumerate(person_kpts):
                            if conf >= self.pose_confidence_threshold:
                                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        
                        # Draw skeleton connections (simplified)
                        connections = [
                            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
                            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  # Legs
                            [5, 11], [6, 12]  # Body
                        ]
                        
                        for connection in connections:
                            kpt1, kpt2 = connection
                            if (kpt1 < len(person_kpts) and kpt2 < len(person_kpts) and
                                person_kpts[kpt1][2] >= self.pose_confidence_threshold and
                                person_kpts[kpt2][2] >= self.pose_confidence_threshold):
                                
                                x1, y1 = int(person_kpts[kpt1][0]), int(person_kpts[kpt1][1])
                                x2, y2 = int(person_kpts[kpt2][0]), int(person_kpts[kpt2][1])
                                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        
        except Exception as e:
            print(f"Pose detection error: {e}")
        
        return frame
    
    def detect_face_mesh(self, frame):
        """Face Mesh detection using MediaPipe"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh contours only
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
        except Exception as e:
            print(f"Face mesh detection error: {e}")
        
        return frame


class VideoStreamer:
    """Video streaming class with AI detection"""
    
    def __init__(self, camera_index=0, fps=45, quality=90):
        self.camera_index = camera_index
        self.fps = fps
        self.quality = quality
        self.frame_queue = queue.Queue(maxsize=2)
        self.cap = None
        self.running = False
        
        # Initialize AI detector
        self.detector = AIDetector()
        
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if self.cap.isOpened():
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
        else:
            print("❌ Failed to open camera")
            return False
    
    def capture_frames(self):
        """Frame capture with AI detection"""
        print("🎥 Starting video capture with AI detection...")
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply AI detection
                processed_frame = self.detector.detect_objects(frame)
                
                # Clear old frames
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Add processed frame
                try:
                    self.frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass
            
            time.sleep(1.0 / (self.fps * 1.1))
    
    def get_frame(self):
        """Get latest frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def encode_frame(self, frame):
        """Encode frame"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        return encoded_img.tobytes()
    
    def start(self):
        """Start streamer"""
        if self.start_camera():
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print("🚀 Video streamer started!")
        else:
            print("❌ Failed to start camera")
    
    def stop(self):
        """Stop streamer"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("⏹️ Video streamer stopped")


# Global streamer instance
streamer = VideoStreamer(fps=30, quality=90)

# Sensor data (dummy)
sensor_data = {
    'temperature': 25.0,
    'humidity': 60.0,
    'timestamp': time.time()
}

def generate_frames():
    """Frame generator for video streaming"""
    while True:
        frame = streamer.get_frame()
        if frame is not None:
            buffer = streamer.encode_frame(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Server status"""
    return jsonify({
        'server_status': 'running',
        'camera_status': 'connected' if streamer.cap and streamer.cap.isOpened() else 'disconnected',
        'ai_detection': {
            'detection_mode': streamer.detector.detection_mode,
            'detection_enabled': streamer.detector.detection_enabled,
            'models_loaded': streamer.detector.models_loaded,
            'gun_model_loaded': streamer.detector.gun_model is not None,
            'grenade_model_loaded': streamer.detector.grenade_model is not None,
            'pose_model_loaded': streamer.detector.pose_model is not None,
            'face_mesh_loaded': streamer.detector.face_mesh is not None
        },
        'settings': {
            'fps': streamer.fps,
            'quality': streamer.quality,
            'gun_confidence': streamer.detector.gun_confidence_threshold,
            'grenade_confidence': streamer.detector.grenade_confidence_threshold,
            'pose_confidence': streamer.detector.pose_confidence_threshold,
            'face_mesh_confidence': streamer.detector.face_mesh_confidence_threshold
        },
        "buzzer": buzzer_state,
        "sensor": {
            "detect": sensor_detect,
            "temperature": sensor_temperature,
            "humidity": sensor_humidity
        }
    })

@app.route('/toggle_detection')
def toggle_detection():
    """Toggle AI detection on/off"""
    streamer.detector.detection_enabled = not streamer.detector.detection_enabled
    status = "enabled" if streamer.detector.detection_enabled else "disabled"
    return jsonify({
        'detection_enabled': streamer.detector.detection_enabled,
        'message': f'AI detection {status}'
    })

@app.route('/set_detection_mode/<mode>')
def set_detection_mode(mode):
    """Set detection mode"""
    valid_modes = ['weapon', 'pose', 'off']
    if mode not in valid_modes:
        return jsonify({'error': 'Invalid mode', 'valid_modes': valid_modes}), 400
    
    streamer.detector.detection_mode = mode
    return jsonify({
        'detection_mode': mode,
        'message': f'Detection mode set to {mode}'
    })

@app.route('/toggle_mode')
def toggle_mode():
    """Toggle between weapon detection and pose detection modes"""
    current_mode = streamer.detector.detection_mode
    
    # Toggle between weapon and pose mode
    if current_mode == 'weapon':
        streamer.detector.detection_mode = 'pose'
        new_mode = 'pose'
    else:
        streamer.detector.detection_mode = 'weapon'
        new_mode = 'weapon'
    
    return jsonify({
        'detection_mode': new_mode,
        'message': f'Switched to {new_mode} mode'
    })

@app.route('/get_sensor_data')
def get_sensor_data():
    global sensor_data, sensor_detect, sensor_temperature, sensor_humidity
    return jsonify({
        'detect': sensor_detect,
        'temperature': sensor_temperature,
        'humidity': sensor_humidity,
        'timestamp': sensor_data['timestamp'],
        'last_update': time.time() - sensor_data['timestamp']
    })

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    try:
        # Start streamer
        streamer.start()
        
        local_ip = get_local_ip()
        port = 5000
        
        print(f"\n{'='*50}")
        print(f"🤖 AI VIDEO SERVER")
        print(f"{'='*50}")
        print(f"Server berjalan di:")
        print(f"  Local:    http://localhost:{port}")
        print(f"  Network:  http://{local_ip}:{port}")
        print(f"\n🎯 Detection Modes:")
        print(f"  weapon  - Gun + Grenade detection")
        print(f"  pose    - Pose + Face detection")
        print(f"  off     - All detections disabled")
        print(f"\nAPI Endpoints:")
        print(f"  GET  /status           - Server status")
        print(f"  GET  /toggle_detection - Toggle detection")
        print(f"  GET  /toggle_mode      - Switch detection mode")
        print(f"  GET  /get_sensor_data  - Get sensor data")
        print(f"{'='*50}\n")
        
        # Run Flask server
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nStopping AI video server...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        streamer.stop()
        cv2.destroyAllWindows()



# Tambahkan buzzer ke response status

# ===== PATCH BUZZER END =====
