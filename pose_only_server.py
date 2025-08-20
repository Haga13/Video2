#!/usr/bin/env python3
"""
Pose Detection Only Server
Server khusus untuk pose detection saja (tanpa gun/grenade detection)
"""

import cv2
import threading
import time
import queue
from flask import Flask, Response, render_template, jsonify, request
import socket

# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe tersedia")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("❌ MediaPipe tidak tersedia - Install dengan: pip install mediapipe")
    mp = None

app = Flask(__name__)

# Add CORS headers for compatibility
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

class PoseDetectionServer:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.cap = None
        self.running = False
        self.pose_detection_active = True  # Selalu aktif untuk pose detection only
        
        # Setup MediaPipe Pose
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Pose initialized")
        else:
            self.pose = None
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Start camera
        self.start_camera()
    
    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Cannot open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.running = True
            # Start frame capture thread
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            print("✅ Camera initialized")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def capture_frames(self):
        """Capture frames from camera"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Process frame with pose detection
                    processed_frame = self.process_frame(frame)
                    
                    # Add to queue (drop old frames if queue full)
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame)
                    
                    self.update_fps()
                else:
                    print("❌ Failed to read frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Frame capture error: {e}")
                time.sleep(0.1)
    
    def process_frame(self, frame):
        """Process frame with pose detection"""
        if not MEDIAPIPE_AVAILABLE or not self.pose:
            # Return frame as is if MediaPipe not available
            return frame
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    bgr_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Add pose info
                cv2.putText(bgr_frame, '🤸 POSE DETECTED', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(bgr_frame, 'No pose detected', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Add FPS
            cv2.putText(bgr_frame, f'FPS: {self.current_fps}', (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return bgr_frame
            
        except Exception as e:
            print(f"❌ Pose processing error: {e}")
            return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time

# Global server instance
pose_server = PoseDetectionServer()

# Mock sensor data
sensor_data = {
    'temperature': 25.0,
    'humidity': 60.0,
    'timestamp': time.time()
}

def generate_frames():
    """Video frame generator"""
    while True:
        if not pose_server.frame_queue.empty():
            frame = pose_server.frame_queue.get()
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/')
def index():
    """Main page"""
    return render_template('pose_only.html')

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Get server status"""
    return jsonify({
        'status': 'running' if pose_server.running else 'stopped',
        'pose_detection': pose_server.pose_detection_active,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'fps': pose_server.current_fps,
        'mode': 'pose_only'
    })

@app.route('/get_sensor_data')
def get_sensor_data():
    """Get sensor data"""
    # Update mock sensor data
    sensor_data['temperature'] = 24.0 + (time.time() % 10)
    sensor_data['humidity'] = 55.0 + (time.time() % 20)
    sensor_data['timestamp'] = time.time()
    
    return jsonify({
        'temperature': sensor_data['temperature'],
        'humidity': sensor_data['humidity'],
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
        print(f"\n{'='*50}")
        print(f"🤸 POSE DETECTION ONLY SERVER")
        print(f"{'='*50}")
        
        if MEDIAPIPE_AVAILABLE:
            print("✅ MediaPipe available - Pose detection ready!")
            print("🎯 Mode: Pose Detection Only")
        else:
            print("❌ MediaPipe not available")
            print("💡 Install dengan: pip install mediapipe")
        
        print(f"\n📹 Starting frame capture...")
        print(f"✅ Pose detection server started")
        
        # Get network info
        local_ip = get_local_ip()
        port = 5001  # Different port to avoid conflict
        
        print(f"\n🌐 Server running at:")
        print(f"   Local:    http://localhost:{port}")
        print(f"   Network:  http://{local_ip}:{port}")
        
        print(f"\n📱 How to use:")
        print(f"   1. Open browser → http://localhost:{port}")
        print(f"   2. Stand in front of camera")
        print(f"   3. See pose detection in action! 🤸‍♂️")
        print(f"   4. No toggle needed - always active")
        
        print(f"{'='*50}\n")
        
        # Run Flask server
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure you have:")
        print(f"   pip install flask opencv-python mediapipe")
    finally:
        if pose_server.cap:
            pose_server.cap.release()
