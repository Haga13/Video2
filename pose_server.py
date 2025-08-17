#!/usr/bin/env python3
"""
Pose Detection Web Server
Server sederhana untuk menjalankan pose detection melalui web interface
"""

import cv2
import threading
import time
from flask import Flask, Response, render_template, jsonify
import socket
import queue
import numpy as np

# Import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe tersedia")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("❌ MediaPipe tidak tersedia - Install dengan: pip install mediapipe")
    mp = None

app = Flask(__name__)

class PoseDetectionServer:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.cap = None
        self.running = False
        self.pose_detection_active = False
        
        # Setup MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
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
        
    def start_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("✅ Camera initialized")
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def process_frame(self, frame):
        """Process frame with pose detection"""
        if not self.pose_detection_active or not MEDIAPIPE_AVAILABLE or not self.pose:
            # Return frame without pose detection overlay
            return frame
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            frame_with_pose = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_with_pose,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
             
            
            return frame_with_pose
            
        except Exception as e:
            print(f"Pose processing error: {e}")
            cv2.putText(frame, f'POSE ERROR: {str(e)[:20]}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame
    
    def capture_frames(self):
        """Frame capture thread"""
        print("📹 Starting frame capture...")
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self.start_camera():
                        time.sleep(1)
                        continue
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Process with pose detection
                frame = self.process_frame(frame)
                
                # Add FPS
                self.update_fps()
                cv2.putText(frame, f'FPS: {self.current_fps}', (frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Update frame queue
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
                    
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get latest frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def encode_frame(self, frame):
        """Encode frame to JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def start(self):
        """Start server"""
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
        print("✅ Pose detection server started")
    
    def stop(self):
        """Stop server"""
        self.running = False
        self.pose_detection_active = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("⏹️  Server stopped")
    
    def toggle_pose_detection(self):
        """Toggle pose detection on/off"""
        if not MEDIAPIPE_AVAILABLE:
            return False, "MediaPipe not available"
        
        self.pose_detection_active = not self.pose_detection_active
        status = "started" if self.pose_detection_active else "stopped"
        print(f"🤸 Pose detection {status}")
        return True, f"Pose detection {status}"
    
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

# Sensor data mock
sensor_data = {
    'temperature': 25.0,
    'humidity': 60.0,
    'timestamp': time.time()
}

def generate_frames():
    """Video frame generator"""
    while True:
        frame = pose_server.get_frame()
        if frame is not None:
            buffer = pose_server.encode_frame(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')
        else:
            time.sleep(0.01)

# Flask Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_mode', methods=['GET', 'POST'])
def toggle_mode():
    """Toggle pose detection mode"""
    success, message = pose_server.toggle_pose_detection()
    
    if success:
        detection_mode = 'pose' if pose_server.pose_detection_active else 'weapon'
        return jsonify({
            'status': 'success',
            'message': message,
            'detection_mode': detection_mode
        })
    else:
        return jsonify({
            'status': 'error',
            'message': message,
            'detection_mode': 'weapon'
        }), 500

@app.route('/status')
def status():
    """Get server status"""
    return jsonify({
        'status': 'running' if pose_server.running else 'stopped',
        'pose_detection': pose_server.pose_detection_active,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'fps': pose_server.current_fps
    })

@app.route('/get_sensor_data')
def get_sensor_data():
    """Get sensor data"""
    return jsonify({
        'temperature': sensor_data['temperature'],
        'humidity': sensor_data['humidity'],
        'timestamp': sensor_data['timestamp'],
        'last_update': time.time() - sensor_data['timestamp']
    })

def get_local_ip():
    """Get local IP"""
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
        print(f"🤸 POSE DETECTION WEB SERVER")
        print(f"{'='*50}")
        
        if MEDIAPIPE_AVAILABLE:
            print("✅ MediaPipe available - Pose detection ready!")
        else:
            print("❌ MediaPipe not available")
            print("   Install: pip install mediapipe")
        
        # Start server
        pose_server.start()
        
        # Get network info
        local_ip = get_local_ip()
        port = 5000
        
        print(f"\n🌐 Server running at:")
        print(f"   Local:    http://localhost:{port}")
        print(f"   Network:  http://{local_ip}:{port}")
        
        print(f"\n📱 How to use:")
        print(f"   1. Open browser → http://localhost:{port}")
        print(f"   2. Click 'SWITCH TO POSE MODE' button")
        print(f"   3. Stand in front of camera")
        print(f"   4. See pose detection in action! 🤸‍♂️")
        
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
        pose_server.stop()
