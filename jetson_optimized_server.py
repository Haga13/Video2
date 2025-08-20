#!/usr/bin/env python3
"""
Pose Detection Optimized for Jetson Nano
Server yang dioptimasi untuk performance tinggi di Jetson Nano
Target: 15-20 FPS dengan latency rendah
"""

import cv2
import threading
import time
import queue
from flask import Flask, Response, render_template, jsonify, request
import socket
import numpy as np

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

# CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

class OptimizedPoseServer:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)  # Smaller queue untuk reduce latency
        self.cap = None
        self.running = False
        self.pose_detection_active = True
        
        # Optimized settings untuk Jetson Nano
        self.frame_width = 320   # Reduced resolution
        self.frame_height = 240  # Reduced resolution
        self.target_fps = 15     # Target FPS untuk Jetson
        self.skip_frames = 2     # Process every 2nd frame only
        self.frame_counter = 0
        
        # Setup MediaPipe dengan optimasi
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Optimized MediaPipe configuration
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Lightest model
                enable_segmentation=False,
                min_detection_confidence=0.7,  # Higher confidence = less false positives
                min_tracking_confidence=0.7,   # Better tracking
                smooth_landmarks=True
            )
            print("✅ MediaPipe Pose initialized (Optimized)")
        else:
            self.pose = None
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processing_time = 0
        
        # Performance stats
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        
        # Start camera
        self.start_camera()
    
    def start_camera(self):
        """Start optimized camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Cannot open camera")
                return False
            
            # Optimized camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            # Try to set additional optimizations
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            self.running = True
            
            # Start optimized capture thread
            self.capture_thread = threading.Thread(target=self.optimized_capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            print(f"✅ Camera initialized ({self.frame_width}x{self.frame_height} @ {self.target_fps}fps)")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def optimized_capture_frames(self):
        """Optimized frame capture with frame skipping"""
        last_process_time = time.time()
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.total_frames += 1
                    current_time = time.time()
                    
                    # Frame skipping untuk performance
                    self.frame_counter += 1
                    if self.frame_counter % (self.skip_frames + 1) != 0:
                        self.skipped_frames += 1
                        continue
                    
                    # Adaptive frame skipping berdasarkan processing time
                    if self.processing_time > 0.1:  # If processing too slow
                        if self.frame_counter % 3 != 0:  # Skip more frames
                            self.skipped_frames += 1
                            continue
                    
                    # Process frame
                    start_time = time.time()
                    processed_frame = self.process_frame(frame)
                    self.processing_time = time.time() - start_time
                    
                    self.processed_frames += 1
                    
                    # Non-blocking queue put
                    try:
                        self.frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        # Drop old frame if queue full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(processed_frame)
                        except queue.Empty:
                            pass
                    
                    self.update_fps()
                    
                    # Adaptive sleep untuk maintain target FPS
                    frame_time = 1.0 / self.target_fps
                    elapsed = time.time() - current_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                        
                else:
                    print("❌ Failed to read frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Frame capture error: {e}")
                time.sleep(0.1)
    
    def process_frame(self, frame):
        """Optimized frame processing"""
        if not MEDIAPIPE_AVAILABLE or not self.pose:
            return frame
        
        try:
            # Resize untuk processing yang lebih cepat
            h, w = frame.shape[:2]
            if w > self.frame_width:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Simplified drawing untuk performance
            if results.pose_landmarks:
                # Draw minimal landmarks (hanya key points)
                self.draw_optimized_pose(bgr_frame, results.pose_landmarks)
                
                # Simple text overlay
                cv2.putText(bgr_frame, 'POSE OK', (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(bgr_frame, 'No pose', (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Performance info
            cv2.putText(bgr_frame, f'FPS:{self.current_fps}', (5, bgr_frame.shape[0] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(bgr_frame, f'Proc:{self.processing_time*1000:.0f}ms', (5, bgr_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return bgr_frame
            
        except Exception as e:
            print(f"❌ Pose processing error: {e}")
            return frame
    
    def draw_optimized_pose(self, image, landmarks):
        """Draw minimal pose landmarks untuk performance"""
        h, w, _ = image.shape
        
        # Key points only (reduced drawing)
        key_points = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        # Draw key points only
        for landmark_id in key_points:
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > 0.5:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        # Draw key connections only
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        for connection in connections:
            start_landmark = landmarks.landmark[connection[0]]
            end_landmark = landmarks.landmark[connection[1]]
            
            if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.total_frames > 0:
            processing_ratio = (self.processed_frames / self.total_frames) * 100
            skip_ratio = (self.skipped_frames / self.total_frames) * 100
        else:
            processing_ratio = 0
            skip_ratio = 0
            
        return {
            'fps': self.current_fps,
            'processing_time_ms': self.processing_time * 1000,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.skipped_frames,
            'processing_ratio': processing_ratio,
            'skip_ratio': skip_ratio
        }

# Global server instance
pose_server = OptimizedPoseServer()

# Mock sensor data
sensor_data = {
    'temperature': 25.0,
    'humidity': 60.0,
    'timestamp': time.time()
}

def generate_frames():
    """Optimized video frame generator"""
    while True:
        if not pose_server.frame_queue.empty():
            frame = pose_server.frame_queue.get()
            
            # Encode dengan quality optimized untuk speed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality for speed
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/')
def index():
    """Main page"""
    return render_template('jetson_optimized.html')

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Get server status with performance stats"""
    stats = pose_server.get_performance_stats()
    return jsonify({
        'status': 'running' if pose_server.running else 'stopped',
        'pose_detection': pose_server.pose_detection_active,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'fps': stats['fps'],
        'processing_time_ms': stats['processing_time_ms'],
        'performance': {
            'total_frames': stats['total_frames'],
            'processed_frames': stats['processed_frames'],
            'skipped_frames': stats['skipped_frames'],
            'processing_ratio': stats['processing_ratio'],
            'skip_ratio': stats['skip_ratio']
        },
        'mode': 'jetson_optimized'
    })

@app.route('/get_sensor_data')
def get_sensor_data():
    """Get sensor data"""
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
        print(f"\n{'='*60}")
        print(f"🚀 POSE DETECTION - JETSON NANO OPTIMIZED")
        print(f"{'='*60}")
        
        if MEDIAPIPE_AVAILABLE:
            print("✅ MediaPipe available - Optimized for Jetson Nano")
            print(f"🎯 Target: {pose_server.target_fps} FPS")
            print(f"📏 Resolution: {pose_server.frame_width}x{pose_server.frame_height}")
            print(f"⚡ Frame skipping: Every {pose_server.skip_frames + 1} frames")
        else:
            print("❌ MediaPipe not available")
            print("💡 Install dengan: pip install mediapipe")
        
        print(f"\n📹 Starting optimized capture...")
        print(f"✅ Jetson optimized server started")
        
        # Get network info
        local_ip = get_local_ip()
        port = 5002  # Different port
        
        print(f"\n🌐 Server running at:")
        print(f"   Local:    http://localhost:{port}")
        print(f"   Network:  http://{local_ip}:{port}")
        
        print(f"\n📱 Jetson Nano Optimizations:")
        print(f"   • Reduced resolution: {pose_server.frame_width}x{pose_server.frame_height}")
        print(f"   • Frame skipping: {pose_server.skip_frames + 1}x")
        print(f"   • Minimal pose drawing")
        print(f"   • Optimized MediaPipe settings")
        print(f"   • Adaptive performance tuning")
        
        print(f"{'='*60}\n")
        
        # Run Flask server dengan optimasi
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, 
               use_reloader=False)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure you have:")
        print(f"   pip install flask opencv-python mediapipe")
    finally:
        if pose_server.cap:
            pose_server.cap.release()
