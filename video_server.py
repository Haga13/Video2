#!/usr/bin/env python3
"""
Video Streaming Server
Menggunakan Flask dan OpenCV untuk streaming video dengan latency rendah
"""

import cv2
import threading
import time
from flask import Flask, Response, render_template, jsonify, request
import socket
import queue
import numpy as np
import base64

app = Flask(__name__)

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
        """Thread untuk mengambil frame dari kamera"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
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
        
        print("Video streamer dimulai")
    
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
        'resolution': f"{streamer.frame_width}x{streamer.frame_height}"
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
        
        print(f"\n{'='*50}")
        print(f"VIDEO STREAMING SERVER")
        print(f"{'='*50}")
        print(f"Server berjalan di:")
        print(f"  Local:    http://localhost:{port}")
        print(f"  Network:  http://{local_ip}:{port}")
        print(f"\nUntuk mengakses dari perangkat lain di jaringan yang sama,")
        print(f"gunakan: http://{local_ip}:{port}")
        print(f"{'='*50}\n")
        
        # Jalankan Flask server
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nMenghentikan server...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
